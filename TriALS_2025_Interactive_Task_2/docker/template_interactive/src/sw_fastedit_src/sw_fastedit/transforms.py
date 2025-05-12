# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gc
import logging
from typing import Dict, Hashable, List, Mapping, Tuple

import torch
import numpy as np
from monai.config import KeysCollection
from monai.data import MetaTensor, PatchIterd
from monai.losses import DiceLoss
from monai.networks.layers import GaussianFilter
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    MapTransform,
    Randomizable,
)
from monai.utils.enums import CommonKeys

from sw_fastedit.click_definitions import LABELS_KEY, ClickGenerationStrategy
from sw_fastedit.utils.distance_transform import get_random_choice_from_tensor
from monai.transforms.utils import distance_transform_edt
from sw_fastedit.utils.helper import get_global_coordinates_from_patch_coordinates, get_tensor_at_coordinates, timeit


logger = logging.getLogger("sw_fastedit")


def get_guidance_tensor_for_key_label(data, key_label, device) -> torch.Tensor:
    """Makes sure the guidance is in a tensor format."""
    tmp_gui = data.get(key_label, torch.tensor([], dtype=torch.int32, device=device))
    if isinstance(tmp_gui, list):
        tmp_gui = torch.tensor(tmp_gui, dtype=torch.int32, device=device)
    assert type(tmp_gui) is torch.Tensor or type(tmp_gui) is MetaTensor
    return tmp_gui



class AddEmptySignalChannels(MapTransform):
    def __init__(self, device, keys: KeysCollection = None):
        """
        Adds empty channels to the signal which will be filled with the guidance signal later.
        E.g. for two labels: 1x192x192x256 -> 3x192x192x256
        """
        super().__init__(keys)
        self.device = device

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        # Set up the initial batch data
        in_channels = 1 + len(data[LABELS_KEY])
        tmp_image = data[CommonKeys.IMAGE][0 : 0 + 1, ...]
        assert len(tmp_image.shape) == 4
        new_shape = list(tmp_image.shape)
        new_shape[0] = in_channels
        # Set the signal to 0 for all input images
        # image is on channel 0 of e.g. (1,128,128,128) and the signals get appended, so
        # e.g. (3,128,128,128) for two labels
        inputs = torch.zeros(new_shape, device=self.device)
        inputs[0] = data[CommonKeys.IMAGE][0]
        if isinstance(data[CommonKeys.IMAGE], MetaTensor):
            data[CommonKeys.IMAGE].array = inputs
        else:
            data[CommonKeys.IMAGE] = inputs

        return data


class NormalizeLabelsInDatasetd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        labels=None,
        allow_missing_keys: bool = False,
        device=None,
    ):
        """
        Normalize label values according to label names dictionary

        Args:
            keys: the ``keys`` parameter will be used to get and set the actual data item to transform
            labels: all label names
            allow_missing_keys: whether to ignore it if keys are missing.
            device: device this transform shall run on

        Returns: data and also the new labels will be stored in data with key LABELS_KEY
        """
        super().__init__(keys, allow_missing_keys)
        self.labels = labels
        self.device = device

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        # Set the labels dict in case no labels were provided
        data[LABELS_KEY] = self.labels
        
        for key in self.key_iterator(data):
            if key == "label":
                try:
                    label = data[key]
                    if isinstance(label, str):
                        # Special case since label has been defined to be a string in MONAILabel
                        raise AttributeError
                except AttributeError:
                    # label does not exist - this might be a validation run
                    break

                # Dictionary containing new label numbers
                new_labels = {}
                label = torch.zeros(data[key].shape, device=self.device)
                # Making sure the range values and number of labels are the same

                for idx, (key_label, val_label) in enumerate(self.labels.items(), start=1):
                    if key_label != "background":
                        new_labels[key_label] = idx
                        label[data[key] == val_label] = idx
                    if key_label == "background":
                        new_labels["background"] = 0
                    else:
                        new_labels[key_label] = idx
                        label[data[key] == val_label] = idx

                data[LABELS_KEY] = new_labels
                if isinstance(data[key], MetaTensor):
                    data[key].array = label
                else:
                    data[key] = label
            else:
                raise UserWarning("Only the key label is allowed here!")
        return data


class AddGuidanceSignal(MapTransform):
    """
    Add Guidance signal for input image.

    Based on the "guidance" points, apply Gaussian to them and add them as new channel for input image.

    Args:
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
        disks: This paraemters fill spheres with a radius of sigma centered around each click.
        device: device this transform shall run on.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma: int = 1,
        number_intensity_ch: int = 1,
        allow_missing_keys: bool = False,
        disks: bool = False,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch
        self.disks = disks
        self.device = device

    def _get_corrective_signal(self, image, guidance, key_label):
        dimensions = 3 if len(image.shape) > 3 else 2
        assert (
            type(guidance) is torch.Tensor or type(guidance) is MetaTensor
        ), f"guidance is {type(guidance)}, value {guidance}"

        if guidance.size()[0]:
            first_point_size = guidance[0].numel()
            if dimensions == 3:
                # Assume channel is first and depth is last CHWD
                # Assuming the guidance has either shape (1, x, y , z) or (x, y, z)
                assert (
                    first_point_size == 4 or first_point_size == 3
                ), f"first_point_size is {first_point_size}, first_point is {guidance[0]}"
                signal = torch.zeros(
                    (1, image.shape[-3], image.shape[-2], image.shape[-1]),
                    device=self.device,
                )
            else:
                assert first_point_size == 3, f"first_point_size is {first_point_size}, first_point is {guidance[0]}"
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)

            sshape = signal.shape

            for point in guidance:
                if torch.any(point < 0):
                    continue
                if dimensions == 3:
                    # Making sure points fall inside the image dimension
                    p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                    p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2, p3] = 1.0
                else:
                    p1 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p2 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2] = 1.0

            # Apply a Gaussian filter to the signal
            if torch.max(signal[0]) > 0:
                signal_tensor = signal[0]
                if self.sigma != 0:
                    pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                    signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                    signal_tensor = signal_tensor.squeeze(0).squeeze(0)

                signal[0] = signal_tensor
                signal[0] = (signal[0] - torch.min(signal[0])) / (torch.max(signal[0]) - torch.min(signal[0]))
                if self.disks:
                    signal[0] = (signal[0] > 0.1) * 1.0  # 0.1 with sigma=1 --> radius = 3, otherwise it is a cube

            if not (torch.min(signal[0]).item() >= 0 and torch.max(signal[0]).item() <= 1.0):
                raise UserWarning(
                    "[WARNING] Bad signal values",
                    torch.min(signal[0]),
                    torch.max(signal[0]),
                )
            if signal is None:
                raise UserWarning("[ERROR] Signal is None")
            return signal
        else:
            if dimensions == 3:
                signal = torch.zeros(
                    (1, image.shape[-3], image.shape[-2], image.shape[-1]),
                    device=self.device,
                )
            else:
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)
            if signal is None:
                print("[ERROR] Signal is None")
            return signal

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        for key in self.key_iterator(data):
            if key == "image":
                image = data[key]
                assert image.is_cuda
                tmp_image = image[0 : 0 + self.number_intensity_ch, ...]

                # e.g. {'spleen': '[[1, 202, 190, 192], [2, 224, 212, 192], [1, 242, 202, 192], [1, 256, 184, 192], [2.0, 258, 198, 118]]',
                # 'background': '[[257, 0, 98, 118], [1.0, 223, 303, 86]]'}

                for _, (label_key, _) in enumerate(data[LABELS_KEY].items()):
                    # label_guidance = data[label_key]
                    label_guidance = get_guidance_tensor_for_key_label(data, label_key, self.device)
                    logger.debug(f"Converting guidance for label {label_key}:{label_guidance} into a guidance signal..")
                    if label_guidance is not None and label_guidance.numel():
                        signal = self._get_corrective_signal(
                            image,
                            label_guidance.to(device=self.device),
                            key_label=label_key,
                        )
                        assert torch.sum(signal) > 0
                    else:
                        # TODO can speed this up here
                        signal = self._get_corrective_signal(
                            image,
                            torch.Tensor([]).to(device=self.device),
                            key_label=label_key,
                        )

                    assert signal.is_cuda
                    assert tmp_image.is_cuda
                    if len(signal.shape) == len(tmp_image.shape) - 1:
                        signal = signal.unsqueeze(0)
                    tmp_image = torch.cat([tmp_image, signal], dim=0)
                    if isinstance(data[key], MetaTensor):
                        data[key].array = tmp_image
                    else:
                        data[key] = tmp_image

                return data
            else:
                raise UserWarning("This transform only applies to image key")
        raise UserWarning("image key has not been been found")


class FindDiscrepancyRegions(MapTransform):
    """
    Find discrepancy between prediction and actual during click interactions during training.

    Args:
        pred_key: key to prediction source.
        discrepancy_key: key to store discrepancies found between label and prediction.
        device: device this transform shall run on.
    """

    def __init__(
        self,
        keys: KeysCollection,
        pred_key: str = "pred",
        discrepancy_key: str = "discrepancy",
        allow_missing_keys: bool = False,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.pred_key = pred_key
        self.discrepancy_key = discrepancy_key
        self.device = device

    def disparity(self, label, pred):
        disparity = label - pred
        # +1 means predicted label is not part of the ground truth
        # -1 means predicted label missed that region of the ground truth
        pos_disparity = (disparity > 0).to(dtype=torch.float32, device=self.device)  # FN
        neg_disparity = (disparity < 0).to(dtype=torch.float32, device=self.device)  # FP
        return [pos_disparity, neg_disparity]

    def _apply(self, label, pred):
        return self.disparity(label, pred)

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        for key in self.key_iterator(data):
            if key == "label":
                assert (
                    (type(data[key]) is torch.Tensor)
                    or (type(data[key]) is MetaTensor)
                    and (type(data[self.pred_key]) is torch.Tensor or type(data[self.pred_key]) is MetaTensor)
                )
                all_discrepancies = {}
                assert data[key].is_cuda and data["pred"].is_cuda

                for _, (label_key, label_value) in enumerate(data[LABELS_KEY].items()):
                    if label_key != "background":
                        label = torch.clone(data[key].detach())
                        # Label should be represented in 1
                        label[label != label_value] = 0
                        label = (label > 0.5).to(dtype=torch.float32)

                        # Taking single prediction
                        pred = torch.clone(data[self.pred_key].detach())
                        pred[pred != label_value] = 0
                        # Prediction should be represented in one
                        pred = (pred > 0.5).to(dtype=torch.float32)
                    else:
                        # Taking single label
                        label = torch.clone(data[key].detach())

                        label[label != label_value] = 1
                        label = 1 - label
                        # Label should be represented in 1
                        label = (label > 0.5).to(dtype=torch.float32)
                        # Taking single prediction
                        pred = torch.clone(data[self.pred_key].detach())
                        pred[pred != label_value] = 1
                        pred = 1 - pred
                        # Prediction should be represented in one
                        pred = (pred > 0.5).to(dtype=torch.float32)
                    all_discrepancies[label_key] = self._apply(label, pred)
                data[self.discrepancy_key] = all_discrepancies
                return data
            else:
                logger.error("This transform only applies to 'label' key")
        raise UserWarning

def add_crop_offset(coord, offset):
    offset = offset.cpu().numpy().astype(np.uint16)
    if len(offset.shape) > 1:
        offset = offset[0] 
    new_coord = [0, coord[1] - offset[0], coord[2] - offset[1], coord[3] - offset[2]]
    return new_coord


class AddGuidanceJSON(Randomizable, MapTransform):
    """
    Add guidance based on pre-computed JSON file with click coordinates
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys = False,
        json_dir : str = None,
        n_clicks = 10,

    ):
        super().__init__(keys, allow_missing_keys)
        self.json_dir = json_dir
        self.n_clicks = n_clicks


    
    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:


        import os
        import json
        assert self.json_dir is not None


        for _, (key_label, _) in enumerate(data[LABELS_KEY].items()):
            json_fn = self.json_dir

            if type(json_fn) == str:
                assert os.path.exists(json_fn)

                with open(json_fn, 'r') as f:
                    json_data = json.load(f)


            elif type(json_fn) == dict:
                json_data = json_fn
            else:
                raise UserWarning
            fn_zoom = lambda x : [int(el * 0.8) for el in x]
            json_data[key_label] = [[0] + el for el in json_data[key_label][:self.n_clicks]]

            if 'foreground_start_coord' in data.keys():
                json_data[key_label] = [add_crop_offset(el, data['foreground_start_coord']) for el in json_data[key_label]]
                json_data[key_label] = [fn_zoom(el) for el in json_data[key_label][:self.n_clicks]]

            data[key_label] = torch.Tensor(json_data[key_label]).to(data['image'].device).long()

        return data
class AddGuidance(Randomizable, MapTransform):
    """
    Add guidance based on different click generation strategies.

    Args:
        discrepancy_key: key to discrepancy map between label and prediction shape (2, C, H, W, D) or (2, C, H, W)
        probability_key: key to click/interaction probability, shape (1)
        device: device this transform shall run on.
        click_generation_strategy_key: sets the used ClickGenerationStrategy.
        patch_size: Only relevant for the patch-based click generation strategy. Sets the size of the cropped patches
        on which then further analysis is run.
    """

    def __init__(
        self,
        keys: KeysCollection,
        discrepancy_key: str = "discrepancy",
        probability_key: str = "probability",
        allow_missing_keys: bool = False,
        device=None,
        click_generation_strategy_key: str = "click_generation_strategy",
        patch_size: Tuple[int] = (128, 128, 128),
    ):
        super().__init__(keys, allow_missing_keys)
        self.discrepancy_key = discrepancy_key
        self.probability_key = probability_key
        self._will_interact = None
        self.is_other = None
        self.default_guidance = None
        self.device = device
        self.click_generation_strategy_key = click_generation_strategy_key
        self.patch_size = patch_size

    def randomize(self, data: Mapping[Hashable, torch.Tensor]):
        probability = data[self.probability_key]
        self._will_interact = self.R.choice([True, False], p=[probability, 1.0 - probability])

    def find_guidance(self, discrepancy) -> List[int | List[int]] | None:
        assert discrepancy.is_cuda
        distance = distance_transform_edt(discrepancy)
        t_index, t_value = get_random_choice_from_tensor(distance)
        return t_index

    def add_guidance_based_on_discrepancy(
        self,
        data: Dict,
        guidance: torch.Tensor,
        key_label: str,
        coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert guidance.dtype == torch.int32
        # Positive clicks of the segment in the iteration
        discrepancy = data[self.discrepancy_key][key_label]
        # idx 0 is positive discrepancy and idx 1 is negative discrepancy
        pos_discr = discrepancy[0]

        if coordinates is None:
            # Add guidance to the current key label
            if torch.sum(pos_discr) > 0:
                tmp_gui = self.find_guidance(pos_discr)
                self.check_guidance_length(data, tmp_gui)
                if tmp_gui is not None:
                    guidance = torch.cat(
                        (
                            guidance,
                            torch.tensor([tmp_gui], dtype=torch.int32, device=guidance.device),
                        ),
                        0,
                    )
        else:
            pos_discr = get_tensor_at_coordinates(pos_discr, coordinates=coordinates)
            if torch.sum(pos_discr) > 0:
                # TODO Add suport for 2d
                tmp_gui = self.find_guidance(pos_discr)
                if tmp_gui is not None:
                    tmp_gui = get_global_coordinates_from_patch_coordinates(tmp_gui, coordinates)
                    self.check_guidance_length(data, tmp_gui)
                    guidance = torch.cat(
                        (
                            guidance,
                            torch.tensor([tmp_gui], dtype=torch.int32, device=guidance.device),
                        ),
                        0,
                    )
        return guidance

    def add_guidance_based_on_label(self, data, guidance, label):
        assert guidance.dtype == torch.int32
        # Add guidance to the current key label
        if torch.sum(label) > 0:
            assert label.is_cuda
            # generate a random sample
            tmp_gui_index, tmp_gui_value = get_random_choice_from_tensor(label)
            if tmp_gui_index is not None:
                self.check_guidance_length(data, tmp_gui_index)
                guidance = torch.cat(
                    (
                        guidance,
                        torch.tensor([tmp_gui_index], dtype=torch.int32, device=guidance.device),
                    ),
                    0,
                )
        return guidance

    def check_guidance_length(self, data, new_guidance):
        dimensions = 3 if len(data[CommonKeys.IMAGE].shape) > 3 else 2
        if dimensions == 3:
            assert len(new_guidance) == 4, f"len(new_guidance) is {len(new_guidance)}, new_guidance is {new_guidance}"
        else:
            assert len(new_guidance) == 3, f"len(new_guidance) is {len(new_guidance)}, new_guidance is {new_guidance}"

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        click_generation_strategy = data[self.click_generation_strategy_key]


        if click_generation_strategy == ClickGenerationStrategy.GLOBAL_NON_CORRECTIVE:
            # uniform random sampling on label
            for idx, (key_label, _) in enumerate(data[LABELS_KEY].items()):
                tmp_gui = get_guidance_tensor_for_key_label(data, key_label, self.device)
                data[key_label] = self.add_guidance_based_on_label(
                    data, tmp_gui, data["label"].eq(idx).to(dtype=torch.int32)
                )
        elif (
            click_generation_strategy == ClickGenerationStrategy.GLOBAL_CORRECTIVE
            or click_generation_strategy == ClickGenerationStrategy.DEEPGROW_GLOBAL_CORRECTIVE
        ):
            if click_generation_strategy == ClickGenerationStrategy.DEEPGROW_GLOBAL_CORRECTIVE:
                # sets self._will_interact
                self.randomize(data)
            else:
                self._will_interact = True

            if self._will_interact:
                for key_label in data[LABELS_KEY].keys():
                    tmp_gui = get_guidance_tensor_for_key_label(data, key_label, self.device)

                    # Add guidance based on discrepancy
                    data[key_label] = self.add_guidance_based_on_discrepancy(data, tmp_gui, key_label)
        elif click_generation_strategy == ClickGenerationStrategy.PATCH_BASED_CORRECTIVE:
            assert data[CommonKeys.LABEL].shape == data[CommonKeys.PRED].shape

            t = [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(
                    keys=("pred", "label"),
                    argmax=(True, False),
                    to_onehot=(len(data[LABELS_KEY]), len(data[LABELS_KEY])),
                ),
            ]
            post_transform = Compose(t)
            t_data = post_transform(data)

            # Split the data into patches of size self.patch_size
            # TODO not working for 2d data yet!
            new_data = PatchIterd(keys=[CommonKeys.PRED, CommonKeys.LABEL], patch_size=self.patch_size)(t_data)
            pred_list = []
            label_list = []
            coordinate_list = []

            for patch in new_data:
                actual_patch = patch[0]
                pred_list.append(actual_patch[CommonKeys.PRED])
                label_list.append(actual_patch[CommonKeys.LABEL])
                coordinate_list.append(actual_patch["patch_coords"])

            label_stack = torch.stack(label_list, 0)
            pred_stack = torch.stack(pred_list, 0)

            dice_loss = DiceLoss(include_background=True, reduction="none")
            with torch.no_grad():
                loss_per_label = dice_loss.forward(input=pred_stack, target=label_stack).squeeze()
                assert len(loss_per_label.shape) == 2
                # 1. dim: patch number, 2. dim: number of labels, e.g. [27,2]
                max_loss_position_per_label = torch.argmax(loss_per_label, dim=0)
                assert len(max_loss_position_per_label) == len(data[LABELS_KEY])

            # We now have the worst patches for each label, now sample clicks on them
            for idx, (key_label, _) in enumerate(data[LABELS_KEY].items()):
                patch_number = max_loss_position_per_label[idx]
                coordinates = coordinate_list[patch_number]

                tmp_gui = get_guidance_tensor_for_key_label(data, key_label, self.device)
                # Add guidance based on discrepancy
                data[key_label] = self.add_guidance_based_on_discrepancy(data, tmp_gui, key_label, coordinates)

            gc.collect()
        else:
            raise UserWarning("Unknown click strategy")
        return data


class SplitPredsLabeld(MapTransform):
    """
    Split preds and labels for individual evaluation
    """

    @timeit
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        for key in self.key_iterator(data):
            if key == "pred":
                for idx, (key_label, _) in enumerate(data[LABELS_KEY].items()):
                    if key_label != "background":
                        data[f"pred_{key_label}"] = data[key][idx + 1, ...][None]
                        data[f"label_{key_label}"] = data["label"][idx + 1, ...][None]
            elif key != "pred":
                logger.info("This transform is only for pred key")
        return data
