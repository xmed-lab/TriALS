from __future__ import annotations

import logging
import time
from typing import Hashable, Iterable, Mapping
import gc

import torch
from monai.config import KeysCollection
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    MapTransform,
    Transform,
)

from sw_fastedit.click_definitions import LABELS_KEY
from sw_fastedit.utils.helper import (  # convert_nii_to_mha,; convert_mha_to_nii,
    describe_batch_data,
)
from sw_fastedit.utils.logger import get_logger, setup_loggers

logger = None

cast_labels_to_zero_and_one = lambda x: torch.where(x > 0, 1, 0)
class PrintImageShape(Transform):
    def __call__(self, data):
        image = data["image"]
        print("Image shape:", image.shape)
        return data  # Return the data unchanged

def threshold_foreground(x):
    return (x > 0.005) & (x < 0.995)


class AbortifNaNd(MapTransform):
    def __init__(self, keys: KeysCollection = None):
        """
        A transform which does nothing
        """
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        for key in self.key_iterator(data):
            assert not torch.isnan(data[key]).any()

        return data


class TrackTimed(Transform):
    def __init__(self, transform):
        """
        A transform which does nothing
        """
        super().__init__()
        # self.keys = keys
        self.transform = transform

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        global logger
        start_time = time.perf_counter()
        data = self.transform(data)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"-------- {self.transform.__class__.__qualname__:<20.20}() took {total_time:.3f} seconds")
        # print(f"{self.transform.__class__.__qualname__}() took {total_time:.3f} seconds")

        return data



class CheckTheAmountOfInformationLossByCropd(MapTransform):
    def __init__(self, keys: KeysCollection, roi_size: Iterable, crop_foreground=True):
        """
        Prints how much information is lost due to the crop.
        """
        super().__init__(keys)
        self.roi_size = roi_size
        self.crop_foreground = crop_foreground

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        labels = data[LABELS_KEY]
        for key in self.key_iterator(data):
            if key == "label":
                t = []
                if self.crop_foreground:
                    t.append(
                        CropForegroundd(
                            keys=("image", "label"),
                            source_key="image",
                            select_fn=threshold_foreground,
                        )
                    )
                if self.roi_size is not None:
                    t.append(CenterSpatialCropd(keys="label", roi_size=self.roi_size))

                if len(t):
                    # copy the label and crop it to the desired size
                    label = data[key]
                    new_data = {"label": label.clone(), "image": data["image"].clone()}

                    cropped_label = Compose(t)(new_data)["label"]

                    # label_num_el = torch.numel(label)
                    for idx, (key_label, _) in enumerate(labels.items(), start=1):
                        # Only count non-background lost labels
                        if key_label != "background":
                            sum_label = torch.sum(label == idx).item()
                            sum_cropped_label = torch.sum(cropped_label == idx).item()
                            # then check how much of the labels is lost
                            lost_pixels = sum_label - sum_cropped_label
                            if sum_label != 0:
                                lost_pixels_ratio = lost_pixels / sum_label * 100
                                logger.info(
                                    f"{lost_pixels_ratio:.1f} % of labelled pixels of the type {key_label} have been lost when cropping"
                                )
                            else:
                                logger.info("No labeled pixels found for current image")
                                logger.debug(f"image {data['image_meta_dict']['filename_or_obj']}")
            else:
                raise UserWarning("This transform only applies to key 'label'")
        return data


class PrintDatad(MapTransform):
    def __init__(
        self,
        keys: KeysCollection = None,
        allow_missing_keys: bool = False,
    ):
        """
        Prints all the information inside data
        """
        super().__init__(keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        global logger
        if self.keys is not None and (None not in self.keys and len(self.keys) == 1):
            data_sub_dict = {key: data[key] for key in self.key_iterator(data)}
        else:
            data_sub_dict = data

        try:
            logger.info(describe_batch_data(data_sub_dict))
        except UnboundLocalError:
            logger = logging.getLogger("sw_fastedit")
            logger.info(describe_batch_data(data_sub_dict))

        return data


class PrintGPUUsaged(MapTransform):
    def __init__(self, device, keys: KeysCollection = None, name=""):
        """
        Prints the GPU usage
        """
        super().__init__(keys)
        self.device = device
        self.name = name

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:        
        if logger is not None:
            logger.info(
                f"{self.name}::Current reserved memory for dataloader: {torch.cuda.memory_reserved(self.device) / (1024**3)} GB"
            )
        return data


class ClearGPUMemoryd(MapTransform):
    def __init__(self, device, keys: KeysCollection = None, garbage_collection: bool = True):
        """
        Prints the GPU usage
        """
        super().__init__(keys)
        self.device = device
        self.garbage_collection = garbage_collection

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        if self.garbage_collection:
            gc.collect()
        torch.cuda.empty_cache()
        if logger is not None:
            logger.info(
                f"Current reserved memory for dataloader: {torch.cuda.memory_reserved(self.device) / (1024**3)} GB"
            )
        return data


class InitLoggerd(MapTransform):
    def __init__(self, loglevel=logging.INFO, no_log=True, log_dir=None):
        """
        Initialises the logger inside the dataloader thread (if it is a separate thread).
        This is only necessary if the data loading is done in multiple threads / processes.

        Otherwise no log messages get print.
        """
        global logger
        super().__init__(None)

        self.loglevel = loglevel
        self.log_dir = log_dir
        self.no_log = no_log

        if self.no_log:
            self.log_dir = None

        setup_loggers(self.loglevel, self.log_dir)
        logger = get_logger()

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        global logger
        if logger is None:
            setup_loggers(self.loglevel, self.log_dir)
        logger = get_logger()
        return data
