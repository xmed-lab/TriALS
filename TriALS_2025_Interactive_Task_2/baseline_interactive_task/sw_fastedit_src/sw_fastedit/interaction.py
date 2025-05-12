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

import logging
import time
from typing import Callable, Dict, Sequence, Union

import nibabel as nib
import numpy as np
import torch
from monai.data import decollate_batch, list_data_collate
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.engines.utils import IterationEvents
from monai.losses import DiceLoss
from monai.transforms import Compose
from monai.utils.enums import CommonKeys
from monai.metrics import DiceMetric

from sw_fastedit.click_definitions import ClickGenerationStrategy, StoppingCriterion
from sw_fastedit.utils.helper import get_gpu_usage, timeit

logger = logging.getLogger("sw_fastedit")

# WARNING Code is not tested on batch_size > 1!!!!!!!!!!!!


class Interaction:
    """
    Ignite process_function used to introduce interactions (simulation of clicks) for DeepEdit Training/Evaluation.

    More details about this can be found at:

        Hadlich, Matthias*, Marinov, Zdravko* et al. "Sliding window fastedit: A framework for lesion annotation in whole-body pet images." 
        2024 IEEE International Symposium on Biomedical Imaging (ISBI). IEEE, 2024.
        https://ieeexplore.ieee.org/document/10635459/

    The code is based on:
        Diaz-Pinto et al., MONAI Label: A framework for AI-assisted Interactive
        Labeling of 3D Medical Images. (2022) https://arxiv.org/abs/2203.12362

    Args:
        transforms: execute additional transformation during every iteration (before train).
            Typically, several Tensor based transforms composed by `Compose`.
        train: True for training mode or False for evaluation mode
        label_names: Dict of label names
        max_interactions: maximum number of interactions per iteration
        click_probability_key: key to click/interaction probability
        click_generation_strategy_key: which key to use for storing the `ClickGenerationStrategy` in the batchdata
        click_generation_strategy: used to select the according `ClickGenerationStrategy`, which decides how clicks are generated
        stopping_criterion: used to select the `StoppingCriterion`, which decides when the click generation is stopped. This may be
            max interaction based, loss based, or completely different. Look into `StoppingCriterion` definition for details.
        iteration_probability: parameter for the `StoppingCriterion`. States after how many iterations the click generation is stopped
        loss_stopping_threshold: parameter for the `StoppingCriterion`. States at which optimal loss the click generation is stopped.
            Usually used in combination with `iteration_probability`, to have a hard upper bound on the amount of clicks.
        deepgrow_probability: probability of simulating clicks in an iteration
        save_nifti: whether to save nifti files to debug the code
        nifti_dir: location where to store the debug nifti files
        nifti_post_transform: post transforms to be run before the information is stored into the nifti files
        loss_function: loss_function to the ran after every interaction to determine if the clicks actually help the model
        non_interactive: set it for non-interactive runs, where no clicks shall be added. The Interaction class only prints the
            shape of image and label, then resumes normal training.
    """

    def __init__(
        self,
        transforms: Union[Sequence[Callable], Callable],
        train: bool,
        label_names: Union[None, Dict[str, int]] = None,
        max_interactions: int = 1,
        *,
        click_probability_key: str = "probability",
        click_generation_strategy_key: str = "click_generation_strategy",
        click_generation_strategy: ClickGenerationStrategy = ClickGenerationStrategy.GLOBAL_CORRECTIVE,
        stopping_criterion: StoppingCriterion = StoppingCriterion.MAX_ITER,
        iteration_probability: float = 0.5,
        loss_stopping_threshold: float = 0.1,
        deepgrow_probability: float = 1.0,
        save_nifti=False,
        nifti_dir=None,
        nifti_post_transform=None,
        loss_function=None,
        non_interactive=False,
    ) -> None:
        self.deepgrow_probability = deepgrow_probability
        self.transforms = Compose(transforms) if not isinstance(transforms, Compose) else transforms  # click transforms

        self.train = train
        self.label_names = label_names
        self.click_probability_key = click_probability_key
        self.max_interactions = max_interactions
        self.save_nifti = save_nifti
        self.nifti_dir = nifti_dir
        self.loss_function = loss_function
        self.nifti_post_transform = nifti_post_transform
        self.click_generation_strategy = click_generation_strategy
        self.stopping_criterion = stopping_criterion
        self.iteration_probability = iteration_probability
        self.loss_stopping_threshold = loss_stopping_threshold
        self.click_generation_strategy_key = click_generation_strategy_key
        self.dice_loss_function = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        self.non_interactive = non_interactive

    @timeit
    def __call__(
        self,
        engine: Union[SupervisedTrainer, SupervisedEvaluator],
        batchdata: Dict[str, torch.Tensor],
    ):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        if not self.train:
            # Evaluation does not print epoch / iteration information
            logger.info(
                (
                    f"### Interaction iteration {((engine.state.iteration - 1) % engine.state.epoch_length) + 1}"
                    f"/{engine.state.epoch_length}"
                )
            )
        logger.info(
            get_gpu_usage(
                device=engine.state.device,
                used_memory_only=True,
                context="START interaction class",
            )
        )
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        val_metrics = {'dsc': [], 'fpv': [], 'fnv': []}


        iteration = 0
        last_dice_loss = 1
        before_it = time.time()
        while True:
            assert iteration < 1000
            # Shape for interactive image e.g. (num_classes+1)x192x192x256, label 1x192x192x256
            # BCHWD
            inputs, labels = engine.prepare_batch(batchdata, device=engine.state.device)
            batchdata[CommonKeys.IMAGE] = inputs
            batchdata[CommonKeys.LABEL] = labels

            if iteration == 0:
                logger.info("inputs.shape is {}".format(inputs.shape))
                logger.info("labels.shape is {}".format(labels.shape))
                # Make sure the signal is empty in the first iteration assertion holds
                
                #assert torch.sum(inputs[:, 1:, ...]) == 0
                logger.info(f"image file name: {batchdata['image_meta_dict']['filename_or_obj']}")
                logger.info(f"label file name: {batchdata['label_meta_dict']['filename_or_obj']}")
                #iteration += 1
                #continue
                for i in range(len(batchdata["label"][0])):
                    if torch.sum(batchdata["label"][i, 0]) < 0.1:
                        logger.warning("No valid labels for this sample (probably due to crop)")

            if self.non_interactive:
                break

            if self.stopping_criterion in [
                StoppingCriterion.MAX_ITER,
                StoppingCriterion.MAX_ITER_AND_PROBABILITY,
                StoppingCriterion.MAX_ITER_AND_DICE,
                StoppingCriterion.MAX_ITER_PROBABILITY_AND_DICE,
                StoppingCriterion.DEEPGROW_PROBABILITY,
            ]:
                # Abort if run for max_interactions
                if iteration > self.max_interactions - 1:

                    logger.info("MAX_ITER stop")
                    break
            if self.stopping_criterion in [StoppingCriterion.MAX_ITER_AND_PROBABILITY]:
                # Abort based on the per iteration probability
                if not np.random.choice(
                    [True, False],
                    p=[self.iteration_probability, 1 - self.iteration_probability],
                ):
                    logger.info("PROBABILITY stop")
                    break
            if self.stopping_criterion in [StoppingCriterion.MAX_ITER_AND_DICE]:
                # Abort if dice / loss is good enough
                if last_dice_loss < self.loss_stopping_threshold:
                    logger.info(f"DICE stop, since {last_dice_loss} < {self.loss_stopping_threshold}")
                    break

            if self.stopping_criterion in [
                StoppingCriterion.MAX_ITER_PROBABILITY_AND_DICE,
            ]:
                if np.random.choice([True, False], p=[1 - last_dice_loss, last_dice_loss]):
                    logger.info(f"DICE_PROBABILITY stop, since dice loss is already {last_dice_loss}")
                    break

            if iteration == 0 and self.stopping_criterion == StoppingCriterion.DEEPGROW_PROBABILITY:
                # Abort before the first iteration if deepgrow_probability yields False
                if not np.random.choice(
                    [True, False],
                    p=[self.deepgrow_probability, 1 - self.deepgrow_probability],
                ):
                    break

            engine.fire_event(IterationEvents.INNER_ITERATION_STARTED)
            engine.network.eval()


            if iteration == 1:
                # Forward Pass            
  
                with torch.no_grad():
                    if engine.amp:
                        with torch.cuda.amp.autocast():
                            predictions = engine.inferer(inputs, engine.network)
                    else:
                        predictions = engine.inferer(inputs, engine.network)

                batchdata[CommonKeys.PRED] = predictions


                last_dice_loss = self.dice_loss_function(batchdata[CommonKeys.PRED], batchdata[CommonKeys.LABEL]).item()
                logger.info(
                    f"It: {iteration} {self.dice_loss_function.__class__.__name__}: {last_dice_loss:.4f} Epoch: {engine.state.epoch}"
                )
                if self.save_nifti:
                    self.debug_viz(inputs, batchdata[CommonKeys.LABEL], batchdata[CommonKeys.PRED], iteration)
                    exit()

            # decollate/collate batchdata to execute click transforms
            batchdata_list = decollate_batch(batchdata)
            for i in range(len(batchdata_list)):
                batchdata_list[i][self.click_probability_key] = self.deepgrow_probability
                batchdata_list[i][self.click_generation_strategy_key] = self.click_generation_strategy.value
                start = time.time()
                batchdata_list[i] = self.transforms(batchdata_list[i])  # Apply click transform
                logger.debug(f"Click transform took: {time.time() - start:.2} seconds")



            batchdata = list_data_collate(batchdata_list)

            engine.fire_event(IterationEvents.INNER_ITERATION_COMPLETED)

            iteration += 1

        logger.debug(f"Interaction took {time.time()- before_it:.2f} seconds..")
        engine.state.batch = batchdata
        return engine._iteration(engine, batchdata)  # train network with the final iteration cycle

    def debug_viz(self, inputs, labels, preds, j):
        self.save_nifti_file(f"{self.nifti_dir}/im", inputs[0, 0].cpu().detach().numpy())
        self.save_nifti_file(f"{self.nifti_dir}/guidance_fgg_{j}", inputs[0, 1].cpu().detach().numpy())
        self.save_nifti_file(f"{self.nifti_dir}/guidance_bgg_{j}", inputs[0, 2].cpu().detach().numpy())
        self.save_nifti_file(f"{self.nifti_dir}/labels", labels[0, 0].cpu().detach().numpy())
        self.save_nifti_file(f"{self.nifti_dir}/pred_{j}", preds[0, 1].cpu().detach().numpy())

    def save_nifti_file(self, name, im):
        affine = np.eye(4)
        affine[0][0] = -1
        ni_img = nib.Nifti1Image(im, affine=affine)
        ni_img.header.get_xyzt_units()
        ni_img.to_filename(f"{name}.nii.gz")
