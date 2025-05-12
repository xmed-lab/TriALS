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

# Code extension and modification by M.Sc. Zdravko Marinov, Karlsuhe Institute of Techonology #
# zdravko.marinov@kit.edu #
# Further code extension and modification by B.Sc. Matthias Hadlich, Karlsuhe Institute of Techonology #
# matthiashadlich@posteo.de #

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import tempfile
import time
import uuid

import torch

from sw_fastedit.transforms import ClickGenerationStrategy
from sw_fastedit.click_definitions import StoppingCriterion
from sw_fastedit.utils.helper import get_actual_cuda_index_of_device, get_git_information, gpu_usage
from sw_fastedit.utils.logger import get_logger, setup_loggers


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-i", "--input_dir", help="Base folder for input images and labels")
    parser.add_argument("-o", "--output_dir", help="All the logs and weights will be stored here")
    parser.add_argument("--json_dir", required=False, help="Path to the click coordinates saved as json files")

    parser.add_argument(
        "-d", "--data_dir", default="None", help="Only used for debugging Niftii files, so usually not required"
    )
    # a subdirectory is created below cache_dir for every run
    parser.add_argument(
        "-c",
        "--cache_dir",
        type=str,
        default="None",
        help="Code uses a CacheDataset, so stores the transforms on the disk. This parameter is where the data gets stored.",
    )
    parser.add_argument(
        "-ta",
        "--throw_away_cache",
        default=False,
        action="store_true",
        help="Use a temporary folder which will be cleaned up after the program run.",
    )
    parser.add_argument(
        "--save_pred",
        default=False,
        action="store_true",
        help="To save the prediction in the output_dir/prediction if that is desired",
    )
    parser.add_argument(
        "-x",
        "--split",
        type=float,
        default=0.8,
        help="Split into training and validation samples, default is 80% training samples.",
    )
    parser.add_argument(
        "--val_fold",
        type=int,
        default=-1,
        help="Split into training and validation folds (5-fold cv).",
    )
    parser.add_argument(
        "--gpu_size",
        default="None",
        choices=["None", "small", "medium", "large"],
        help="Influcences some performance options of the code",
    )
    parser.add_argument(
        "--limit_gpu_memory_to",
        type=float,
        default=-1,
        help="Set it to the fraction of the GPU memory that shall be used, e.g. 0.5",
    )
    parser.add_argument(
        "-t",
        "--limit",
        type=int,
        default=0,
        help="Limit the amount of training/validation samples to a fixed number",
    )
    parser.add_argument(
        "--fixed_value",
        type=int,
        default=None,
        help="Fixed fragment index to use for inference (only for testing)",
    )
    parser.add_argument(
        "--dataset", default="TriALS", choices=["TriALS"]
    )
    parser.add_argument(
        "--use_test_data_for_validation", default=False, action="store_true", help="Use the test data instead of the split of the training data for validation. "
    )
    parser.add_argument("--train_on_all_samples", action="store_true")
    parser.add_argument(
        "--positive_crop_rate", type=float, default=0.6, help="The rate of positive samples for RandCropByPosNegLabeld"
    )

    # Configuration
    parser.add_argument("-s", "--seed", type=int, default=36)
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use.")
    parser.add_argument("--no_log", default=False, action="store_true")
    parser.add_argument("--no_data", default=False, action="store_true")
    parser.add_argument("--dont_check_output_dir", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")

    # Model
    parser.add_argument(
        "-n",
        "--network",
        default="dynunet",
        choices=["dynunet", "smalldynunet", "bigdynunet", "bigdynunet2", "matteodynunet"],
    )
    parser.add_argument(
        "-in",
        "--inferer",
        default="SlidingWindowInferer",
        choices=["SimpleInferer", "SlidingWindowInferer"],
    )
    parser.add_argument("--sw_roi_size", default="(128,128,128)", action="store")
    # crop_size multiples of sliding window size (128,128,128) with overlap 0.25 (default): 128, 224, 320, 416, 512
    parser.add_argument("--train_crop_size", default="(224,224,224)", action="store")
    parser.add_argument("--val_crop_size", default="None", action="store")
    # 1 on 24 Gb, 8 on 50 Gb,
    parser.add_argument("--train_sw_batch_size", type=int, default=8)
    parser.add_argument("--val_sw_batch_size", type=int, default=1)
    parser.add_argument("--train_sw_overlap", type=float, default=0.25)
    # Reduce this if you run into OOMs
    parser.add_argument("--val_sw_overlap", type=float, default=0.25)
    parser.add_argument("--sw_cpu_output", default=False, action="store_true")

    # Training
    parser.add_argument("-a", "--amp", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    # LOSS
    # If learning rate is set to 0.001, the DiceCELoss will produce Nans very quickly
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "Novograd"])
    parser.add_argument("--loss", default="DiceCELoss", choices=["DiceCELoss", "DiceLoss"])
    parser.add_argument(
        "--scheduler",
        default="CosineAnnealingLR",
        choices=["MultiStepLR", "PolynomialLR", "CosineAnnealingLR"],
    )
    parser.add_argument("--loss_dont_include_background", default=False, action="store_true")
    parser.add_argument("--loss_no_squared_pred", default=False, action="store_true")

    parser.add_argument("--resume_from", type=str, default="None")
    # Use this parameter to change the scheduler..
    parser.add_argument("--resume_override_scheduler", default=False, action="store_true")
    parser.add_argument("--use_scale_intensity_ranged", default=False, action="store_true")
    parser.add_argument("--additional_metrics", default=False, action="store_true")
    # Can speed up the training by cropping away some percentiles of the data
    parser.add_argument("--crop_foreground", default=False, action="store_true")

    # Logging
    parser.add_argument("-f", "--val_freq", type=int, default=1)  # Epoch Level
    parser.add_argument("--save_interval", type=int, default=50)  # Save checkpoints every x epochs

    parser.add_argument("--eval_only", default=False, action="store_true")
    parser.add_argument("--save_nifti", default=False, action="store_true")


    # Interactions
    parser.add_argument(
        "--non_interactive",
        action="store_true",
        help="Default training of neural network. Don't add any guidance channels, do normal backprop only.",
    )
    parser.add_argument("-it", "--max_train_interactions", type=int, default=10)
    parser.add_argument("-iv", "--max_val_interactions", type=int, default=10)
    parser.add_argument("-dpt", "--deepgrow_probability_train", type=float, default=1.0)
    parser.add_argument("-dpv", "--deepgrow_probability_val", type=float, default=1.0)

    # Guidance Signal Hyperparameters
    parser.add_argument("--sigma", type=int, default=1)
    parser.add_argument("--no_disks", default=False, action="store_true")

    # Guidance Signal Click Generation - for details see the mappings below
    parser.add_argument("-n_clicks", "--n_clicks", type=int, default=10)

    parser.add_argument("-tcg", "--train_click_generation", type=int, default=2, choices=[1, 2])
    parser.add_argument("-vcg", "--val_click_generation", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "-tcgsc",
        "--train_click_generation_stopping_criterion",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
    )
    # Usually this setting should be at 1, so max_iter
    parser.add_argument(
        "-vcgsc",
        "--val_click_generation_stopping_criterion",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
    )
    # only needed for training
    parser.add_argument("--train_loss_stopping_threshold", type=float, default=0.1)
    parser.add_argument("--train_iteration_probability", type=float, default=0.5)

    parser.add_argument("--loop", default=False, action="store_true")


    # Set up additional information concerning the environment and the way the script was called
    args, unknown = parser.parse_known_args()
    return args


def setup_environment_and_adapt_args(args, docker=False):
    args.caller_args = sys.argv
    args.env = os.environ
    args.git = get_git_information()

    device = torch.device(f"cuda:{args.gpu}")

    # TriALS specific
    args.throw_away_cache = True
    args.dont_check_output_dir = True
    args.resume_from = 'sw_fastedit_src/model.pt'
    args.dataset = 'TriALS'
    args.no_log = True
    args.no_data = True

    args.labels = {"lesion": 1, "background": 0} # TriALS


    if not args.dont_check_output_dir and os.path.isdir(args.output_dir):
        raise UserWarning(
            f"output path {args.output_dir} already exists. Please choose another path or set --dont_check_output_dir"
        )
    if not os.path.exists(args.output_dir):
        pathlib.Path(args.output_dir).mkdir(parents=True)

    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    if args.no_log:
        log_folder_path = None
    else:
        log_folder_path = args.output_dir
    setup_loggers(loglevel, log_folder_path)
    logger = get_logger()

    if args.eval_only:
        # Avoid a loading error from the training where it complains the number of epochs is too low
        args.epochs = 100000

    if args.cache_dir == "None":
        if not args.throw_away_cache:
            raise UserWarning("Cache directory (-c) has to be set if args.throw_away_cache is not True")
        else:
            args.cache_dir = tempfile.TemporaryDirectory().name
    else:
        if args.throw_away_cache:
            args.cache_dir = f"{args.cache_dir}/{uuid.uuid4()}"
        else:
            logger.warning("Reusing the cache_dir between different network runs may lead to cache inconsistencies.")
            logger.warning("Most importantly the crops may not be updated if you set them differently")
            logger.warning("PersistentDataset does not detect this automatically but only checks if the hash matches")
            logger.warning("Waiting shortly...")
            time.sleep(10)
            args.cache_dir = f"{args.cache_dir}"

    if not os.path.exists(args.cache_dir):
        pathlib.Path(args.cache_dir).mkdir(parents=True)

    if args.data_dir == "None":
        args.data_dir = f"{args.output_dir}/data"
        logger.info(f"--data was None, so that {args.data_dir}/data was selected instead")

    if not args.no_data:
        if not os.path.exists(args.data_dir):
            pathlib.Path(args.data_dir).mkdir(parents=True)

    # Training only, so done on the patch of size train_crop_size
    train_click_generation_mapping = {
        1: ClickGenerationStrategy.GLOBAL_NON_CORRECTIVE,  # "non-corrective",
        2: ClickGenerationStrategy.GLOBAL_CORRECTIVE,  # "corrective",
    }
    args.train_click_generation = train_click_generation_mapping[args.train_click_generation]
    # Validation, so everything is done on the full volume
    val_click_generation_mapping = {
        1: ClickGenerationStrategy.GLOBAL_CORRECTIVE,  # "patch-based corrective",
        # Sample directly from the global error
        2: ClickGenerationStrategy.PATCH_BASED_CORRECTIVE,  # "global corrective",
    }
    args.val_click_generation = val_click_generation_mapping[args.val_click_generation]

    args.train_click_generation_stopping_criterion = StoppingCriterion(args.train_click_generation_stopping_criterion)
    args.val_click_generation_stopping_criterion = StoppingCriterion(args.val_click_generation_stopping_criterion)

    # NOTE Added for backwards compatibility with DeepGrow. Manual override of some settings, thus need to accept it
    if args.deepgrow_probability_val != 1 or args.deepgrow_probability_val != 1:
        logger.warning("############## DeepGrow mode activated ###################")
        logger.warning(
            """args.train_click_generation, args.val_click_generation, args.train_click_generation_stopping_criterion
             and args.val_click_generation_stopping_criterion will be overwritten by this option"""
        )
        logger.warning("##########################################################")
        args.train_click_generation_stopping_criterion = StoppingCriterion.DEEPGROW_PROBABILITY
        args.val_click_generation_stopping_criterion = StoppingCriterion.DEEPGROW_PROBABILITY
        args.train_click_generation = ClickGenerationStrategy.DEEPGROW_GLOBAL_CORRECTIVE
        args.val_click_generation = ClickGenerationStrategy.DEEPGROW_GLOBAL_CORRECTIVE

    args.real_cuda_device = get_actual_cuda_index_of_device(torch.device(f"cuda:{args.gpu}"))

    logger.info(f"CPU Count: {os.cpu_count()}")
    logger.info(f"Num threads: {torch.get_num_threads()}")

    args.cwd = os.getcwd()

    if args.gpu_size == "None":
        nv_total = gpu_usage(device, used_memory_only=False)[3]
        if nv_total < 25:
            args.gpu_size = "small"
        elif nv_total < 55:
            args.gpu_size = "medium"
        else:
            args.gpu_size = "large"
        logger.info(f"Selected GPU size: {args.gpu_size}, since GPU Memory: {nv_total} GB")

    # Init the Inferer
    args.sw_roi_size = eval(args.sw_roi_size)
    assert len(args.sw_roi_size) == 3

    if args.val_crop_size == "None":
        args.val_crop_size = None
    else:
        args.val_crop_size = eval(args.val_crop_size)
        assert len(args.val_crop_size) == 3

    if args.train_crop_size == "None":
        args.train_crop_size = None
    else:
        args.train_crop_size = eval(args.train_crop_size)
        assert len(args.train_crop_size) == 3

    # verify both have a valid size (for Unet with seven layers)
    if args.inferer == "SimpleInferer":
        for size in args.train_crop_size:
            assert (size % 32) == 0
        for size in args.val_crop_size:
            assert (size % 32) == 0

    return args, logger
