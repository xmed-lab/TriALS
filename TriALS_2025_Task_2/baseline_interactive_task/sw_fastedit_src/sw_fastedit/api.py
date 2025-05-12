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

import logging
import os
import random
from collections import OrderedDict
from functools import reduce
from pickle import dump
from typing import Iterable, List
import sys

import cupy as cp
import numpy as np
import torch
from ignite.engine import Events
from ignite.handlers import TerminateOnNan
from monai.data import set_track_meta
from monai.engines import EnsembleEvaluator, SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointLoader,
    CheckpointSaver,
    GarbageCollector,
    IgniteMetricHandler,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import SurfaceDiceMetric
from monai.networks.nets.dynunet import DynUNet
from monai.optimizers.novograd import Novograd
from monai.transforms import Compose
from monai.utils import set_determinism

from sw_fastedit.data import (
    get_click_transforms,
    get_click_transforms_json,
    get_post_transforms,
    get_pre_transforms_train_as_list,
    get_pre_transforms_val_as_list,
    get_train_loader,
    get_val_loader,
    get_test_loader,
)
from sw_fastedit.interaction import Interaction
from sw_fastedit.utils.helper import count_parameters, is_docker, run_once, handle_exception

logger = logging.getLogger("sw_fastedit")
output_dir = None


def get_optimizer(optimizer: str, lr: float, network):
    # OPTIMIZER
    if optimizer == "Novograd":
        optimizer = Novograd(network.parameters(), lr)
    elif optimizer == "Adam":  # default
        optimizer = torch.optim.Adam(network.parameters(), lr)
    return optimizer


def get_loss_function(loss_args, loss_kwargs=None):  # squared_pred=True, include_background=True):
    if loss_kwargs is None:
        loss_kwargs = {}
    if loss_args == "DiceCELoss":
        # squared_pred enables faster convergence, possibly even better results in the long run
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, **loss_kwargs)
    elif loss_args == "DiceLoss":
        loss_function = DiceLoss(to_onehot_y=True, softmax=True, **loss_kwargs)
    return loss_function


def get_network(network_str: str, labels: Iterable, non_interactive: bool = False):
    """
    in_channels: 1 slice for the image, the other ones for the signal per label whereas each signal is the size of image.
        The signal is only added for interactive runs of this code.
    out_channels: amount of labels
    """
    in_channels = 1 if non_interactive else 1 + len(labels)
    out_channels = len(labels)

    if network_str == "dynunet":
        network = DynUNet(
            spatial_dims=3,
            #
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )
    elif network_str == "smalldynunet":
        network = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3],
            strides=[1, 2, [2, 2, 1]],
            upsample_kernel_size=[2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )
    elif network_str == "bigdynunet":
        network = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, 2, [2, 2, 1]],
            # filters=[64, 96, 128, 192, 256, 384, 512],#, 768, 1024, 2048],
            # dropout=0.1,
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )
    # No good results with this dynunet so far
    elif network_str == "hugedynunet":
        network = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2, 2, 2],
            filters=[32, 64, 128, 192, 256, 384, 512],  # , 768, 1024, 2048],
            dropout=0.1,
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )

    logger.info(f"Selected network {network.__class__.__qualname__}")
    logger.info(f"Number of parameters: {count_parameters(network):,}")

    return network


def get_inferers(
    inferer: str,
    *,
    sw_roi_size,
    train_crop_size,
    val_crop_size,
    train_sw_batch_size,
    val_sw_batch_size,
    train_sw_overlap=0.25,
    val_sw_overlap=0.25,
    cache_roi_weight_map: bool = True,
    device="cpu",
    sw_cpu_output=False,
):
    if inferer == "SimpleInferer":
        train_inferer = SimpleInferer()
        eval_inferer = SimpleInferer()
    elif inferer == "SlidingWindowInferer":
        # train_batch_size is limited due to this bug: https://github.com/Project-MONAI/MONAI/issues/6628
        assert train_crop_size is not None
        train_batch_size = max(
            1,
            min(
                reduce(
                    lambda x, y: x * y,
                    [round(train_crop_size[i] / sw_roi_size[i]) for i in range(len(sw_roi_size))],
                ),
                train_sw_batch_size,
            ),
        )
        logger.info(f"{train_batch_size=}")
        average_sample_shape = (300, 300, 400)
        if val_crop_size is not None:
            average_sample_shape = val_crop_size

        val_batch_size = max(
            1,
            min(
                reduce(
                    lambda x, y: x * y,
                    [round(average_sample_shape[i] / sw_roi_size[i]) for i in range(len(sw_roi_size))],
                ),
                val_sw_batch_size,
            ),
        )
        logger.info(f"{val_batch_size=}")

        sw_params = {
            "roi_size": sw_roi_size,
            "mode": "gaussian",
            "cache_roi_weight_map": cache_roi_weight_map,
        }

        if sw_cpu_output:
            logger.warning("Enabling Sliding Window output on the CPU")
            logger.warning(
                "Note that this only works well for validation! For training AMP has to be turned off and it has no real effect"
            )
            sw_params.update({"sw_device": device, "device": "cpu"})
        train_inferer = SlidingWindowInferer(sw_batch_size=train_batch_size, overlap=train_sw_overlap, **sw_params)
        eval_inferer = SlidingWindowInferer(sw_batch_size=val_batch_size, overlap=val_sw_overlap, **sw_params)
    return train_inferer, eval_inferer


def get_scheduler(optimizer, scheduler_str: str, epochs_to_run: int):
    if scheduler_str == "MultiStepLR":
        steps = 4
        steps_per_epoch = round(epochs_to_run / steps)
        if steps_per_epoch < 1:
            logger.error(f"Chosen number of epochs {epochs_to_run}/{steps} < 0")
            milestones = range(0, epochs_to_run)
        else:
            milestones = [num for num in range(0, epochs_to_run) if num % round(steps_per_epoch) == 0][1:]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.333)
    elif scheduler_str == "PolynomialLR":
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs_to_run, power=2)
    elif scheduler_str == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_run, eta_min=1e-8)
    return lr_scheduler


def get_val_handlers(sw_roi_size: List, inferer: str, gpu_size: str, *,garbage_collector=True, non_interactive=False):
    every_x_iterations = 2
    if non_interactive:
        every_x_iterations *= 10
    
    if sw_roi_size[0] < 128:
        val_trigger_event = Events.ITERATION_COMPLETED(every=every_x_iterations) if gpu_size == "large" else Events.ITERATION_COMPLETED
    else:
        val_trigger_event = Events.ITERATION_COMPLETED(every=every_x_iterations) if gpu_size == "large" else Events.ITERATION_COMPLETED

    # define event-handlers for engine
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        # End of epoch GarbageCollection
        GarbageCollector(log_level=10),
    ]
    if garbage_collector:
        # https://github.com/Project-MONAI/MONAI/issues/3423
        iteration_gc = GarbageCollector(log_level=10, trigger_event=val_trigger_event)
        val_handlers.append(iteration_gc)

    return val_handlers


def get_train_handlers(
    lr_scheduler,
    evaluator,
    val_freq,
    eval_only: bool,
    sw_roi_size: List,
    inferer: str,
    gpu_size: str,
    garbage_collector=True,
    non_interactive=False,
):
    every_x_iterations = 4
    if non_interactive:
        every_x_iterations *= 10
    
    if sw_roi_size[0] <= 128:
        train_trigger_event = Events.ITERATION_COMPLETED(every=every_x_iterations) if gpu_size == "large" else Events.ITERATION_COMPLETED
    else:        
        train_trigger_event = (
            Events.ITERATION_COMPLETED(every=every_x_iterations*2) if gpu_size == "large" else Events.ITERATION_COMPLETED(every=every_x_iterations)
        )

    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(
            validator=evaluator,
            interval=val_freq,
            epoch_level=(not eval_only),
        ),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        # End of epoch GarbageCollection
        GarbageCollector(log_level=10),
    ]
    if garbage_collector:
        # https://github.com/Project-MONAI/MONAI/issues/3423
        iteration_gc = GarbageCollector(log_level=10, trigger_event=train_trigger_event)
        train_handlers.append(iteration_gc)

    return train_handlers


def get_key_metric(str_to_prepend="") -> OrderedDict:
    key_metrics = OrderedDict()
    key_metrics[f"{str_to_prepend}dice"] = MeanDice(
        output_transform=from_engine(["pred", "label"]), include_background=False, save_details=False
    )
    return key_metrics


def get_additional_metrics(labels, include_background=False, loss_kwargs=None, str_to_prepend=""):
    # loss_function_metric = loss_function
    if loss_kwargs is None:
        loss_kwargs = {}
    mid = "with_bg_" if include_background else "without_bg_"
    loss_function = DiceCELoss(softmax=True, **loss_kwargs)
    # loss_function_metric = LossMetric(loss_fn=loss_function, reduction="mean", get_not_nans=False)
    loss_function_metric_ignite = IgniteMetricHandler(
        loss_fn=loss_function,
        output_transform=from_engine(["pred", "label"]),
        save_details=False,
    )
    amount_of_classes = len(labels) if include_background else (len(labels) - 1)
    class_thresholds = (0.5,) * amount_of_classes
    surface_dice_metric = SurfaceDiceMetric(
        include_background=include_background,
        class_thresholds=class_thresholds,
        reduction="mean",
        get_not_nans=False,
        use_subvoxels=True,
    )
    surface_dice_metric_ignite = IgniteMetricHandler(
        metric_fn=surface_dice_metric,
        output_transform=from_engine(["pred", "label"]),
        save_details=False,
    )

    additional_metrics = OrderedDict()
    additional_metrics[f"{str_to_prepend}{loss_function.__class__.__name__.lower()}"] = loss_function_metric_ignite
    additional_metrics[f"{str_to_prepend}{mid}surface_dice"] = surface_dice_metric_ignite

    # Disabled since it led to weird artefacts in the Tensorboard diagram
    # for key_label in args.labels:
    #     if key_label != "background":
    #         all_val_metrics[key_label + "_dice"] = MeanDice(
    #             output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
    #         )

    return additional_metrics


def get_test_evaluator(
    args,
    network,
    inferer,
    device,
    val_loader,
    post_transform,
    resume_from="None",
) -> SupervisedEvaluator:
    init(args)

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        inferer=inferer,
        postprocessing=post_transform,
        amp=args.amp,
        val_handlers=get_val_handlers(
            sw_roi_size=args.sw_roi_size,
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=(not args.non_interactive),
        ),
    )

    save_dict = {
        "net": network,
    }

    if resume_from != "None":
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        logger.info(f"CWD: {os.getcwd()}")
        map_location = device
        checkpoint = torch.load(resume_from)
        logger.info(f"{checkpoint.keys()=}")
        network.load_state_dict(checkpoint["net"])
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        handler(evaluator)

    return evaluator



def create_supervised_evaluator(args, resume_from="None") -> SupervisedEvaluator:
    init(args)

    device = torch.device(f"cuda:{args.gpu}")

    pre_transforms_val = Compose(get_pre_transforms_val_as_list(args.labels, device, args))
    if args.use_test_data_for_validation:
        val_loader = get_test_loader(args, pre_transforms_val)
    else:
        val_loader = get_val_loader(args, pre_transforms_val)


    click_transforms = get_click_transforms(device, args)
    post_transform = get_post_transforms(args.labels, save_pred=args.save_pred, output_dir=args.output_dir, pretransform=pre_transforms_val)

    network = get_network(args.network, args.labels, args.non_interactive).to(device)
    _, eval_inferer = get_inferers(
        args.inferer,
        sw_roi_size=args.sw_roi_size,
        train_crop_size=args.train_crop_size,
        val_crop_size=args.val_crop_size,
        train_sw_batch_size=args.train_sw_batch_size,
        val_sw_batch_size=args.val_sw_batch_size,
        val_sw_overlap=args.val_sw_overlap,
        cache_roi_weight_map=True,
    )

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_function = get_loss_function(loss_args=args.loss, loss_kwargs=loss_kwargs)
    val_key_metric = get_key_metric(str_to_prepend="val_")
    val_additional_metrics = {}

    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )  # (not args.loss_dont_include_background)

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_val,
            transforms=click_transforms,
            train=False,
            label_names=args.labels,
            max_interactions=args.max_val_interactions,
            save_nifti=args.save_nifti,
            nifti_dir=args.data_dir,
            loss_function=loss_function,
            nifti_post_transform=post_transform,
            click_generation_strategy=args.val_click_generation,
            stopping_criterion=args.val_click_generation_stopping_criterion,
            non_interactive=args.non_interactive,
        ),
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
        val_handlers=get_val_handlers(sw_roi_size=args.sw_roi_size, inferer=args.inferer, gpu_size=args.gpu_size),
    )

    save_dict = {
            "net": network,
    }

    if resume_from != "None":
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        map_location = device  # {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        checkpoint = torch.load(resume_from)

        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys()}"

        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)
        handler(evaluator)

    return evaluator, val_key_metric, val_additional_metrics

def get_supervised_evaluator(
    args,
    network,
    inferer,
    device,
    val_loader,
    loss_function,
    click_transforms,
    post_transform,
    key_val_metric,
    additional_metrics,
) -> SupervisedEvaluator:
    init(args)

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_val,
            transforms=click_transforms,
            train=False,
            label_names=args.labels,
            max_interactions=args.max_val_interactions,
            save_nifti=args.save_nifti,
            nifti_dir=args.data_dir,
            loss_function=loss_function,
            nifti_post_transform=post_transform,
            click_generation_strategy=args.val_click_generation,
            stopping_criterion=args.val_click_generation_stopping_criterion,
            non_interactive=args.non_interactive,
        )
        if not args.non_interactive
        else None,
        inferer=inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=key_val_metric,
        additional_metrics=additional_metrics,
        val_handlers=get_val_handlers(
            sw_roi_size=args.sw_roi_size,
            inferer=args.inferer,
            gpu_size=args.gpu_size,
            garbage_collector=True,
        ),
    )
    return evaluator




def get_ensemble_evaluator(
    args, networks, inferer, device, val_loader, post_transform, resume_from="None", nfolds=5
) -> EnsembleEvaluator:
    init(args)

    device = torch.device(f"cuda:{args.gpu}")
    prediction_keys = [f"pred_{i}" for i in range(nfolds)]

    evaluator = EnsembleEvaluator(
        device=device,
        val_data_loader=val_loader,
        networks=networks,
        inferer=inferer,
        postprocessing=post_transform,
        pred_keys=prediction_keys,
        amp=args.amp,
    )

    if resume_from != "None":
        logger.info(f"{args.gpu}:: Loading Networks...")
        logger.info(f"CWD: {os.getcwd()}")
        resume_path = os.path.abspath(resume_from)
        logger.info(f"{resume_path=}")

        for i in range(nfolds):
            file_path = os.path.join(resume_path, f"{i}.pt")
            logger.info(f"{file_path=}")
            checkpoint = torch.load(file_path)
            networks[i].load_state_dict(checkpoint["net"])
    return evaluator


def get_trainer(
    args, file_prefix="", ensemble_mode: bool = False, resume_from="None"
) -> List[SupervisedTrainer | None, SupervisedEvaluator | None, List]:
    init(args)
    device = torch.device(f"cuda:{args.gpu}") if not args.sw_cpu_output else "cpu"
    sw_device = torch.device(f"cuda:{args.gpu}")

    pre_transforms_val = Compose(get_pre_transforms_val_as_list(args.labels, device, args))
    if args.use_test_data_for_validation:
        val_loader = get_test_loader(args, pre_transforms_val)
    else:
        val_loader = get_val_loader(args, pre_transforms_val)


    if args.dataset == "PENGWIN" or args.dataset == "PENGWIN_Frag" or args.dataset == "ODIN" or args.dataset == "TriALS":
        n_clicks = 5 if args.dataset != 'TriALS' else 10
        click_transforms = get_click_transforms_json(device, args, n_clicks=n_clicks)
    else:
        click_transforms = get_click_transforms(sw_device, args)

    if args.eval_only:
        post_transform = get_post_transforms(args.labels, save_pred=args.save_pred, output_dir=args.output_dir, pretransform=pre_transforms_val)
    else:
        post_transform = get_post_transforms(args.labels, save_pred=args.save_pred, output_dir=args.output_dir)
    
    network = get_network(args.network, args.labels, args.non_interactive).to(sw_device)
    train_inferer, eval_inferer = get_inferers(
        args.inferer,
        sw_roi_size=args.sw_roi_size,
        train_crop_size=args.train_crop_size,
        val_crop_size=args.val_crop_size,
        train_sw_batch_size=args.train_sw_batch_size,
        val_sw_batch_size=args.val_sw_batch_size,
        train_sw_overlap=args.train_sw_overlap,
        val_sw_overlap=args.val_sw_overlap,
    )

    loss_kwargs = {
        "squared_pred": (not args.loss_no_squared_pred),
        "include_background": (not args.loss_dont_include_background),
    }
    loss_function = get_loss_function(loss_args=args.loss, loss_kwargs=loss_kwargs)
    optimizer = get_optimizer(args.optimizer, args.learning_rate, network)
    lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)
    
    val_key_metric = get_key_metric(str_to_prepend="val_")
    val_additional_metrics = {}
    if args.additional_metrics:
        val_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="val_"
        )

    evaluator = get_supervised_evaluator(
        args,
        network=network,
        inferer=eval_inferer,
        device=device,
        val_loader=val_loader,
        loss_function=loss_function,
        click_transforms=click_transforms,
        post_transform=post_transform,
        key_val_metric=val_key_metric,
        additional_metrics=val_additional_metrics,
    )

    pre_transforms_train = Compose(get_pre_transforms_train_as_list(args.labels, device, args))
    train_loader = get_train_loader(args, pre_transforms_train)
    train_key_metric = get_key_metric(str_to_prepend="train_")
    train_additional_metrics = {}
    if args.additional_metrics:
        train_additional_metrics = get_additional_metrics(
            args.labels, include_background=False, loss_kwargs=loss_kwargs, str_to_prepend="train_"
        )


    train_handlers = get_train_handlers(
        lr_scheduler,
        evaluator,
        args.val_freq,
        args.eval_only,
        args.sw_roi_size,
        args.inferer,
        args.gpu_size,
        garbage_collector=True,
    )
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_train,
            transforms=click_transforms,
            train=True,
            label_names=args.labels,
            max_interactions=args.max_train_interactions,
            save_nifti=args.save_nifti,
            nifti_dir=args.data_dir,
            loss_function=loss_function,
            nifti_post_transform=post_transform,
            click_generation_strategy=args.train_click_generation,
            stopping_criterion=args.train_click_generation_stopping_criterion,
            iteration_probability=args.train_iteration_probability,
            loss_stopping_threshold=args.train_loss_stopping_threshold,
            non_interactive=args.non_interactive,
        )
        if not args.non_interactive
        else None,
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=train_key_metric,
        additional_metrics=train_additional_metrics,
        train_handlers=train_handlers,
    )

    if not args.eval_only:
            save_dict = {
                "trainer": trainer,
                "net": network,
                #"opt": optimizer,
                #"lr": lr_scheduler,
            }
    else:
        save_dict = {
            "net": network,
        }

    if ensemble_mode:
        save_dict = {
            "net": network,
        }


    if not ensemble_mode:
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_interval=args.save_interval,
            save_final=True,
            final_filename="checkpoint.pt",
            save_key_metric=True,
            n_saved=2,
            file_prefix="train",
        ).attach(trainer)
        CheckpointSaver(
            save_dir=args.output_dir,
        save_dict=save_dict,
            save_key_metric=True,
            save_final=True,
            save_interval=args.save_interval,
            final_filename="pretrained_deepedit_" + args.network + "-final.pt",
        ).attach(evaluator)
    else:
        CheckpointSaver(
            save_dir=args.output_dir,
            save_dict=save_dict,
            save_key_metric=True,
            file_prefix=file_prefix,
        ).attach(evaluator)

    if trainer is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if resume_from != "None":
        if args.resume_override_scheduler:
            # Remove those parts
            saved_opt = save_dict["opt"]
            saved_lr = save_dict["lr"]
            del save_dict["opt"]
            del save_dict["lr"]
        logger.info(f"{args.gpu}:: Loading Network...")
        logger.info(f"{save_dict.keys()=}")
        map_location = device  # {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        checkpoint = torch.load(resume_from)
        

        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys()}"

        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(load_path=resume_from, load_dict=save_dict, map_location=map_location)

        if trainer is not None:
            handler(trainer)
        else:
            handler(evaluator)

        if args.resume_override_scheduler:
            # Restore params
            save_dict["opt"] = saved_opt
            save_dict["lr"] = saved_lr

    return trainer, evaluator, train_key_metric, train_additional_metrics, val_key_metric, val_additional_metrics



@run_once
def init(args):
    global output_dir
    # for OOM debugging
    output_dir = args.output_dir
    sys.excepthook = handle_exception

    if not is_docker():
        torch.set_num_threads(int(os.cpu_count()))  # Limit number of threads to 1/3 of resources

        # # needed for the interaction training for some reason
        # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        # resource.setrlimit(resource.RLIMIT_NOFILE, (8 * 8192, rlimit[1]))

    if args.limit_gpu_memory_to != -1:
        limit = args.limit_gpu_memory_to
        assert limit > 0 and limit < 1, f"Percentage GPU memory limit is invalid! {limit} > 0 or < 1"
        torch.cuda.set_per_process_memory_fraction(limit, args.gpu)

        #    # Slurm only: Speed up the creation of temporary files
        #    if os.environ.get("SLURM_JOB_ID") is not None:
        #        tmpdir = "/local/work/mhadlich/tmp"
        #        os.environ["TMPDIR"] = tmpdir
        #        if not os.path.exists(tmpdir):
        #            pathlib.Path(tmpdir).mkdir(parents=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True

    set_track_meta(True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    set_determinism(seed=args.seed)

    if not is_docker():
        with cp.cuda.Device(args.gpu):
            cp.random.seed(seed=args.seed)

    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        np.seterr(all="raise")


def oom_observer(device, alloc, device_alloc, device_free):
    if device is not None and logger is not None:
        logger.critical(torch.cuda.memory_summary(device))
    # snapshot right after an OOM happened
    print("saving allocated state during OOM")
    print("Tips: \nReduce sw_batch_size if there is an OOM (maybe even roi_size)")
    snapshot = torch.cuda.memory._snapshot()
    dump(snapshot, open(f"{output_dir}/oom_snapshot.pickle", "wb"))
    # logger.critical(snapshot)
    torch.cuda.memory._save_memory_usage(filename=f"{output_dir}/memory.svg", snapshot=snapshot)
    torch.cuda.memory._save_segment_usage(filename=f"{output_dir}/segments.svg", snapshot=snapshot)
