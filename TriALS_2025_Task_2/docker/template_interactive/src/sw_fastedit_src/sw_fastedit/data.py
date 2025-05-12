from __future__ import annotations

import glob
import logging
import os
import shutil

# from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from monai.apps import CrossValidation
from monai.data import DataLoader, Dataset, ThreadDataLoader, partition_dataset
from monai.data.utils import select_cross_validation_folds

from monai.data.dataset import PersistentDataset
from monai.data.folder_layout import FolderLayout
from monai.transforms import (  
    Activationsd,
    AsDiscreted,
    Compose,
    CopyItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Identityd,
    Invertd,
    LoadImaged,
    MeanEnsembled,
    SaveImaged,
    ScaleIntensityRanged,
    SignalFillEmptyd,
    ToDeviced,
    ToTensord,
    VoteEnsembled,
    Zoomd,
)
from monai.utils.enums import CommonKeys

from sw_fastedit.helper_transforms import (  
    InitLoggerd,
    TrackTimed,
)
from sw_fastedit.transforms import (
    AddEmptySignalChannels,
    AddGuidance,
    AddGuidanceJSON,
    AddGuidanceSignal,
    FindDiscrepancyRegions,
    NormalizeLabelsInDatasetd,
    SplitPredsLabeld,

)

logger = logging.getLogger("sw_fastedit")




def get_pre_transforms(labels: Dict, device, args, input_keys=("image", "label")):
    return Compose(get_pre_transforms_train_as_list(labels, device, args, input_keys)), Compose(
        get_pre_transforms_val_as_list(labels, device, args, input_keys)
    )


def get_spacing(args):
    AUTOPET_SPACING = (2.03642011, 2.03642011, 3.0)
    MSD_SPLEEN_SPACING = (2 * 0.79296899, 2 * 0.79296899, 5.0)
    # Apply this only to the label!
    #HECKTOR_SPACING = (4, 4, 4)
    HECKTOR_SPACING = (2.03642011, 2.03642011, 3.0)
    #HECKTOR_SPACING = (2,2,2)
    #HECKTOR_SPACING = (4 * 0.98, 4 * 0.98, 1 * 3.27)

    if args.dataset == "AutoPET" or args.dataset == "AutoPET2" or args.dataset == "AutoPET2_Challenge":
        return AUTOPET_SPACING
    elif args.dataset == "HECKTOR":
        return HECKTOR_SPACING
    elif args.dataset == "MSD_Spleen":
        return MSD_SPLEEN_SPACING
    # return spacing


def get_pre_transforms_train_as_list(labels: Dict, device, args, input_keys=("image", "label", "liver")):
    cpu_device = torch.device("cpu")
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO


    if args.dataset == 'TriALS':
        input_keys=("image", "label", "liver")
        t = [
            InitLoggerd(
                loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(
                keys=input_keys,
                reader="ITKReader",
                image_only=False,
                simple_keys=True,
            ),
            ToTensord(keys=input_keys, device=cpu_device, track_meta=True),
            EnsureChannelFirstd(keys=input_keys),
            SignalFillEmptyd(input_keys),
            ScaleIntensityRanged(keys="image", a_min=-75, a_max=177, b_min=0.0, b_max=1.0, clip=True), # 0.05 and 99.95 percentiles of the liver HUs on the training data
            NormalizeLabelsInDatasetd(keys="label", labels=labels, device=cpu_device, allow_missing_keys=True),
            AddEmptySignalChannels(keys="image", device=cpu_device),
            CropForegroundd(
                keys=("image", "label", "pred", "liver"),
                source_key="liver",
                allow_missing_keys=True
            ),
            Zoomd(keys=("image", "label", "pred", "liver"), zoom=0.8, mode=["area", "nearest", "area", "nearest"], keep_size=False, allow_missing_keys=True),
        ]
    return t


def get_pre_transforms_val_as_list(labels: Dict, device, args, input_keys=("image", "label")):
    cpu_device = torch.device("cpu")
    spacing = get_spacing(args)

    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    # data Input keys have to be at least ["image"] for val
    if args.dataset == 'TriALS':
        input_keys=("image", "label", "liver") # assume liver is pre-computed with TotalSegmentator

        t = [
            InitLoggerd(
                loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(
                keys=input_keys,
                reader="ITKReader",
                image_only=False,
                simple_keys=True,
                allow_missing_keys=True
            ),
            ToTensord(keys=("image", "label", "liver"), device=cpu_device, track_meta=True, allow_missing_keys=True),
            EnsureChannelFirstd(keys=input_keys, allow_missing_keys=True),
            SignalFillEmptyd(input_keys, allow_missing_keys=True),
            ScaleIntensityRanged(keys="image", a_min=-75, a_max=177, b_min=0.0, b_max=1.0, clip=True), # 0.05 and 99.95 percentiles of the liver HUs on the training data
            NormalizeLabelsInDatasetd(keys="label", labels=labels, device=cpu_device, allow_missing_keys=True),
            AddEmptySignalChannels(keys='image', device=cpu_device),
            CropForegroundd(
                keys=("image", "label", "pred", "liver"),
                source_key="liver",
                allow_missing_keys=True
            ),
            Zoomd(keys=("image", "label", "pred", "liver"), zoom=0.8, mode=["area", "nearest", "area", "nearest"], keep_size=False, allow_missing_keys=True),
        ]
    if args.debug:
        for i in range(len(t)):
            t[i] = TrackTimed(t[i])

    return t


def get_device(data):
    return f"device - {data.device}"


def get_click_transforms(device, args): # for training

    cpu_device = torch.device("cpu")

    logger.info(f"{device=}")

    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        FindDiscrepancyRegions(keys="label", pred_key="pred", discrepancy_key="discrepancy", device=device),
        AddGuidance(
            keys="NA",
            discrepancy_key="discrepancy",
            probability_key="probability",
            device=device,
        ),
        # Overwrites the image entry
        AddGuidanceSignal(
            keys="image",
            sigma=args.sigma,
            disks=(not args.no_disks),
            device=device,
        ),
        CropForegroundd(
            keys=("image", "label", "liver"),
            source_key="liver",
        ),

        ToTensord(keys=("image", "label", "pred"), device=cpu_device)
        if args.sw_cpu_output
        else Identityd(keys=("pred",), allow_missing_keys=True),
    ]

    return Compose(t)

def get_click_transforms_json(device, args, n_clicks=10):
    cpu_device = torch.device("cpu")

    logger.info(f"{device=}")

    number_intensity_ch = 1

    t = [
        AddGuidanceJSON(
            keys="NA",
            json_dir=args.json_dir,
            n_clicks=n_clicks,
        ),
        # Overwrites the image entry
        AddGuidanceSignal(
            keys="image",
            sigma=args.sigma,
            disks=(not args.no_disks),
            device=device,
            number_intensity_ch=number_intensity_ch,
        ),
        ToTensord(keys=("image", "label", "pred"), device=cpu_device)
        if args.sw_cpu_output
        else Identityd(keys=("pred",), allow_missing_keys=True),
    ]


    return Compose(t)


def get_post_transforms(labels, *, save_pred=False, output_dir=None, pretransform=None, output_postfix="", docker=False):
    cpu_device = torch.device("cpu")
    if save_pred:
        if output_dir is None:
            raise UserWarning("output_dir may not be empty when save_pred is enabled...")
        if pretransform is None:
            logger.warning("Make sure to add a pretransform here if you want the prediction to be inverted")

    input_keys = ("pred",)
    t = [
        CopyItemsd(keys=("pred",), times=1, names=("pred_for_save",))
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),
        Invertd(
            keys=("pred_for_save",),
            orig_keys="pred",
            nearest_interp=False,
            transform=pretransform,
        )
        if (save_pred and pretransform is not None)
        else Identityd(keys=input_keys, allow_missing_keys=True),
        Activationsd(keys=("pred",), softmax=True),
        AsDiscreted(
            keys="pred_for_save",
            argmax=True,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ) 
        if not docker
        else 
        AsDiscreted(
            keys=("pred"),
            argmax=(True),
            to_onehot=(len(labels)),
            ), 
        SaveImaged(
            keys=("pred_for_save",),
            writer="ITKWriter",
            output_dir=os.path.join(output_dir, "predictions"),
            output_postfix=output_postfix,
            output_dtype=np.uint8,
            separate_folder=False,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),
        ToTensord(keys=("image", "label", "pred"), device=cpu_device)
        if not docker
        else ToTensord(keys=("image", "pred"), device=cpu_device)

    ]
    return Compose(t)


def get_post_transforms_unsupervised(labels, device, pred_dir, pretransform):
    os.makedirs(pred_dir, exist_ok=True)
    nii_layout = FolderLayout(output_dir=pred_dir, postfix="", extension=".nii.gz", makedirs=False)

    t = [
        Invertd(
            keys="pred",
            orig_keys="image",
            nearest_interp=False,
            transform=pretransform,
        ),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys="pred",
            argmax=True,  # to_onehot=(len(labels),),
        ),
        # This transform is to check dice score per segment/label, disabled not needed right now
        # SplitPredsLabeld(keys="pred"),
        SaveImaged(
            keys="pred",
            writer="ITKWriter",
            output_postfix="",
            output_ext=".nii.gz",
            folder_layout=nii_layout,
            output_dtype=np.uint8,
            separate_folder=False,
            resample=False,
        ),
    ]
    return Compose(t)


def get_post_ensemble_transforms(labels, device, pred_dir, pretransform, nfolds=5, weights=None):
    prediction_keys = [f"pred_{i}" for i in range(nfolds)]

    os.makedirs(pred_dir, exist_ok=True)
    nii_layout = FolderLayout(output_dir=pred_dir, postfix="", extension=".nii.gz", makedirs=False)

    t = [
        Invertd(
            keys=prediction_keys,
            orig_keys="image",
            nearest_interp=False,
            transform=pretransform,
        ),
    ]

    mean_or_vote = "vote"
    if mean_or_vote == "mean":
        t += [
            EnsureTyped(keys=prediction_keys),
            MeanEnsembled(
                keys=prediction_keys,
                output_key="pred",
            ),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
        ]
    else:
        t += [
            EnsureTyped(keys=prediction_keys),
            Activationsd(keys=prediction_keys, softmax=True),
            AsDiscreted(keys=prediction_keys, argmax=True),
            VoteEnsembled(keys=prediction_keys, output_key="pred"),
        ]
    t += [
        SaveImaged(
            keys="pred",
            writer="ITKWriter",
            output_postfix="",
            output_ext=".nii.gz",
            folder_layout=nii_layout,
            output_dtype=np.uint8,
            separate_folder=False,
            resample=False,
        ),
    ]
    return Compose(t)


def get_val_post_transforms(labels, device):
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys=("pred"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        # This transform is to check dice score per segment/label
        SplitPredsLabeld(keys="pred"),
    ]
    return Compose(t)




def get_filename_without_extensions(nifti_path):
    # Strips up to two extensions from the filename, e.g. SUV.nii.gz -> SUV
    return Path(os.path.basename(nifti_path)).with_suffix("").with_suffix("").name



def get_TriALS_file_list(args, test=False) -> List[List, List, List]:
    if test:

        train_images = [args.input_dir]
        train_livers = [os.path.join(os.path.dirname(args.input_dir), 'livers', os.path.basename(args.input_dir))]
        data = [{"image": image_name, "liver": liver_name} for image_name, liver_name in zip(train_images, train_livers)]
        logger.info(f"{data[-5:]=}")
 
        return [], [], data


    train_images = sorted(glob.glob(os.path.join(args.input_dir, "imagesTr", "*.nii.gz")))


    train_labels = [
        os.path.join(args.input_dir, "labelsTr", os.path.basename(image).replace("_0000", "") )
        for image in train_images
    ]
    train_livers = sorted(glob.glob(os.path.join(args.input_dir, "livers", "*.nii.gz")))

    data = [{"image": image_name, "label": label_name, "liver": liver_name} for image_name, label_name, liver_name in zip(train_images, train_labels, train_livers)]

    logger.info(f"{data[-5:]=}")


    if args.val_fold != -1:
        partitions = partition_dataset(
            data,
            num_partitions=5,
            shuffle=True,
            seed=args.seed,
        )
        train_folds = [0, 1, 2, 3, 4]
        val_fold = args.val_fold
        train_folds.remove(val_fold) 
        train_data = select_cross_validation_folds(partitions, train_folds)
        val_data = select_cross_validation_folds(partitions, val_fold)
    else:
        train_data, val_data = partition_dataset(
            data,
            ratios=[args.split, (1 - args.split)],
            shuffle=True,
            seed=args.seed,
        )

    return train_data, val_data, []


def get_data(args, test=False):
    logger.info(f"{args.dataset=}")

    test_data = []
    if args.dataset == "TriALS":
        train_data, val_data, test_data = get_TriALS_file_list(args, test=test)

    if args.train_on_all_samples:
        train_data += val_data
        val_data = train_data
        test_data = train_data
        logger.warning("All validation data has been added to the training. Validation on them no longer makes sense.")

    logger.info(f"len(train_data): {len(train_data)}, len(val_data): {len(val_data)}, len(test_data): {len(test_data)}")

    # For debugging with small dataset size
    train_data = train_data[0 : args.limit] if args.limit else train_data
    val_data = val_data[0 : args.limit] if args.limit else val_data
    test_data = test_data[0 : args.limit] if args.limit else test_data

    return train_data, val_data, test_data


def get_test_loader(args, pre_transforms_test):
    train_data, val_data, test_data = get_data(args, test=True)
    if not len(test_data):
        if len(val_data) > 0:
            test_data = val_data
        elif len(train_data) > 0:
            test_data = train_data
        else:
            raise UserWarning("No valid data found..")

    total_l = len(test_data)
    test_ds = Dataset(test_data, pre_transforms_test)
    test_loader = DataLoader(
        test_ds,
        # shuffle=True,
        # num_workers=args.num_workers,
        batch_size=1,
        # The two options below are needed if ToDeviced('cuda' ,..) is activated..
        # multiprocessing_context="spawn",
        # persistent_workers=True,
    )
    logger.info("{} :: Total Records used for Testing is: {}".format(args.gpu, total_l))

    return test_loader


def get_train_loader(args, pre_transforms_train):
    train_data, val_data, test_data = get_data(args)
    total_l = len(train_data) + len(val_data)

    train_ds = PersistentDataset(train_data, pre_transforms_train, cache_dir=args.cache_dir)
    train_loader = ThreadDataLoader(
        train_ds,
        shuffle=True,
        num_workers=args.num_workers,
        batch_size=1,
        # The two options below are needed if ToDeviced('cuda' ,..) is activated..
        # multiprocessing_context="spawn",
        # persistent_workers=True,
    )
    logger.info("{} :: Total Records used for Training is: {}/{}".format(args.gpu, len(train_ds), total_l))

    return train_loader


def get_val_loader(args, pre_transforms_val):
    train_data, val_data, test_data = get_data(args)


    total_l = len(train_data) + len(val_data)

    val_ds = PersistentDataset(val_data, pre_transforms_val, cache_dir=args.cache_dir)
    val_loader = ThreadDataLoader(
        val_ds,
        num_workers=args.num_workers,
        batch_size=1,
        # multiprocessing_context="spawn",
        # persistent_workers=True,
    )
    logger.info("{} :: Total Records used for Validation is: {}/{}".format(args.gpu, len(val_ds), total_l))

    return val_loader


def get_cross_validation(args, nfolds, pre_transforms_train, pre_transforms_val):
    folds = list(range(nfolds))

    train_data, val_data, test_data = get_data(args)

    cvdataset = CrossValidation(
        dataset_cls=PersistentDataset,
        data=train_data,
        nfolds=nfolds,
        seed=args.seed,
        transform=pre_transforms_train,
        cache_dir=args.cache_dir,
    )

    train_dss = [cvdataset.get_dataset(folds=folds[0:i] + folds[(i + 1) :]) for i in folds]
    val_dss = [cvdataset.get_dataset(folds=i, transform=pre_transforms_val) for i in range(nfolds)]

    train_loaders = [
        ThreadDataLoader(
            train_dss[i],
            shuffle=True,
            num_workers=args.num_workers,
            batch_size=1,
        )
        for i in folds
    ]

    val_loaders = [
        ThreadDataLoader(
            val_dss[i],
            num_workers=args.num_workers,
            batch_size=1,
        )
        for i in folds
    ]

    return train_loaders, val_loaders  # , test_loader


def get_metrics_loader(args, file_glob="*.nii.gz"):
    labels_dir = args.labels_dir
    predictions_dir = args.predictions_dir
    predictions_glob = os.path.join(predictions_dir, file_glob)
    test_predictions = sorted(glob.glob(predictions_glob))
    test_datalist = []

    for pred_file_name in test_predictions:
        logger.info(f"{pred_file_name=}")
        assert os.path.exists(pred_file_name)
        file_name = get_filename_without_extensions(pred_file_name)
        label_file_name = os.path.join(labels_dir, f"{file_name}{file_glob[1:]}")
        assert os.path.exists(label_file_name)
        logger.info(f"{label_file_name=}")
        test_datalist.append({CommonKeys.LABEL: label_file_name, CommonKeys.PRED: pred_file_name})

    test_datalist = test_datalist[0 : args.limit] if args.limit else test_datalist
    total_l = len(test_datalist)
    assert total_l > 0

    logger.info("{} :: Total Records used for Dataloader is: {}".format(args.gpu, total_l))

    return test_datalist


def get_metrics_transforms(device, labels, args):
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    t = [
        InitLoggerd(loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir),
        LoadImaged(
            keys=["pred", "label"],
            reader="ITKReader",
            image_only=False,
        ),
        ToDeviced(keys=["pred", "label"], device=device),
        EnsureChannelFirstd(keys=["pred", "label"]),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(False, False),
            to_onehot=(len(labels), len(labels)),
        ),
    ]

    return Compose(t)
