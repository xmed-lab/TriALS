#!/usr/bin/env python
# from __future__ import print_function
import os
import csv
import nibabel as nb
import numpy as np
from scipy.ndimage.measurements import label as label_connected_components
import glob
import gc
from helpers.calc_metric import dice, detect_lesions, compute_segmentation_scores, compute_tumor_burden, LARGE
from helpers.utils import time_elapsed
import argparse



# Set up argument parser
parser = argparse.ArgumentParser(description='Process input paths and configurations.')

parser.add_argument('--fold_number', nargs='*', default=["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"], help='Fold number(s) for the model, e.g., fold_0. Defaults to all folds.')
parser.add_argument('--base_prediction_path', type=str, required=True, help='Base prediction path')
parser.add_argument('--truth_base_path', type=str, required=True, help='Truth base path')
parser.add_argument('--output_base_path', type=str, required=True, help='Output base path')
parser.add_argument('--model_folders', type=str, nargs='*', default=[
    "nnUNetTrainerLightMUNet__nnUNetPlans__3d_fullres",
    "nnUNetTrainerSwinUNETR__nnUNetPlans__3d_fullres",
    "nnUNetTrainer__nnUNetPlans__2d",
    "nnUNetTrainer__nnUNetPlans__3d_fullres",
    "nnUNetTrainerSegResNet__nnUNetPlans__3d_fullres",
    "nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres",
    "nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres",
    "nnUNetTrainerV2_MedNeXt_B_kernel5__nnUNetPlans__3d_fullres",
    "nnUNetTrainerV2_SAMed_b_r_4__nnUNetPlans__2d_p256",
    "nnUNetTrainerV2_SAMed_h_r_4__nnUNetPlans__2d_p512"
], help='List of model folders. Defaults to a predefined list of model folders.')



args = parser.parse_args()

# Assigning parsed arguments to variables
fold_number_list = args.fold_number
base_prediction_path = args.base_prediction_path
truth_base_path = args.truth_base_path
output_base_path = args.output_base_path
model_folders = args.model_folders  # Will default to the provided list if not specified

# Check and create output directory if needed.
for model_folder in model_folders:
    print(f"Processing {model_folder}")
    for fold in fold_number_list:
        print(f" - Processing {fold}")
        segmentation_metrics = {'dice': 0,
                                'jaccard': 0,
                                'voe': 1,
                                'rvd': LARGE,
                                'assd': LARGE,
                                'rmsd': LARGE,
                                'msd': LARGE}

        # Initialize results dictionaries
        lesion_detection_stats = {0: {'TP': 0, 'FP': 0, 'FN': 0},
                                  0.5: {'TP': 0, 'FP': 0, 'FN': 0}}
        lesion_segmentation_scores = {}
        liver_segmentation_scores = {}
        dice_per_case = {'lesion': [], 'liver': []}
        image_name_list=[]
        dice_global_x = {'lesion': {'I': 0, 'S': 0},
                         'liver': {'I': 0, 'S': 0}}  # 2*I/S
        tumor_burden_list = []

        # Initialize or reset metrics dictionaries for each fold.
        fold_metrics = []
        # TODO for particpants iterate over the groundtruth
        submit_dir = os.path.join(base_prediction_path, model_folder, fold, "validation")
        # print(submit_dir)
        predicted_volume_list = sorted(glob.glob(submit_dir + '/*.nii.gz'))
        # print(predicted_volume_list)
        truth_dir = truth_base_path
        # try:
        for predicted_volume_fn in predicted_volume_list:
            # Extract the base filename of the predicted volume.
            predicted_volume_fn_base=os.path.basename(predicted_volume_fn)
            submission_volume_path = os.path.join(submit_dir, predicted_volume_fn_base)
            reference_volume_fn = os.path.join(truth_dir, predicted_volume_fn_base)
            print(submission_volume_path, reference_volume_fn )
            if os.path.exists(reference_volume_fn):
                print(f"Found corresponding reference file for {predicted_volume_fn_base}")
                image_name_list.append(predicted_volume_fn_base)

                t = time_elapsed()

                # Load reference and submission volumes with Nibabel.
                reference_volume = nb.load(reference_volume_fn)
                submission_volume = nb.load(submission_volume_path)

                # Get the current voxel spacing.
                voxel_spacing = reference_volume.header.get_zooms()[:3]

                # Get Numpy data and compress to int8.
                reference_volume = (reference_volume.get_fdata()).astype(np.int8)
                submission_volume = (submission_volume.get_fdata()).astype(np.int8)
                print(submission_volume.shape, reference_volume.shape)
                # Ensure that the shapes of the masks match.
                if submission_volume.shape != reference_volume.shape:
                    raise AttributeError("Shapes do not match! Prediction mask {}, "
                                         "ground truth mask {}"
                                         "".format(submission_volume.shape,
                                                   reference_volume.shape))
                print("Done loading files ({:.2f} seconds)".format(t()))

                # Create lesion and liver masks with labeled connected components.
                # (Assuming there is always exactly one liver - one connected comp.)
                pred_mask_lesion, num_predicted = label_connected_components( \
                    submission_volume == 2, output=np.int16)
                true_mask_lesion, num_reference = label_connected_components( \
                    reference_volume == 2, output=np.int16)
                pred_mask_liver = submission_volume >= 1
                true_mask_liver = reference_volume >= 1
                liver_prediction_exists = np.any(submission_volume == 1)
                print("Done finding connected components ({:.2f} seconds)".format(t()))

                # Identify detected lesions.
                # Retain detected_mask_lesion for overlap > 0.5
                for overlap in [0, 0.5]:
                    detected_mask_lesion, mod_ref_mask, num_detected = detect_lesions( \
                        prediction_mask=pred_mask_lesion,
                        reference_mask=true_mask_lesion,
                        min_overlap=overlap)

                    # Count true/false positive and false negative detections.
                    lesion_detection_stats[overlap]['TP'] += num_detected
                    lesion_detection_stats[overlap]['FP'] += num_predicted - num_detected
                    lesion_detection_stats[overlap]['FN'] += num_reference - num_detected
                print("Done identifying detected lesions ({:.2f} seconds)".format(t()))

                # Compute segmentation scores for DETECTED lesions.
                if num_detected > 0:
                    lesion_scores = compute_segmentation_scores( \
                        prediction_mask=detected_mask_lesion,
                        reference_mask=mod_ref_mask,
                        voxel_spacing=voxel_spacing)
                    for metric in segmentation_metrics:
                        if metric not in lesion_segmentation_scores:
                            lesion_segmentation_scores[metric] = []
                        lesion_segmentation_scores[metric].extend(lesion_scores[metric])
                    print("Done computing lesion scores ({:.2f} seconds)".format(t()))
                else:
                    print("No lesions detected, skipping lesion score evaluation")

                # Compute liver segmentation scores.
                if liver_prediction_exists:
                    liver_scores = compute_segmentation_scores( \
                        prediction_mask=pred_mask_liver,
                        reference_mask=true_mask_liver,
                        voxel_spacing=voxel_spacing)
                    for metric in segmentation_metrics:
                        if metric not in liver_segmentation_scores:
                            liver_segmentation_scores[metric] = []
                        liver_segmentation_scores[metric].extend(liver_scores[metric])
                    print("Done computing liver scores ({:.2f} seconds)".format(t()))
                else:
                    # No liver label. Record default score values (zeros, inf).
                    # NOTE: This will make some metrics evaluate to inf over the entire
                    # dataset.
                    for metric in segmentation_metrics:
                        if metric not in liver_segmentation_scores:
                            liver_segmentation_scores[metric] = []
                        liver_segmentation_scores[metric].append( \
                            segmentation_metrics[metric])
                    print("No liver label provided, skipping liver score evaluation")

                # Compute per-case (per patient volume) dice.
                if not np.any(pred_mask_lesion) and not np.any(true_mask_lesion):
                    dice_per_case['lesion'].append(1.)
                else:
                    dice_per_case['lesion'].append(dice(pred_mask_lesion,
                                                        true_mask_lesion))
                if liver_prediction_exists:
                    dice_per_case['liver'].append(dice(pred_mask_liver,
                                                       true_mask_liver))
                else:
                    dice_per_case['liver'].append(0)

                # Accumulate stats for global (dataset-wide) dice score.
                dice_global_x['lesion']['I'] += np.count_nonzero( \
                    np.logical_and(pred_mask_lesion, true_mask_lesion))
                dice_global_x['lesion']['S'] += np.count_nonzero(pred_mask_lesion) + \
                                                np.count_nonzero(true_mask_lesion)
                if liver_prediction_exists:
                    dice_global_x['liver']['I'] += np.count_nonzero( \
                        np.logical_and(pred_mask_liver, true_mask_liver))
                    dice_global_x['liver']['S'] += np.count_nonzero(pred_mask_liver) + \
                                                   np.count_nonzero(true_mask_liver)
                else:
                    # NOTE: This value should never be zero.
                    dice_global_x['liver']['S'] += np.count_nonzero(true_mask_liver)

                print("Done computing additional dice scores ({:.2f} seconds)"
                      "".format(t()))

                # Compute tumor burden.
                tumor_burden = compute_tumor_burden(prediction_mask=submission_volume,
                                                    reference_mask=reference_volume)
                tumor_burden_list.append(tumor_burden)
                print("Done computing tumor burden diff ({:.2f} seconds)".format(t()))

                print("Done processing volume (total time: {:.2f} seconds)"
                      "".format(t.total_elapsed()))
                gc.collect()

            # else:
            #     print(f"No corresponding reference file found for {predicted_volume_fn_base}")
        # Your existing code for loading volumes, calculating metrics, etc., goes here.
        # At the end of processing each volume, append metrics to `fold_metrics`.

        # Example append (you'll replace this with actual metrics calculation):
        # fold_metrics.append({'volume': os.path.basename(reference_volume_fn), 'dice': 0.95, 'jaccard': 0.9, ...})
        # Compute lesion detection metrics.
        _det = {}
        for overlap in [0, 0.5]:
            TP = lesion_detection_stats[overlap]['TP']
            FP = lesion_detection_stats[overlap]['FP']
            FN = lesion_detection_stats[overlap]['FN']
            precision = float(TP) / (TP + FP) if TP + FP else 0
            recall = float(TP) / (TP + FN) if TP + FN else 0
            _det[overlap] = {'p': precision, 'r': recall}
        lesion_detection_metrics = {'precision': _det[0.5]['p'],
                                    'recall': _det[0.5]['r'],
                                    'precision_greater_zero': _det[0]['p'],
                                    'recall_greater_zero': _det[0]['r']}

        # Compute lesion segmentation metrics.
        lesion_segmentation_metrics = {}
        for m in lesion_segmentation_scores:
            lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        if len(lesion_segmentation_scores) == 0:
            # Nothing detected - set default values.
            lesion_segmentation_metrics.update(segmentation_metrics)
        lesion_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['lesion'])
        dice_global = 2. * dice_global_x['lesion']['I'] / dice_global_x['lesion']['S']
        lesion_segmentation_metrics['dice_global'] = dice_global

        # Compute liver segmentation metrics.
        liver_segmentation_metrics = {}
        for m in liver_segmentation_scores:
            liver_segmentation_metrics[m] = np.mean(liver_segmentation_scores[m])
        if len(liver_segmentation_scores) == 0:
            # Nothing detected - set default values.
            liver_segmentation_metrics.update(segmentation_metrics)
        liver_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['liver'])
        dice_global = 2. * dice_global_x['liver']['I'] / dice_global_x['liver']['S']
        liver_segmentation_metrics['dice_global'] = dice_global

        # Compute tumor burden.
        tumor_burden_rmse = np.sqrt(np.mean(np.square(tumor_burden_list)))
        tumor_burden_max = np.max(tumor_burden_list)

        # After processing all volumes in a fold, calculate average metrics.
        # avg_metrics = calculate_average_metrics(fold_metrics)  # Implement this function based on your needs.

        # Append average metrics for the fold to fold_metrics for CSV output.
        # fold_metrics.append({'volume': 'Average', **avg_metrics})
        output_fold_metrics_csv_path=os.path.join(output_base_path,'results')
        if not os.path.exists(output_fold_metrics_csv_path):
            os.makedirs(output_fold_metrics_csv_path)

        # Write fold metrics to a CSV file.
        print("Computed LESION DETECTION metrics:")
        for metric, value in lesion_detection_metrics.items():
            print("{}: {:.3f}".format(metric, float(value)))
        print("Computed LESION SEGMENTATION metrics (for detected lesions):")
        for metric, value in lesion_segmentation_metrics.items():
            print("{}: {:.3f}".format(metric, float(value)))
        print("Computed LIVER SEGMENTATION metrics:")
        for metric, value in liver_segmentation_metrics.items():
            print("{}: {:.3f}".format(metric, float(value)))
        print("Computed TUMOR BURDEN: \n"
              "rmse: {:.3f}\nmax: {:.3f}".format(tumor_burden_rmse, tumor_burden_max))

        output_csv_file = os.path.join(output_fold_metrics_csv_path, f"{model_folder}_{fold}_metrics.csv")

        # Open the CSV file for writing
        with open(output_csv_file, mode='w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            # Write the headers
            writer.writerow(['Metric', 'Value'])
            # Write the lesion detection metrics
            for metric, value in lesion_detection_metrics.items():
                writer.writerow([f"lesion_{metric}", value])  # Writing the value directly without formatting
            # Write the lesion segmentation metrics
            for metric, value in lesion_segmentation_metrics.items():
                writer.writerow([f"lesion_{metric}", value])  # Writing the value directly without formatting
            # Write the liver segmentation metrics
            for metric, value in liver_segmentation_metrics.items():
                writer.writerow([f"liver_{metric}", value])  # Writing the value directly without formatting

        new_output_base_path=os.path.join(output_base_path,'results','dice')
        if not os.path.exists(new_output_base_path):
            os.makedirs(new_output_base_path)

        new_output_csv_file = os.path.join(new_output_base_path, f"{model_folder}_{fold}_dice_scores.csv")

        # Open the CSV file for writing
        with open(new_output_csv_file, mode='w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)

            # Write the headers
            writer.writerow(['Image Name', 'Lesion Dice Score', 'Liver Dice Score'])

            # Iterate through each image name and write the corresponding dice scores
            for i, image_name in enumerate(image_name_list):
                lesion_dice_score = dice_per_case['lesion'][i] if i < len(dice_per_case['lesion']) else 'N/A'
                liver_dice_score = dice_per_case['liver'][i] if i < len(dice_per_case['liver']) else 'N/A'

                # Write the data row
                writer.writerow([image_name, lesion_dice_score, liver_dice_score])

        print(f" - Completed {fold}")
        # except:
        #     print(f" - Error in {model_folder} {fold}")
    print(f"Completed {model_folder}")

print("All models processed.")
