import sys
import os
import csv
import nibabel as nb
import numpy as np
from scipy.ndimage.measurements import label as label_connected_components
import gc
from helpers.calc_metric import dice, detect_lesions, compute_segmentation_scores, LARGE
from helpers.utils import time_elapsed



def evaluate_case(pred_file, gt_file):

    reference_volume = nb.load(gt_file)
    submission_volume = nb.load(pred_file)

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
        submission_volume == 1, output=np.int16)
    true_mask_lesion, num_reference = label_connected_components( \
        reference_volume == 1, output=np.int16)

    print("Done finding connected components ({:.2f} seconds)".format(t()))

    # Identify detected lesions.
    # Retain detected_mask_lesion for overlap > 0.5
    case_lesion_detection_stats = {0: {'TP': 0, 'FP': 0, 'FN': 0},
                                   0.5: {'TP': 0, 'FP': 0, 'FN': 0}}
    for overlap in [0, 0.5]:
        detected_mask_lesion, mod_ref_mask, num_detected = detect_lesions( \
            prediction_mask=pred_mask_lesion,
            reference_mask=true_mask_lesion,
            min_overlap=overlap)

        # Count true/false positive and false negative detections.
        case_lesion_detection_stats[overlap]['TP'] = num_detected
        case_lesion_detection_stats[overlap]['FP'] = num_predicted - num_detected
        case_lesion_detection_stats[overlap]['FN'] = num_reference - num_detected

    print("Done identifying detected lesions ({:.2f} seconds)".format(t()))

    # Compute segmentation scores for DETECTED lesions.
    lesion_scores = None
    if num_detected > 0:
        lesion_scores = compute_segmentation_scores( \
            prediction_mask=detected_mask_lesion,
            reference_mask=mod_ref_mask,
            voxel_spacing=voxel_spacing)
        print("Done computing lesion scores ({:.2f} seconds)".format(t()))
    else:
        print("No lesions detected, skipping lesion score evaluation")

    # Compute per-case (per patient volume) dice.
    if not np.any(pred_mask_lesion) and not np.any(true_mask_lesion):
        dice_score = 1.
    else:
        dice_score = dice(pred_mask_lesion,
                          true_mask_lesion)

    # Accumulate stats for global (dataset-wide) dice score.
    dice_per_case_global_acc_I = np.count_nonzero( \
        np.logical_and(pred_mask_lesion, true_mask_lesion))
    dice_case_global_accum_S = np.count_nonzero(pred_mask_lesion) + \
                               np.count_nonzero(true_mask_lesion)

    print("Done computing additional dice scores ({:.2f} seconds)"
          "".format(t()))

    print("Done processing volume (total time: {:.2f} seconds)"
          "".format(t.total_elapsed()))
    gc.collect()
    return lesion_scores, case_lesion_detection_stats, dice_score, dice_per_case_global_acc_I, dice_case_global_accum_S


if __name__ == "__main__":
    # the below 4 lines are the arguments that are passed using the command line
    test_csv_path = sys.argv[1]
    predicted_masks_path = sys.argv[2]
    groundtruth_masks_path = sys.argv[3]
    metrics_csv_path = sys.argv[4]

    lesion_detection_stats = {0: {'TP': 0, 'FP': 0, 'FN': 0},
                              0.5: {'TP': 0, 'FP': 0, 'FN': 0}}
    lesion_segmentation_scores = {}
    image_name_list = []
    dice_per_case = {'lesion': []}
    dice_global_x = {'lesion': {'I': 0, 'S': 0},
                     }
    segmentation_metrics = {'dice': 0,
                            'jaccard': 0,
                            'voe': 1,
                            'rvd': LARGE,
                            'assd': LARGE,
                            'rmsd': LARGE,
                            'msd': LARGE}

    os.makedirs(metrics_csv_path,exist_ok=True)

    cases_csv_path = os.path.join(metrics_csv_path, "per_case_metrics.csv")
    full_csv_path = os.path.join(metrics_csv_path, "metrics.csv")

    with open(test_csv_path, 'r') as reader_file:
        reader = csv.reader(reader_file)
        # loop through each row in the CSV file
        with open(cases_csv_path, mode='w', newline='') as writer_file:
            writer = csv.writer(writer_file)

            writer.writerow(["Image_gt", "Image_pm", "DSC", "precision", "recall", "precision_greater_zero",
                             "recall_greater_zero"])  # write header row

            for row in reader:
                image_hash = row[0]
                print(f"Evaluating volume: {image_hash} ...")
                # read ground truth mask image
                gt_path = os.path.join(groundtruth_masks_path, f"{image_hash}.nii.gz")
                if not os.path.exists(gt_path):
                    print('Ground Truth for volume {} does not exist!'.format(image_hash))
                    continue  # skip if ground truth mask does not exist

                # check if predicted mask exists
                pm_path = os.path.join(predicted_masks_path, f"{image_hash}.nii.gz")
                if os.path.exists(pm_path):
                    t = time_elapsed()
                    #per case metrics
                    lesion_scores, case_lesion_detection_stats, dice_score, dice_per_case_global_acc_I, dice_case_global_accum_S = evaluate_case(
                        pm_path, gt_path)

                    #accumate global metrics
                    dice_per_case['lesion'].append(dice_score)

                    _det_case = {}
                    for overlap in [0, 0.5]:
                        TP = case_lesion_detection_stats[overlap]['TP']
                        FP = case_lesion_detection_stats[overlap]['FP']
                        FN = case_lesion_detection_stats[overlap]['FN']

                        lesion_detection_stats[overlap]['TP'] += TP
                        lesion_detection_stats[overlap]['FP'] += FP
                        lesion_detection_stats[overlap]['FN'] += FN

                        precision = float(TP) / (TP + FP) if TP + FP else 0
                        recall = float(TP) / (TP + FN) if TP + FN else 0
                        _det_case[overlap] = {'p': precision, 'r': recall}

                    if lesion_scores:
                        for metric in segmentation_metrics:
                            if metric not in lesion_segmentation_scores:
                                lesion_segmentation_scores[metric] = []
                            lesion_segmentation_scores[metric].extend(lesion_scores[metric])

                    dice_global_x['lesion']['I'] += dice_per_case_global_acc_I
                    dice_global_x['lesion']['S'] += dice_case_global_accum_S

                    # get names of gt and pm images without extension
                    gt_name = f"{image_hash}"
                    pm_name = f"pred-{image_hash}"
                    # Iterate through each image name and write the corresponding dice scores
                    writer.writerow(
                        [gt_name, pm_name, dice_score, _det_case[0.5]['p'], _det_case[0.5]['r'], _det_case[0]['p'],
                         _det_case[0]['r']])

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

    # Write fold metrics to a CSV file.
    print("Computed LESION DETECTION metrics:")
    for metric, value in lesion_detection_metrics.items():
        print("{}: {:.3f}".format(metric, float(value)))
    print("Computed LESION SEGMENTATION metrics (for detected lesions):")
    for metric, value in lesion_segmentation_metrics.items():
        print("{}: {:.3f}".format(metric, float(value)))

    # Open the CSV file for writing
    with open(full_csv_path, mode='w', newline='') as file:
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
