import argparse
import os

import nibabel as nib

import cc3d
import numpy as np
import cupy as cp
from cucim.core.operations import morphology
import json


    
# For click visualization 
def generate_gaussian_heatmap(coords, shape, sigma=2.0):
    from scipy.ndimage import gaussian_filter
    """
    Generate a 3D Gaussian heatmap for given coordinates.

    Args:
        coords (list): List of [x, y, z] coordinates.
        shape (tuple): Shape of the output volume.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: 3D volume with Gaussian heatmaps at the specified coordinates.
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    for coord in coords:
        if 0 <= coord[0] < shape[0] and 0 <= coord[1] < shape[1] and 0 <= coord[2] < shape[2]:
            heatmap[tuple(coord)] = 1.0

    heatmap = gaussian_filter(heatmap, sigma=sigma)    
    return heatmap

def save_click_heatmaps(clicks, ref_shape, ref_affine, debug_output, input_label):
    tumor_coords = clicks['tumor']
    non_tumor_coords = clicks['background']

    tumor_heatmap = generate_gaussian_heatmap(tumor_coords, ref_shape, 1)
    non_tumor_heatmap = generate_gaussian_heatmap(non_tumor_coords, ref_shape, 1)

    tumor_nifti = nib.Nifti1Image(tumor_heatmap, ref_affine)
    non_tumor_nifti = nib.Nifti1Image(non_tumor_heatmap, ref_affine)

    nib.save(tumor_nifti, os.path.join(debug_output, f'{input_label.split("/")[-1].split(".nii.gz")[0]}_fg.nii.gz')) # foreground clicks
    nib.save(non_tumor_nifti, os.path.join(debug_output, f'{input_label.split("/")[-1].split(".nii.gz")[0]}_bg.nii.gz')) # background clicks

def perturb_click(offset, click, label_im):
    import random
    random_offset = [random.randint(0, int(offset)) for _ in range(3)]
    if label_im[
        int(click[0] + random_offset[0]),
        int(click[1] + random_offset[1]),
        int(click[2] + random_offset[2])
    ]:
        return [
            int(click[0] + random_offset[0]),
            int(click[1] + random_offset[1]),
            int(click[2] + random_offset[2])
        ]
    else:
        # Fallback to original click if perturbed click is invalid
        return [int(click[0]), int(click[1]), int(click[2])]

def uniform_sample_coordinates(in_liver, label_im, num_samples=10):
    """
    Samples `num_samples` coordinates from `in_liver` where `label_im == 0`.

    Args:
        in_liver (numpy.ndarray): 3D binary mask of the liver (1 = liver, 0 = outside).
        label_im (numpy.ndarray): 3D label image where 0 indicates the region of interest.
        num_samples (int): Number of points to sample.

    Returns:
        numpy.ndarray: Array of shape (num_samples, 3) containing sampled coordinates.
    """
    # Get coordinates where the liver is present and label_im == 0

    valid_coords = np.argwhere((in_liver == 1) & (label_im == 0))

    # Ensure there are points to sample from
    if len(valid_coords) == 0:
        raise ValueError("No valid voxels found where in_liver == 1 and label_im == 0.")

    # Randomly sample from available coordinates
    sampled_indices = np.random.choice(len(valid_coords), size=min(num_samples, len(valid_coords)), replace=False)
    sampled_coords = valid_coords[sampled_indices]
    sampled_coords = [[int(el[0]), int(el[1]), int(el[2])] for el in sampled_coords]

    return sampled_coords

def simulate_clicks(args):


    SEED = 42
    np.random.seed(SEED)
    #cp.random.seed(SEED)

    label_im = nib.load(args.input_label)
    ref_shape = label_im.shape
    ref_affine = label_im.affine
    label_im = label_im.get_fdata()
    clicks = {'tumor':[], 'background': []}


    if np.sum(label_im) == 0:
        print("[WARNING] GT is empty, generating background clicks only!")
    else: 
        ##### Tumor Clicks #####
        connected_components = cc3d.connected_components(label_im, connectivity=26)
        unique_labels = np.unique(connected_components)[1:] # Skip background label 0
        size = min(10, len(unique_labels))
        sampled_labels = np.random.choice(unique_labels, size=size, replace=False)

        # Sample center clicks for 10 random components
        for label in sampled_labels:    
            labeled_mask = connected_components == label
            labeled_mask = cp.array(labeled_mask)
            try:
                # Attempt to compute EDT using GPU
                edt = morphology.distance_transform_edt(labeled_mask)
            except MemoryError:
                from scipy import ndimage

                print("Out of memory on GPU! Applying CPU-based EDT. This might be a bit slower...")
                edt = ndimage.morphology.distance_transform_edt(labeled_mask.get())
                edt = cp.array(edt)
                labeled_mask = cp.array(labeled_mask)


            center = cp.unravel_index(cp.argmax(edt), edt.shape)
            if args.center_offset is not None:
                center = perturb_click(args.center_offset, center, label_im)

            clicks['tumor'].append([int(center[0]), int(center[1]), int(center[2])])
            assert label_im[int(center[0]), int(center[1]), int(center[2])]
        n_clicks = len(clicks['tumor'])

        # Sample boundary clicks if center clicks were not enough to fill the click budget (n=10)
        while n_clicks < 10:
            for label in sampled_labels: 
                labeled_mask = connected_components == label
                labeled_mask = cp.array(labeled_mask)
                try:
                    # Attempt to compute EDT using GPU
                    edt = morphology.distance_transform_edt(labeled_mask)
                except MemoryError:
                    from scipy import ndimage

                    print("Out of memory on GPU! Applying CPU-based EDT. This might be a bit slower...")
                    edt = ndimage.morphology.distance_transform_edt(labeled_mask.get())
                    edt = cp.array(edt)
                    labeled_mask = cp.array(labeled_mask)
                edt_inverted = (cp.max(edt) - edt) * (edt > 0) 
                boundary_elements = (edt_inverted == cp.max(edt_inverted)) * (labeled_mask > 0)
                indices = cp.array(cp.nonzero(boundary_elements)).T.get()  # Shape: (num_true, ndim)
                boundary_click = indices[np.random.choice(indices.shape[0])]

                if args.edge_offset is not None:
                    boundary_click = perturb_click(args.edge_offset, boundary_click, label_im)

                clicks['tumor'].append([int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])])
                assert label_im[int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])]
                n_clicks += 1
                if n_clicks == 10:
                    break
    
    ##### Background Clicks #####
    in_liver = nib.load(args.input_liver).get_fdata()

    bg_clicks = uniform_sample_coordinates(in_liver, label_im) # sample non-tumor clicks in the liver
    clicks['background'] = bg_clicks

    if args.debug_output is not None:
        os.makedirs(args.debug_output, exist_ok = True)
        save_click_heatmaps(clicks, ref_shape, ref_affine, args.debug_output, args.input_label)

    os.makedirs(args.json_output, exist_ok=True)

    with open(os.path.join(args.json_output, args.input_label.replace('.nii.gz', '_clicks.json').split('/')[-1]), 'w') as f:
        json.dump(clicks, f)
    print('Done with writing', os.path.join(args.json_output, args.input_label.replace('.nii.gz', '_clicks.json').split('/')[-1]))
    return clicks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_label", required=True, help="Path to the nifti label")
    parser.add_argument("-i_ct", "--input_ct", required=False, help="Path to the nifti ct image")
    parser.add_argument("-i_liver", "--input_liver", required=False, help="Path to the liver (pseudo-)label")

    parser.add_argument("--debug_output", required=False, help="Output path for click visualization for debugging")
    parser.add_argument("--json_output", required=True, help="Output path for JSON containing all clicks")
    parser.add_argument("--center_offset", required=False, help="Random perturbation for center clicks (could be used as data augmentation for training)")
    parser.add_argument("--edge_offset", required=False, help="Random perturbation for edge clicks (could be used as data augmentation for training)")

    args = parser.parse_args()

    simulate_clicks(args)
