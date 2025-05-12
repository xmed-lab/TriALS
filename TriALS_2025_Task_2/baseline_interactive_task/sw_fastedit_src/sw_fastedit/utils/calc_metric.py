from __future__ import annotations

from medpy import metric
import numpy as np
from scipy import ndimage
import argparse
import nibabel as nib
import os
import json
from multiprocessing import Pool, cpu_count
from scipy import integrate

LARGE = 9001

"""
@package medpy.metric.surface
Holds a metrics class computing surface metrics over two 3D-images contain each a binary object.

Classes:
    - Surface: Computes different surface metrics between two 3D-images contain each an object.

@author Oskar Maier
@version r0.4.1
@since 2011-12-01
@status Release
"""

# build-in modules
import math

# third-party modules
import scipy.spatial
import scipy.ndimage


# own modules

# code
class Surface(object):
    """
    Computes different surface metrics between two 3D-images contain each an object.
    The surface of the objects is computed using a 18-neighbourhood edge detection.
    The distance metrics are computed over all points of the surfaces using the nearest
    neighbour approach.
    Beside this provides a number of statistics of the two images.

    During the initialization the edge detection is run for both images, taking up to
    5 min (on 512^3 images). The first call to one of the metric measures triggers the
    computation of the nearest neighbours, taking up to 7 minutes (based on 250.000 edge
    point for each of the objects, which corresponds to a typical liver mask). All
    subsequent calls to one of the metrics measures can be expected be in the
    sub-millisecond area.

    Metrics defined in:
    Heimann, T.; van Ginneken, B.; Styner, M.A.; Arzhaeva, Y.; Aurich, V.; Bauer, C.; Beck, A.; Becker, C.; Beichel, R.; Bekes, G.; Bello, F.; Binnig, G.; Bischof, H.; Bornik, A.; Cashman, P.; Ying Chi; Cordova, A.; Dawant, B.M.; Fidrich, M.; Furst, J.D.; Furukawa, D.; Grenacher, L.; Hornegger, J.; Kainmuller, D.; Kitney, R.I.; Kobatake, H.; Lamecker, H.; Lange, T.; Jeongjin Lee; Lennon, B.; Rui Li; Senhu Li; Meinzer, H.-P.; Nemeth, G.; Raicu, D.S.; Rau, A.-M.; van Rikxoort, E.M.; Rousson, M.; Rusko, L.; Saddi, K.A.; Schmidt, G.; Seghers, D.; Shimizu, A.; Slagmolen, P.; Sorantin, E.; Soza, G.; Susomboon, R.; Waite, J.M.; Wimmer, A.; Wolf, I.; , "Comparison and Evaluation of Methods for Liver Segmentation From CT Datasets," Medical Imaging, IEEE Transactions on , vol.28, no.8, pp.1251-1265, Aug. 2009
    doi: 10.1109/TMI.2009.2013851
    """

    # The edge points of the mask object.
    __mask_edge_points = None
    # The edge points of the reference object.
    __reference_edge_points = None
    # The nearest neighbours distances between mask and reference edge points.
    __mask_reference_nn = None
    # The nearest neighbours distances between reference and mask edge points.
    __reference_mask_nn = None
    # Distances of the two objects surface points.
    __distance_matrix = None

    def __init__(self, mask, reference, physical_voxel_spacing=[1, 1, 1], mask_offset=[0, 0, 0],
                 reference_offset=[0, 0, 0]):
        """
        Initialize the class with two binary images, each containing a single object.
        Assumes the input to be a representation of a 3D image, that fits one of the
        following formats:
            - 1. all 0 values denoting background, all others the foreground/object
            - 2. all False values denoting the background, all others the foreground/object
        The first image passed is referred to as 'mask', the second as 'reference'. This
        is only important for some metrics that are not symmetric (and therefore not
        really metrics).
        @param mask binary mask as an scipy array (3D image)
        @param reference binary reference as an scipy array (3D image)
        @param physical_voxel_spacing The physical voxel spacing of the two images
            (must be the same for both)
        @param mask_offset offset of the mask array to 0,0,0-origin
        @param reference_offset offset of the reference array to 0,0,0-origin
        """
        # compute edge images
        mask_edge_image = Surface.compute_contour(mask)
        reference_edge_image = Surface.compute_contour(reference)

        # collect the object edge voxel positions
        # !TODO: When the distance matrix is already calculated here
        # these points don't have to be actually stored, only their number.
        # But there might be some later metric implementation that requires the
        # points and then it would be good to have them. What is better?
        mask_pts = mask_edge_image.nonzero()
        mask_edge_points = list(zip(mask_pts[0], mask_pts[1], mask_pts[2]))
        reference_pts = reference_edge_image.nonzero()
        reference_edge_points = list(zip(reference_pts[0], reference_pts[1], reference_pts[2]))

        # check if there is actually an object present
        if 0 >= len(mask_edge_points):
            raise Exception('The mask image does not seem to contain an object.')
        if 0 >= len(reference_edge_points):
            raise Exception('The reference image does not seem to contain an object.')

        # add offsets to the voxels positions and multiply with physical voxel spacing
        # to get the real positions in millimeters
        physical_voxel_spacing = np.array(physical_voxel_spacing)
        mask_edge_points += np.array(mask_offset)
        mask_edge_points *= physical_voxel_spacing
        reference_edge_points += np.array(reference_offset)
        reference_edge_points *= physical_voxel_spacing

        # set member vars
        self.__mask_edge_points = mask_edge_points
        self.__reference_edge_points = reference_edge_points

    def get_maximum_symmetric_surface_distance(self):
        """
        Computes the maximum symmetric surface distance, also known as Hausdorff
        distance, between the two objects surfaces.

        @return the maximum symmetric surface distance in millimeters

        For a perfect segmentation this distance is 0. This metric is sensitive to
        outliers and returns the true maximum error.

        Metric definition:
        Let \f$S(A)\f$ denote the set of surface voxels of \f$A\f$. The shortest
        distance of an arbitrary voxel \f$v\f$ to \f$S(A)\f$ is defined as:
        \f[
            d(v,S(A)) = \min_{s_A\in S(A)} ||v-s_A||
        \f]
        where \f$||.||\f$ denotes the Euclidean distance. The maximum symmetric
        surface distance is then given by:
        \f[
            MSD(A,B) = \max
                \left\{
                    \max_{s_A\in S(A)} d(s_A,S(B)),
                    \max_{s_B\in S(B)} d(s_B,S(A)),
                \right\}
        \f]
        """
        # Get the maximum of the nearest neighbour distances
        A_B_distance = self.get_mask_reference_nn().max()
        B_A_distance = self.get_reference_mask_nn().max()

        # compute result and return
        return max(A_B_distance, B_A_distance)

    def get_root_mean_square_symmetric_surface_distance(self):
        """
        Computes the root mean square symmetric surface distance between the
        two objects surfaces.

        @return root mean square symmetric surface distance in millimeters

        For a perfect segmentation this distance is 0. This metric punishes large
        deviations from the true contour stronger than the average symmetric surface
        distance.

        Metric definition:
        Let \f$S(A)\f$ denote the set of surface voxels of \f$A\f$. The shortest
        distance of an arbitrary voxel \f$v\f$ to \f$S(A)\f$ is defined as:
        \f[
            d(v,S(A)) = \min_{s_A\in S(A)} ||v-s_A||
        \f]
        where \f$||.||\f$ denotes the Euclidean distance. The root mean square
        symmetric surface distance is then given by:
        \f[
          RMSD(A,B) =
            \sqrt{\frac{1}{|S(A)|+|S(B)|}}
            \times
            \sqrt{
                \sum_{s_A\in S(A)} d^2(s_A,S(B))
                +
                \sum_{s_B\in S(B)} d^2(s_B,S(A))
            }
        \f]
        """
        # get object sizes
        mask_surface_size = len(self.get_mask_edge_points())
        reference_surface_sice = len(self.get_reference_edge_points())

        # get minimal nearest neighbours distances
        A_B_distances = self.get_mask_reference_nn()
        B_A_distances = self.get_reference_mask_nn()

        # square the distances
        A_B_distances_sqrt = A_B_distances * A_B_distances
        B_A_distances_sqrt = B_A_distances * B_A_distances

        # sum the minimal distances
        A_B_distances_sum = A_B_distances_sqrt.sum()
        B_A_distances_sum = B_A_distances_sqrt.sum()

        # compute result and return
        return math.sqrt(1. / (mask_surface_size + reference_surface_sice)) * math.sqrt(
            A_B_distances_sum + B_A_distances_sum)

    def get_average_symmetric_surface_distance(self):
        """
        Computes the average symmetric surface distance between the
        two objects surfaces.

        @return average symmetric surface distance in millimeters

        For a perfect segmentation this distance is 0.

        Metric definition:
        Let \f$S(A)\f$ denote the set of surface voxels of \f$A\f$. The shortest
        distance of an arbitrary voxel \f$v\f$ to \f$S(A)\f$ is defined as:
        \f[
            d(v,S(A)) = \min_{s_A\in S(A)} ||v-s_A||
        \f]
        where \f$||.||\f$ denotes the Euclidean distance. The average symmetric
        surface distance is then given by:
        \f[
            ASD(A,B) =
                \frac{1}{|S(A)|+|S(B)|}
                \left(
                    \sum_{s_A\in S(A)} d(s_A,S(B))
                    +
                    \sum_{s_B\in S(B)} d(s_B,S(A))
                \right)
        \f]
        """
        # get object sizes
        mask_surface_size = len(self.get_mask_edge_points())
        reference_surface_sice = len(self.get_reference_edge_points())

        # get minimal nearest neighbours distances
        A_B_distances = self.get_mask_reference_nn()
        B_A_distances = self.get_reference_mask_nn()

        # sum the minimal distances
        A_B_distances = A_B_distances.sum()
        B_A_distances = B_A_distances.sum()

        # compute result and return
        return 1. / (mask_surface_size + reference_surface_sice) * (A_B_distances + B_A_distances)

    def get_mask_reference_nn(self):
        """
        @return The distances of the nearest neighbours of all mask edge points to all
                reference edge points.
        """
        # Note: see note for @see get_reference_mask_nn
        if self.__mask_reference_nn is None:
            tree = scipy.spatial.cKDTree(self.get_mask_edge_points())
            self.__mask_reference_nn, _ = tree.query(self.get_reference_edge_points())
        return self.__mask_reference_nn

    def get_reference_mask_nn(self):
        """
        @return The distances of the nearest neighbours of all reference edge points
                to all mask edge points.

        The underlying algorithm used for the scipy.spatial.KDTree implementation is
        based on:
        Sunil Arya, David M. Mount, Nathan S. Netanyahu, Ruth Silverman, and
        Angela Y. Wu. 1998. An optimal algorithm for approximate nearest neighbor
        searching fixed dimensions. J. ACM 45, 6 (November 1998), 891-923
        """
        # Note: KDTree is faster than scipy.spatial.distance.cdist when the number of
        # voxels exceeds 10.000 (computationally tested). The maximum complexity is
        # O(D*N^2) vs. O(D*N*log(N), where D=3 and N=number of voxels
        if self.__reference_mask_nn is None:
            tree = scipy.spatial.cKDTree(self.get_reference_edge_points())
            self.__reference_mask_nn, _ = tree.query(self.get_mask_edge_points())
        return self.__reference_mask_nn

    def get_mask_edge_points(self):
        """
        @return The edge points of the mask object.
        """
        return self.__mask_edge_points

    def get_reference_edge_points(self):
        """
        @return The edge points of the reference object.
        """
        return self.__reference_edge_points

    @staticmethod
    def compute_contour(array):
        """
        Uses a 18-neighbourhood filter to create an edge image of the input object.
        Assumes the input to be a representation of a 3D image, that fits one of the
        following formats:
            - 1. all 0 values denoting background, all others the foreground/object
            - 2. all False values denoting the background, all others the foreground/object
        The area outside the array is assumed to contain background voxels. The method
        does not ensure that the object voxels are actually connected, this is silently
        assumed.

        @param array a numpy array with only 0/N\{0} or False/True values.
        @return a boolean numpy array with the input objects edges
        """
        # set 18-neighbourhood/conectivity (for 3D images) alias face-and-edge kernel
        # all values covered by 1/True passed to the function
        # as a 1D array in order left-right, top-down
        # Note: all in all 19 ones, as the center value
        # also has to be checked (if it is a masked pixel)
        # [[[0, 1, 0], [[1, 1, 1],  [[0, 1, 0],
        #   [1, 1, 1],  [1, 1, 1],   [1, 1, 1],
        #   [0, 1, 0]], [1, 1, 1]],  [0, 1, 0]]]
        footprint = scipy.ndimage.generate_binary_structure(3, 2)

        # create an erode version of the array
        erode_array = scipy.ndimage.binary_erosion(array, footprint)

        # xor the erode_array with the original and return
        return array ^ erode_array

def dice(input1, input2):
    return metric.dc(input1, input2)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def group_jsons_by_prefix(json_dir, predicted_case):
    grouped = []

    for fname in os.listdir(json_dir):
        if not fname.endswith('.json') or predicted_case != '_'.join(fname.split('_')[:-1]):
            continue
        full_path = os.path.join(json_dir, fname)
        grouped.append(full_path)

    return grouped

def aggregate_group_metrics(group_paths):
    all_scores = [load_json(p) for p in sorted(group_paths)]

    # keys are assumed the same across JSONs
    keys = all_scores[0].keys()
    aggregated = {}

    for k in keys:
        if k not in ['dice', 'assd', 'msd']:
            continue

        values = np.array([score[k][0] for score in all_scores])
        auc = integrate.cumulative_trapezoid(values, np.arange(11))[-1]
        final = values[-1]
        aggregated[k] = {
            "AUC": auc,
            "Final": final
        }

    return aggregated


def detect_lesions(prediction_mask, reference_mask, min_overlap=0.5):
    """
    Produces a mask for predicted lesions and a mask for reference lesions,
    with label IDs matching lesions together.
    
    Given a prediction and a reference mask, output a modified version of
    each where objects that overlap between the two mask share a label. This
    requires merging labels in the reference mask that are spanned by a single
    prediction and merging labels in the prediction mask that are spanned by
    a single reference. In cases where a label can be merged, separately, with
    more than one other label, a single merge option (label) is chosen to 
    accord the greatest overlap between the reference and prediction objects.
    
    After merging and matching, objects in the reference are considered
    detected if their respective predictions overlap them by more than
    `min_overlap` (intersection over union).
    
    :param prediction_mask: numpy.array
    :param reference_mask: numpy.array
    :param min_overlap: float in range [0, 1.]
    :return: prediction mask (int),
             reference mask (int),
             num_detected
    """
    
    # Initialize
    detected_mask = np.zeros(prediction_mask.shape, dtype=np.uint8)
    mod_reference_mask = np.copy(reference_mask)
    num_detected = 0
    if not np.any(reference_mask):
        return detected_mask, num_detected, 0
    
    if not min_overlap>0 and not min_overlap<=1:
        raise ValueError("min_overlap must be in [0, 1.]")
    
    # Get available IDs (excluding 0)
    # 
    # To reduce computation time, check only those lesions in the prediction 
    # that have any overlap with the ground truth.
    p_id_list = np.unique(prediction_mask[reference_mask.nonzero()])
    if p_id_list[0]==0:
        p_id_list = p_id_list[1:]
    g_id_list = np.unique(reference_mask)
    if g_id_list[0]==0:
        g_id_list = g_id_list[1:]
    
    # To reduce computation time, get views into reduced size masks.
    reduced_prediction_mask = rpm = prediction_mask.copy()
    for p_id in np.unique(prediction_mask):
        if p_id not in p_id_list and p_id!=0:
            reduced_prediction_mask[(rpm==p_id).nonzero()] = 0
    target_mask = np.logical_or(reference_mask, reduced_prediction_mask)
    bounding_box = ndimage.find_objects(target_mask)[0]
    r = reference_mask[bounding_box]
    p = prediction_mask[bounding_box]
    d = detected_mask[bounding_box]
    m = mod_reference_mask[bounding_box]

    # Compute intersection of predicted lesions with reference lesions.
    intersection_matrix = np.zeros((len(p_id_list), len(g_id_list)),
                                    dtype=np.int32)
    for i, p_id in enumerate(p_id_list):
        for j, g_id in enumerate(g_id_list):
            intersection = np.count_nonzero(np.logical_and(p==p_id, r==g_id))
            intersection_matrix[i, j] = intersection
    
    def sum_dims(x, axis, dims):
        '''
        Given an array x, collapses dimensions listed in dims along the 
        specified axis, summing them together. Returns the reduced array.
        '''
        x = np.array(x)
        if len(dims) < 2:
            return x
        
        # Initialize output
        new_shape = list(x.shape)
        new_shape[axis] -= len(dims)-1
        x_ret = np.zeros(new_shape, dtype=x.dtype)
        
        # Sum over dims on axis
        sum_slices = [slice(None)]*x.ndim
        sum_slices[axis] = dims
        dim_sum = np.sum(x[tuple(sum_slices)], axis=axis, keepdims=True)
        
        # Remove all but first dim in dims
        mask = np.ones(x.shape, dtype=np.bool_)
        mask_slices = [slice(None)]*x.ndim
        mask_slices[axis] = dims[1:]
        mask[tuple(mask_slices)] = 0
        x_ret.ravel()[...] = x[mask]
        
        # Put dim_sum into array at first dim
        replace_slices = [slice(None)]*x.ndim
        replace_slices[axis] = [dims[0]]
        x_ret[tuple(replace_slices)] = dim_sum
        
        return x_ret
            
    # Merge and label reference lesions that are connected by predicted
    # lesions.
    g_merge_count = dict([(g_id, 1) for g_id in g_id_list])
    for i, p_id in enumerate(p_id_list):
        # Identify g_id intersected by p_id
        g_id_indices = intersection_matrix[i].nonzero()[0]
        g_id_intersected = g_id_list[g_id_indices]
        
        # Make sure g_id are matched to p_id deterministically regardless of 
        # label order. Only merge those g_id which overlap this p_id more than
        # others.
        g_id_merge = []
        g_id_merge_indices = []
        for k, g_id in enumerate(g_id_intersected):
            idx = g_id_indices[k]
            if np.argmax(intersection_matrix[:, idx], axis=0)==i:
                # This g_id has the largest overlap with this p_id: merge.
                g_id_merge.append(g_id)
                g_id_merge_indices.append(idx)
                
        # Update merge count
        for g_id in g_id_merge:
            g_merge_count[g_id] = len(g_id_merge)
                
        # Merge. Update g_id_list, intersection matrix, mod_reference_mask.
        # Merge columns in intersection_matrix.
        g_id_list = np.delete(g_id_list, obj=g_id_merge_indices[1:])
        for g_id in g_id_merge:
            m[m==g_id] = g_id_merge[0]
        intersection_matrix = sum_dims(intersection_matrix,
                                       axis=1,
                                       dims=g_id_merge_indices)
    
    # Match each predicted lesion to a single (merged) reference lesion.
    max_val = np.max(intersection_matrix, axis=1)
    max_indices = np.argmax(intersection_matrix, axis=1)
    intersection_matrix[...] = 0
    intersection_matrix[np.arange(len(p_id_list)), max_indices] = max_val
    
    # Merge and label predicted lesions that are connected by reference
    # lesions.
    #
    # Merge rows in intersection_matrix.
    #
    # Here, it's fine to merge all p_id that are connected by a g_id since
    # each p_id has already been associated with only one g_id.
    for j, g_id in enumerate(g_id_list):
        p_id_indices = intersection_matrix[:,j].nonzero()[0]
        p_id_intersected = p_id_list[p_id_indices]
        intersection_matrix = sum_dims(intersection_matrix,
                                       axis=0,
                                       dims=p_id_indices)
        p_id_list = np.delete(p_id_list, obj=p_id_indices[1:])
        for p_id in p_id_intersected:
            d[p==p_id] = g_id
            
    # Trim away lesions deemed undetected.
    num_detected = len(p_id_list)
    for i, p_id in enumerate(p_id_list):
        for j, g_id in enumerate(g_id_list):
            intersection = intersection_matrix[i, j]
            if intersection==0:
                continue
            union = np.count_nonzero(np.logical_or(d==p_id, m==g_id))
            overlap_fraction = float(intersection)/union
            if overlap_fraction <= min_overlap:
                d[d==g_id] = 0      # Assuming one-to-one p_id <--> g_id
                num_detected -= g_merge_count[g_id]
                
    return detected_mask, mod_reference_mask, num_detected


def compute_tumor_burden(prediction_mask, reference_mask):
    """
    Calculates the tumor_burden and evalutes the tumor burden metrics RMSE and
    max error.
    
    :param prediction_mask: numpy.array
    :param reference_mask: numpy.array
    :return: dict with RMSE and Max error
    """
    def calc_tumor_burden(vol):
        num_liv_pix=np.count_nonzero(vol>=1)
        num_les_pix=np.count_nonzero(vol==2)
        return num_les_pix/float(num_liv_pix)
    tumor_burden_r = calc_tumor_burden(reference_mask)
    if np.count_nonzero(prediction_mask==1):
        tumor_burden_p = calc_tumor_burden(prediction_mask)
    else:
        tumor_burden_p = LARGE

    tumor_burden_diff = tumor_burden_r - tumor_burden_p
    return tumor_burden_diff


def compute_segmentation_scores(prediction_mask, reference_mask,
                                voxel_spacing):
    """
    Calculates metrics scores from numpy arrays and returns an dict.
    
    Assumes that each object in the input mask has an integer label that 
    defines object correspondence between prediction_mask and 
    reference_mask.
    
    :param prediction_mask: numpy.array, int
    :param reference_mask: numpy.array, int
    :param voxel_spacing: list with x,y and z spacing
    :return: dict with dice, jaccard, voe, rvd, assd, rmsd, and msd
    """
    
    scores = {'dice': [],
              'jaccard': [],
              'voe': [],
              'rvd': [],
              'assd': [],
              'rmsd': [],
              'msd': []}
    
    for i, obj_id in enumerate(np.unique(reference_mask)):
        if obj_id==0:
            continue    # 0 is background, not an object; skip

        # Limit processing to the bounding box containing both the prediction
        # and reference objects.
        target_mask = (reference_mask==obj_id)+(prediction_mask==obj_id)
        bounding_box = ndimage.find_objects(target_mask)[0]
        p = (prediction_mask==obj_id)[bounding_box]
        r = (reference_mask==obj_id)[bounding_box]
        if np.any(p) and np.any(r):
            dice = metric.dc(p,r)
            jaccard = dice/(2.-dice)
            scores['dice'].append(dice)
            scores['jaccard'].append(jaccard)
            scores['voe'].append(1.-jaccard)
            scores['rvd'].append(metric.ravd(r,p))
            evalsurf = Surface(p, r,
                               physical_voxel_spacing=voxel_spacing,
                               mask_offset=[0.,0.,0.],
                               reference_offset=[0.,0.,0.])
            assd = evalsurf.get_average_symmetric_surface_distance()
            rmsd = evalsurf.get_root_mean_square_symmetric_surface_distance()
            msd = evalsurf.get_maximum_symmetric_surface_distance()
            scores['assd'].append(assd)
            scores['rmsd'].append(rmsd)
            scores['msd'].append(msd)
        else:
            # There are no objects in the prediction, in the reference, or both
            scores['dice'].append(0)
            scores['jaccard'].append(0)
            scores['voe'].append(1.)
            
            # Surface distance (and volume difference) metrics between the two
            # masks are meaningless when any one of the masks is empty. Assign 
            # maximum penalty. The average score for these metrics, over all 
            # objects, will thus also not be finite as it also loses meaning.
            scores['rvd'].append(LARGE)
            scores['assd'].append(LARGE)
            scores['rmsd'].append(LARGE)
            scores['msd'].append(LARGE)

    return scores

def process_case(prediction_path_and_ref_dir):
    prediction_path, ref_dir, output_dir = prediction_path_and_ref_dir

    try:
        # Find matching reference path
        base_id = '_'.join(os.path.basename(prediction_path).split('_')[:-1])
        reference_path = os.path.join(ref_dir, base_id + '.nii.gz')
        # Load NIfTI files
        pred_nii = nib.load(prediction_path)
        ref_nii = nib.load(reference_path)

        prediction_mask = pred_nii.get_fdata()
        reference_mask = ref_nii.get_fdata()
        voxel_spacing = pred_nii.header.get_zooms()

        # Compute metrics
        scores = compute_segmentation_scores(prediction_mask, reference_mask, voxel_spacing)

        # Output path
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.basename(prediction_path).replace('.nii.gz', '.json')
        output_path = os.path.join(output_dir, base_filename)
        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"[✓] Saved: {output_path}")

    except Exception as e:
        print(f"[✗] Failed: {prediction_path} — {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute segmentation scores in parallel.')
    parser.add_argument('--predictions', type=str, required=True, help='Folder with predicted NIfTI (.nii.gz) files')
    parser.add_argument('--reference', type=str, required=True, help='Folder with reference NIfTI files')
    args = parser.parse_args()

    predicted_cases = [os.path.join(args.predictions, f) for f in os.listdir(args.predictions) if 'venous_' in f]
    print(f"{len(predicted_cases)} predicted cases!")

    for predicted_case in predicted_cases:
        print(f"Evaluating {predicted_case}")
        prediction_paths = [
            os.path.join(predicted_case, f)
            for f in os.listdir(predicted_case)
            if f.endswith('.nii.gz')
        ]

        output_dir = os.path.join(os.path.dirname(args.predictions), 'val_metrics')


        # Bundle paths into a list of tuples
        input_data = [(path, args.reference, output_dir) for path in prediction_paths]


        # Use multiprocessing Pool
        with Pool(processes=cpu_count()) as pool:
            pool.map(process_case, input_data)


        grouped_jsons = group_jsons_by_prefix(output_dir, os.path.basename(predicted_case))
        output_path = output_dir.replace('val_metrics', 'interactive_metrics')
        os.makedirs(output_path, exist_ok=True)
        print(f'\n  Case: {os.path.basename(predicted_case)} ({len(grouped_jsons)} files)')
        assert len(grouped_jsons) == 11
        agg = aggregate_group_metrics(grouped_jsons)
        output_json = os.path.join(output_path, f'{os.path.basename(predicted_case)}.json')
        with open(output_json, 'w') as f:
            json.dump(agg, f, indent=2)
        print(f"[✓] Saved: {output_json}")
    
    final_metric_jsons = [os.path.join(output_path, f) for f in os.listdir(output_path)]
    assert len(final_metric_jsons) == len(predicted_cases)
    mean_metrics = {
        "dice": {
            "AUC": 0,
            "Final": 0,
        },
        "assd": {
            "AUC": 0,
            "Final": 0
        },
        "msd": {
            "AUC": 0,
            "Final": 0
        }
    }
    for final_metric_json in final_metric_jsons:
        with open(final_metric_json, 'r') as f:
            final_metric = json.load(f)
            for key in mean_metrics:
                for subkey in mean_metrics[key]:
                    mean_metrics[key][subkey] += final_metric[key][subkey]

    # Now average
    num_files = len(final_metric_jsons)
    for key in mean_metrics:
        for subkey in mean_metrics[key]:
            mean_metrics[key][subkey] /= num_files

    final_json = os.path.join(args.predictions, 'final_interactive_metrics.json')
    with open(final_json, 'w') as f:
        json.dump(mean_metrics, f, indent=2)
    print(f"[✓] Saved: {final_json}")


