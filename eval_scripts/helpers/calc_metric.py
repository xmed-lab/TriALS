from medpy import metric
import numpy as np
from scipy import ndimage
import time

from .surface import Surface


LARGE = 9001


def dice(input1, input2):
    return metric.dc(input1, input2)


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
    
    for i, obj_id in enumerate(np.unique(prediction_mask)):
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
