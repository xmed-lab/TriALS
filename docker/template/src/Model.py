import torch
import os
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
# predict a numpy array
import numpy as np


class MySegmentation:
    def __init__(self, task="Dataset102_TriALS", nnunet_model_dir='nnUNet_results',
                 model_name='nnUNetTrainer__nnUNetPlans__3d_fullres',
                 folds=(0, 1, 2, 3, 4)
                 ):
        # network parameters
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=True,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        self.predictor.initialize_from_trained_model_folder(
            os.path.join(nnunet_model_dir,
                         f'{task}/{model_name}'),
            use_folds=folds,
            checkpoint_name='checkpoint_final.pth',
        )

    def process_image(self, image_np, properties):
        ret = self.predictor.predict_single_npy_array(
            image_np, properties, None, None, False)

        # the lesion class should be set to 1 if it is not and others to 0
        ret = (ret == 2).astype(np.uint8)
        return ret
