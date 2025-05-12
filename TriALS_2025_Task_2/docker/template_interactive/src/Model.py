import torch
import os
from sw_fastedit_src.simplified_inference import simplified_predict
# predict a numpy array
import numpy as np


class MySegmentation:
    def __init__(self, task="Dataset102_TriALS",
                 model_name='SW_FastEdit',
                 ):
        print('Model initialization done!')

    def process_image(self, image_np, properties, input_click_dict, input_output_paths):
        input_image_path, output_path  = input_output_paths[0], input_output_paths[1]
        pred_np = simplified_predict(input_image_path, output_path, input_click_dict)
        return pred_np
        


