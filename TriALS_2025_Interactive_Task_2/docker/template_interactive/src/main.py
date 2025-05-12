import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'sw_fastedit_src'))
from os.path import join as osjoin
import csv
from Model import MySegmentation
import numpy as np
import json
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

if len(sys.argv) != 4:
    raise (RuntimeError(f"Expected 3 arguments, was provided {len(sys.argv) - 1} argument(s)."))

test_csv_path = sys.argv[1]
input_dir_path = sys.argv[2]
output_dir_path = sys.argv[3]

print("=" * 30)
print("Running segmentation:")
print(f"  For IDs listed in {test_csv_path}")
print(f"  Using images under {input_dir_path}")
print(f"  Storing predictions under {output_dir_path}")
print("=" * 30)

# check csv file
if not os.path.exists(test_csv_path):
    raise (FileNotFoundError(f"Could not find csv file: {test_csv_path}"))

# check folders
if not os.path.exists(input_dir_path):
    raise (NotADirectoryError(f"Could not find directory: {input_dir_path}"))

if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

# read csv file containing file identifiers
# csv file contains a single column specifying the identifiers for the images
# such that the input image filename can be constructed as venous_<identifier>.nii.gz
with open(test_csv_path, "r") as csvfile:
    reader_obj = csv.reader(csvfile)
    orders = list(reader_obj)

model = MySegmentation()

row_counter = 0
for row in orders:
    input_image_file = osjoin(input_dir_path, f"{row[0]}_0000.nii.gz")
    input_click_file = input_image_file.replace('images', 'clicks').replace('0000', 'clicks').replace('nii.gz', 'json')

    output_path = os.path.join(output_dir_path, f"{row[0]}")
    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(input_image_file):
        FileNotFoundError(f"Could not find input image at: {input_image_file}")

    if not os.path.exists(input_click_file):
        FileNotFoundError(f"Could not find input clicks at: {input_click_file}")
    with open(input_click_file, 'r') as f:
        input_click_dict = json.load(f)

    print(f"Segmenting image {row_counter:03d}: {row[0]}_0000.nii.gz")

    image_np, properties = SimpleITKIO().read_images([input_image_file])

    input_output_paths = [input_image_file, output_path]

    for n_clicks in np.arange(0, 11):
        print(f'Predicting for {n_clicks} clicks...')
        input_click_dict_n_clicks = {'lesion': input_click_dict['lesion'][:n_clicks], 'background': input_click_dict['background'][:n_clicks] }
        pred_labels = model.process_image(image_np, properties, input_click_dict_n_clicks, input_output_paths
        )
        assert image_np.shape[-3:] == pred_labels.shape[-3:] 
        pred_labels = pred_labels.reshape(pred_labels.shape[-3:]) # remove batch dim

        SimpleITKIO().write_seg(pred_labels, os.path.join(output_path, f"{row[0]}_{n_clicks}.nii.gz"), properties)


    print("Done.")

    row_counter += 1
