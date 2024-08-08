import sys
import os
from os.path import join as osjoin
import csv
from Model import MySegmentation
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
    input_image_path = osjoin(input_dir_path, f"{row[0]}_0000.nii.gz")

    if not os.path.exists(input_image_path):
        FileNotFoundError(f"Could not find input image at: {input_image_path}")

    #read the input volume
    image_np, properties = SimpleITKIO().read_images([input_image_path])

    print(f"Segmenting image {row_counter:03d}: {row[0]}_0000.nii.gz")

    #segment the volume
    pred_labels = model.process_image(
        image_np, properties)

    #write the segmentation volume
    SimpleITKIO().write_seg(pred_labels, os.path.join(output_dir_path, f"{row[0]}.nii.gz"), properties)

    print("Done.")

    row_counter += 1
