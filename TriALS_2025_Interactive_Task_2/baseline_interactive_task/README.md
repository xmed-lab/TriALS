
# Baseline Model for [**TriALS 2025 Task 2: Interactive Click-based Portal-Venous Lesion Segmentation**](https://www.synapse.org/Synapse:syn65878273/wiki/631558#:~:text=Task%202%3A%20Interactive%20Click%2Dbased%20Portal%2DVenous%20Lesion%20Segmentation) 



### Step 1: Create Conda Environment and Download Model Weights
Create conda environment. **Prerequisite: Please make sure you have an installed GPU driver supporting CUDA 12.x!**
```bash
conda create -n TriALS_Task2_Baseline python=3.10 -y
conda activate TriALS_Task2_Baseline
```

Download the model weights from this [link](https://drive.google.com/file/d/1CP3vhsKag9u26k4AiuEaxiVhCMM3R50z/view?usp=sharing) and move them to `sw_fastedit_src/model.pt`

### Step 2: Install Requirements

Install the required packages using the provided `requirements.txt` file:

```bash
pip install -U monailabel
pip install -r requirements.txt
```

### Step 3: Liver segmentation
Our baseline requires liver masks as an input channel. We simply use TotalSegmentator to generate them with this command:

```bash
ls PATH_TO/imagesTr/ | xargs -I {} TotalSegmentator -i PATH_TO/imagesTr/{} -o PATH_TO/liversTr/{} --roi_subset liver
```

### Step 4: Inference
To predict for a certain image and clicks, e.g. 10, download the `model.py` and put it in `sw_fastedit_src` and then run this command:

```bash
python sw_fastedit_src/simplified_inference.py --input_dir YOUR_INPUT_PATH/venous_X_0000.nii.gz --json_dir YOUR_JSON_PATH/venous_X_clicks.json --output_dir YOUR_OUTPUT_PATH --n_clicks 10
```

For example on the demo images from the template:
```bash
python sw_fastedit_src/simplified_inference.py --input_dir ../docker/template_interactive/sample-test-data/images/venous_0_0000.nii.gz --json_dir ../docker/template_interactive/sample-test-data/clicks/venous_0_clicks.json --output_dir ./trials_test/ --n_clicks 10
```

### Step 5: Training
To predict for a certain image and clicks, e.g. 10, download the `model.py` and put it in `sw_fastedit_src` and then run this command:

```bash
python sw_fastedit_src/train.py -i PATH_TO/Task_2/ -o YOUR_OUTPUT_PATH --json_dir PATH_TO/Task_2/clicksTr/ -c ./cache -x 1.0 -a -e 50 --val_freq 200 --dont_check_output_dir --train
```


## Citation

We would like to acknowledge the contributions of the author behind the [LiTs challenge scoring](https://github.com/PatrickChrist/lits-challenge-scoring). Our work follows the guidelines and codebase established by their project
