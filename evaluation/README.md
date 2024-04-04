
# Evaluation Setup Guide

We follow the [LiTs challenge scoring](https://github.com/PatrickChrist/lits-challenge-scoring) for the metrics.



### Step 1: Create Conda Environment

```bash
conda create -n lits_nnunet_env python=3.8
conda activate lits_nnunet_env

```

### Step 2: Install Requirements

Install the required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```


## Running the Script
To evalute one model (e.g., SAM-B), please run the following script
```bash
python LiTS_nnunet.py --model_folders "nnUNetTrainerV2_SAMed_b_r_4__nnUNetPlans__2d_p256" --base_prediction_path "path/to/nnUNet_results/Dataset003_Liver/" --truth_base_path "path/to//nnUNet_preprocessed/Dataset003_Liver/gt_segmentations" --output_base_path "/home/mkfmelbatel/lighter-net/results_LiTS"
```


## Acknowledgment

We would like to acknowledge the contributions of the author behind the [LiTs challenge scoring](https://github.com/PatrickChrist/lits-challenge-scoring). Our work follows the guidelines and codebase established by their project
