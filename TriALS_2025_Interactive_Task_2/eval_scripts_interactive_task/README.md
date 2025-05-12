
# Evaluation Setup Guide for Task 2

We follow the [LiTs challenge scoring](https://github.com/PatrickChrist/lits-challenge-scoring) for the interactive metrics in [**Task 2: Interactive Click-based Portal-Venous Lesion Segmentation**](https://www.synapse.org/Synapse:syn65878273/wiki/631558#:~:text=Task%202%3A%20Interactive%20Click%2Dbased%20Portal%2DVenous%20Lesion%20Segmentation) 


### Step 1: Create Conda Environment

```bash
conda create -n interactive_lits_env python=3.9
conda activate interactive_lits_env
```

### Step 2: Install Requirements

Install the required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The following variables are to be pointed to using arguments via the command line:
```
test_csv_path = sys.argv[1] # READ IN`
predicted_masks_path = sys.argv[2] # READ IN`
groundtruth_masks_path = sys.argv[3] # READ IN`
metrics_csv_path = sys.argv[4] # OUTPUT`
```

Example command line code for the script:
```
python evaluate.py ../docker/template_interactive/sample-test-data/test.csv ../docker/template_interactive/sample-test-data/predictions/ ../docker/template_interactive/sample-test-data/ground_truth/ interactive_metrics/team-check/
```


Example of folders structure used by the organizers for challenge submissions:

```|__ binary/
    |
    |__ test.csv
    |
    |__ groundtruth/
    |   |__ venous_0.nii.gz
    |   |__ venous_1.nii.gz
    |   |__ ...
    |
    |__ inputs/
    |   |__ venous_0.nii.gz
    |   |__ venous_1.nii.gz
    |   |__ ...
    |
    |__ team-xyz/
    |   |__ metrics/
    |   |   
    |   |__ predictions/
    |       |__ venous_0
    |           |__ venous_0_0.nii.gz # 0 clicks
    |           |__ venous_0_1.nii.gz # 1st click iteration
                |   ...
    |           |__ venous_0_10.nii.gz # 10th click iteration
    |       |__ venous_1
    |           |__ venous_1_0.nii.gz # 0 clicks
    |           |__ venous_1_1.nii.gz # 1st click iteration
                |   ...
    |           |__ venous_1_10.nii.gz # 10th click iteration
    |       |__ ...
    |
    |__ team-medhacker/
        |__ ...```
```


## Acknowledgment

We would like to acknowledge the contributions of the author behind the [LiTs challenge scoring](https://github.com/PatrickChrist/lits-challenge-scoring). Our work follows the guidelines and codebase established by their project
