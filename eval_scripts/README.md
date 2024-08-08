
# Evaluation Setup Guide

We follow the [LiTs challenge scoring](https://github.com/PatrickChrist/lits-challenge-scoring) for the metrics.



### Step 1: Create Conda Environment

```bash
conda create -n lits_env python=3.8
conda activate lits_env
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
python evaluate.py ../docker/template/sample-test-data/test.csv  ../docker/template/sample-test-data/team-check/predictions/ ../docker/template/sample-test-data/ground_truth/ metrics/team-check/
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
    |       |__ venous_0.nii.gz
    |       |__ venous_1.nii.gz
    |       |__ ...
    |
    |__ team-medhacker/
        |__ ...```
```


## Acknowledgment

We would like to acknowledge the contributions of the author behind the [LiTs challenge scoring](https://github.com/PatrickChrist/lits-challenge-scoring). Our work follows the guidelines and codebase established by their project
