#!/bin/bash
# $1 is the csv file containing sample identifiers of test images
# $2 is the input path where test images are located
# $3 is the output path where predicted masks will be stored as images.
python main.py $1 $2 $3