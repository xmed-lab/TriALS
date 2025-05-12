from __future__ import annotations

from enum import IntEnum

LABELS_KEY = "label_names"


class ClickGenerationStrategy(IntEnum):
    # Sample a click randomly based on the label, so no correction based on the prediction
    GLOBAL_NON_CORRECTIVE = 1
    # Sample a click based on the discrepancy between label and predition
    # Thus generate corrective clicks where the networks predicts incorrectly so far
    GLOBAL_CORRECTIVE = 2
    # Subdivide volume into patches of size train_crop_size, calculate the dice score for each, then sample click on the worst one
    PATCH_BASED_CORRECTIVE = 3
    # At each iteration sample from the probability and don't add a click if it yields False
    DEEPGROW_GLOBAL_CORRECTIVE = 4


class StoppingCriterion(IntEnum):
    # Sample max_train_interactions amount of clicks (can be done in the first iteration if non-corrective)
    MAX_ITER = 1
    # Sample clicks iteratively. At each step sample p~(0,1). If p > x continue sampling
    MAX_ITER_AND_PROBABILITY = 2
    # Sample clicks iteratively. Stop when dice good enough (e.g. 0.9) or when max_train_interactions amount of clicks
    MAX_ITER_AND_DICE = 3
    # Sample clicks iteratively. At each step: Stop if max_train_interactions is reached. Otherwise sample p~(0,1).
    # If p > dice continue sampling, then check if dice is good enough. If so no more clicks are required.
    MAX_ITER_PROBABILITY_AND_DICE = 4
    # Stopping as previously implemented with Deepgrow
    DEEPGROW_PROBABILITY = 5
