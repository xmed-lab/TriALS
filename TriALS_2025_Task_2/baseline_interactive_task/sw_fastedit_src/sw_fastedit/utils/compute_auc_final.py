import os
import argparse
import json
from collections import defaultdict
from scipy import integrate
import numpy as np

def load_json(path):
    with open(path) as f:
        return json.load(f)

def group_jsons_by_prefix(json_dir):
    grouped = defaultdict(list)

    for fname in os.listdir(json_dir):
        if not fname.endswith('.json'):
            continue
        prefix = '_'.join(fname.split('_')[:-1])  # this grabs up to venous_0_0000
        full_path = os.path.join(json_dir, fname)
        grouped[prefix].append(full_path)

    return grouped

def aggregate_group_metrics(group_paths):
    all_scores = [load_json(p) for p in sorted(group_paths)]

    # keys are assumed the same across JSONs
    keys = all_scores[0].keys()
    aggregated = {}

    for k in keys:
        if k not in ['dice', 'assd', 'msd']:
            continue

        values = np.array([score[k][0] for score in all_scores])
        auc = integrate.cumulative_trapezoid(values, np.arange(11))[-1]
        final = values[-1]
        aggregated[k] = {
            "AUC": auc,
            "Final": final
        }

    return aggregated

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute interactive scores (AUC / Final).')
    parser.add_argument('--val_metrics', type=str, required=True, help='Folder with non-interactive val-metrics')
    args = parser.parse_args()
    grouped_jsons = group_jsons_by_prefix(args.val_metrics)
    output_path = args.val_metrics.replace('val_metrics', 'interactive_metrics')
    os.makedirs(output_path, exist_ok=True)
    for prefix, files in grouped_jsons.items():
        print(f'\n📁 Group: {prefix} ({len(files)} files)')
        agg = aggregate_group_metrics(files)
        output_json = os.path.join(output_path, f'{prefix}.json')
        with open(output_json, 'w') as f:
            json.dump(agg, f, indent=2)
        print(f"[✓] Saved: {output_json}")
