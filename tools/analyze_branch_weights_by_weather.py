import argparse
import copy
import csv
import json
import os
import sys
import traceback
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import datasets
from models.skeletons import build_skeleton
from utils.util_config import cfg as global_cfg
from utils.util_config import cfg_from_yaml_file


def load_cfg(path_cfg):
    cfg = copy.deepcopy(global_cfg)
    cfg = cfg_from_yaml_file(path_cfg, cfg)
    cfg.GENERAL.LOGGING.IS_LOGGING = False
    cfg.VAL.IS_VALIDATE = False
    return cfg


def build_loader(cfg, split='test', batch_size=4, num_workers=4):
    dataset = datasets.__all__[cfg.DATASET.NAME](cfg=cfg, split=split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataset, loader


def safe_weather(meta):
    if not isinstance(meta, dict):
        return 'unknown'
    desc = meta.get('desc', {})
    if not isinstance(desc, dict):
        return 'unknown'
    weather = str(desc.get('climate', 'unknown')).strip().lower()
    return weather if weather else 'unknown'


def tensor_to_numpy(tensor):
    if not torch.is_tensor(tensor):
        return None
    return tensor.detach().float().cpu().numpy()


def summarize_rows(rows):
    if len(rows) == 0:
        return {
            'count': 0,
            'mean': [],
            'std': [],
        }
    arr = np.asarray(rows, dtype=np.float32)
    return {
        'count': int(arr.shape[0]),
        'mean': arr.mean(axis=0).astype(np.float32).tolist(),
        'std': arr.std(axis=0).astype(np.float32).tolist(),
    }


def build_payload(epoch, split, checkpoint, branch_names, dataset, per_weather_real, per_weather_mix, per_weather_pref, global_real, global_mix, global_pref, skipped_batches):
    weather_keys = sorted(set(per_weather_real.keys()) | set(getattr(dataset, 'weather_list', [])))
    payload = {
        'epoch': int(epoch),
        'split': split,
        'checkpoint': checkpoint,
        'branch_names': branch_names,
        'global': {
            'real': summarize_rows(global_real),
            'mix': summarize_rows(global_mix),
            'pref': summarize_rows(global_pref),
        },
        'weather': {},
        'skipped_batches': skipped_batches,
    }
    for weather in weather_keys:
        payload['weather'][weather] = {
            'real': summarize_rows(per_weather_real[weather]),
            'mix': summarize_rows(per_weather_mix[weather]),
            'pref': summarize_rows(per_weather_pref[weather]),
        }
    return payload


def save_payload(output_dir, payload, epoch, split, is_partial=False):
    os.makedirs(output_dir, exist_ok=True)
    suffix = '_partial' if is_partial else ''
    json_path = os.path.join(output_dir, f'epoch_{epoch:03d}_{split}_branch_weather{suffix}.json')
    csv_path = os.path.join(output_dir, f'epoch_{epoch:03d}_{split}_branch_weather{suffix}.csv')

    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    branch_names = payload.get('branch_names', ['lidar', 'radar', 'camera'])
    fieldnames = ['weather', 'count']
    for prefix in ['real', 'mix', 'pref']:
        for branch_name in branch_names:
            fieldnames.append(f'{prefix}_{branch_name}')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for weather, weather_payload in payload.get('weather', {}).items():
            row = {
                'weather': weather,
                'count': int(weather_payload['real']['count']),
            }
            for prefix in ['real', 'mix', 'pref']:
                mean_vals = weather_payload[prefix]['mean']
                for idx, branch_name in enumerate(branch_names):
                    row[f'{prefix}_{branch_name}'] = float(mean_vals[idx]) if idx < len(mean_vals) else ''
            writer.writerow(row)

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--epoch', type=int, default=6)
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--save-every', type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is required for this analysis script because the current project still contains '
            'legacy .cuda() calls in several model modules.'
        )

    cfg = load_cfg(args.config)
    print(f'[1/5] Config loaded: {args.config}')
    dataset, loader = build_loader(
        cfg,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f'[2/5] Dataset ready: split={args.split}, num_samples={len(dataset)}, batch_size={args.batch_size}')

    print('[3/5] Building network and loading text encoder...')
    network = build_skeleton(cfg).cuda()
    print(f'[4/5] Loading checkpoint: {args.checkpoint}')
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    network.load_state_dict(state_dict, strict=False)
    network.eval()
    print('[5/5] Running inference and aggregating branch weights by weather...')

    per_weather_real = defaultdict(list)
    per_weather_mix = defaultdict(list)
    per_weather_pref = defaultdict(list)
    global_real = []
    global_mix = []
    global_pref = []
    skipped_batches = []
    branch_names = ['lidar', 'radar', 'camera']

    if args.output_dir is None:
        exp_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        output_dir = os.path.join(exp_dir, 'branch_weather_analysis')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    save_every = max(1, int(args.save_every))

    try:
        with torch.no_grad():
            for batch_idx, dict_datum in enumerate(tqdm(loader, desc='Analyzing', dynamic_ncols=True)):
                if dict_datum is None:
                    skipped_batches.append({
                        'batch_idx': int(batch_idx),
                        'reason': 'dict_datum_is_none',
                    })
                    continue

                try:
                    dict_net = network(dict_datum)
                except Exception as e:
                    skipped_batches.append({
                        'batch_idx': int(batch_idx),
                        'reason': str(e),
                        'traceback_tail': traceback.format_exc().splitlines()[-5:],
                    })
                    continue

                real = tensor_to_numpy(dict_net.get('branch_real_weights', None))
                mix = tensor_to_numpy(dict_net.get('token_guided_mix_weights', None))
                pref = tensor_to_numpy(dict_net.get('branch_preference', None))
                meta_list = dict_datum.get('meta', [])
                if isinstance(meta_list, tuple):
                    meta_list = list(meta_list)

                if real is None:
                    skipped_batches.append({
                        'batch_idx': int(batch_idx),
                        'reason': 'missing_branch_real_weights',
                    })
                    continue

                batch_size = real.shape[0]
                for idx in range(batch_size):
                    weather = safe_weather(meta_list[idx] if idx < len(meta_list) else {})
                    per_weather_real[weather].append(real[idx])
                    global_real.append(real[idx])

                    if mix is not None and idx < mix.shape[0]:
                        per_weather_mix[weather].append(mix[idx])
                        global_mix.append(mix[idx])
                    if pref is not None and idx < pref.shape[0]:
                        per_weather_pref[weather].append(pref[idx])
                        global_pref.append(pref[idx])

                if ((batch_idx + 1) % save_every) == 0:
                    partial_payload = build_payload(
                        args.epoch, args.split, args.checkpoint, branch_names, dataset,
                        per_weather_real, per_weather_mix, per_weather_pref,
                        global_real, global_mix, global_pref, skipped_batches,
                    )
                    save_payload(output_dir, partial_payload, args.epoch, args.split, is_partial=True)
    finally:
        payload = build_payload(
            args.epoch, args.split, args.checkpoint, branch_names, dataset,
            per_weather_real, per_weather_mix, per_weather_pref,
            global_real, global_mix, global_pref, skipped_batches,
        )
        save_payload(output_dir, payload, args.epoch, args.split, is_partial=True)

    json_path, csv_path = save_payload(output_dir, payload, args.epoch, args.split, is_partial=False)

    print(f'Analysis saved to: {json_path}')
    print(f'Analysis saved to: {csv_path}')
    if skipped_batches:
        print(f'Skipped batches: {len(skipped_batches)}')
    print('Global means:')
    for prefix in ['real', 'mix', 'pref']:
        vals = payload['global'][prefix]['mean']
        if len(vals) == 0:
            continue
        print(
            f"  {prefix}: "
            + ', '.join(f'{name}={vals[idx]:.4f}' for idx, name in enumerate(branch_names))
        )
    print('Per-weather real means:')
    for weather, weather_payload in payload.get('weather', {}).items():
        vals = weather_payload['real']['mean']
        count = weather_payload['real']['count']
        if len(vals) == 0:
            continue
        print(
            f"  {weather:10s} count={count:4d} "
            + ', '.join(f'{name}={vals[idx]:.4f}' for idx, name in enumerate(branch_names))
        )


if __name__ == '__main__':
    main()
