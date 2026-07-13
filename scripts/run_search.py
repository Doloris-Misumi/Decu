#!/usr/bin/env python3
"""
V2 分支参数搜参脚本
基于 cfg_rl_3df_gate_mini_search.yml，修改搜参变量后依次运行。
用法: python3 scripts/run_search.py --gpu 3
"""
import os, sys, argparse, subprocess, yaml, shutil
from datetime import datetime

BASE_CONFIG = './configs/cfg_rl_3df_gate_mini_search.yml'
SEARCH_DIR = './logs/search_' + datetime.now().strftime('%m%d_%H%M')

# ============================================================
# 搜参空间定义（按优先级排列）
# ============================================================
SEARCH_RUNS = [
    # Phase 1: Entropy lambda sweep（优先，因为 mini 上 ent=0.05 导致过均匀）
    # (floor, ent_lambda, gate_floor, label)
    (0.15, 0.01, 0.20, "baseline"),
    (0.15, 0.005, 0.20, "ent_0005"),
    (0.15, 0.02, 0.20, "ent_002"),
    (0.15, 0.03, 0.20, "ent_003"),
    
    # Phase 2: Floor sweep
    # (0.10, 0.01, 0.20, "floor_010"),
    # (0.15, 0.01, 0.20, "floor_015"),
    # (0.20, 0.01, 0.20, "floor_020"),
]

def modify_and_run(floor, ent_lambda, gate_floor, label, gpu_id, search_dir):
    """修改配置文件并运行训练。"""
    temp_config = f'{search_dir}/cfg_{label}.yml'
    os.makedirs(search_dir, exist_ok=True)
    shutil.copy(BASE_CONFIG, temp_config)
    
    with open(temp_config, 'r') as f:
        cfg_text = f.read()
    
    # 精确替换搜参变量
    cfg_text = cfg_text.replace(
        'BRANCH_WEIGHT_FLOOR: 0.15  # [搜参范围: 0.10 ~ 0.20]',
        f'BRANCH_WEIGHT_FLOOR: {floor:.2f}  # [搜参范围: 0.10 ~ 0.20]')
    cfg_text = cfg_text.replace(
        'GATE_RESIDUAL_FLOOR: 0.20  # [搜参范围: 0.15 ~ 0.25]',
        f'GATE_RESIDUAL_FLOOR: {gate_floor:.2f}  # [搜参范围: 0.15 ~ 0.25]')
    cfg_text = cfg_text.replace(
        'LAMBDA: 0.01  # [搜参范围: 0.005 ~ 0.03]',
        f'LAMBDA: {ent_lambda:.3f}  # [搜参范围: 0.005 ~ 0.03]')
    
    with open(temp_config, 'w') as f:
        f.write(cfg_text)
    
    log_file = f'{search_dir}/{label}.log'
    print(f'\n{"="*60}')
    print(f'  Running: {label}')
    print(f'  floor={floor:.2f}  ent_lambda={ent_lambda:.2f}  gate_floor={gate_floor:.2f}')
    print(f'  Config: {temp_config}')
    print(f'  Log:    {log_file}')
    print(f'{"="*60}\n')
    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python main_train_0.py --config {temp_config}'
    with open(log_file, 'w') as log:
        proc = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    
    if proc.returncode != 0:
        print(f'  ⚠️  {label} failed (exit code {proc.returncode})')
    else:
        print(f'  ✅ {label} done')
    
    return proc.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true', help='只打印不执行')
    args = parser.parse_args()
    
    print(f'搜参目录: {SEARCH_DIR}')
    print(f'GPU: {args.gpu}')
    print(f'共 {len(SEARCH_RUNS)} 组参数\n')
    
    for i, (floor, ent, gate, label) in enumerate(SEARCH_RUNS):
        print(f'  [{i+1}/{len(SEARCH_RUNS)}] {label}: floor={floor:.2f} ent={ent:.2f} gate={gate:.2f}')
    
    if args.dry_run:
        print('\n[Dry run — 不执行训练]')
        return
    
    for floor, ent, gate, label in SEARCH_RUNS:
        ret = modify_and_run(floor, ent, gate, label, args.gpu, SEARCH_DIR)
        if ret != 0:
            print(f'\n停止：{label} 失败')
            break
    
    print(f'\n搜参完成！结果: {SEARCH_DIR}/')
    print(f'查看: python3 scripts/parse_search_results.py {SEARCH_DIR}')

if __name__ == '__main__':
    main()
