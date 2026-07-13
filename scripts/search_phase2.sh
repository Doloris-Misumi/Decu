#!/bin/bash
# V2 超参搜索脚本 — Phase 2: Entropy sweep
# 基于 Phase 1 最佳 floor，搜索最佳 entropy lambda
# 用法: bash scripts/search_phase2.sh [GPU_ID] [BEST_FLOOR]

GPU_ID=${1:-3}
BEST_FLOOR=${2:-0.20}
BASE_CONFIG="./configs/cfg_rl_3df_gate_mini_noclip.yml"
SEARCH_DIR="./logs/search_phase2_entropy"
mkdir -p "$SEARCH_DIR"

echo "============================================"
echo " Phase 2: BRANCH_ENTROPY.LAMBDA sweep"
echo " GPU: $GPU_ID | best_floor=$BEST_FLOOR"
echo "============================================"

declare -a RUNS=(
    "ent_002|${BEST_FLOOR}|0.02|0.25"
    "ent_005|${BEST_FLOOR}|0.05|0.25"
    "ent_010|${BEST_FLOOR}|0.10|0.25"
)

for run_info in "${RUNS[@]}"; do
    IFS='|' read -r name floor ent_lambda gate_floor <<< "$run_info"
    echo ">>> Running: $name (floor=$floor, ent=$ent_lambda)"
    
    TMP_CONFIG="${SEARCH_DIR}/cfg_${name}.yml"
    cp "$BASE_CONFIG" "$TMP_CONFIG"
    
    sed -i "s/BRANCH_WEIGHT_FLOOR: .*/BRANCH_WEIGHT_FLOOR: $floor/" "$TMP_CONFIG"
    sed -i "s/GATE_RESIDUAL_FLOOR: .*/GATE_RESIDUAL_FLOOR: $gate_floor/" "$TMP_CONFIG"
    sed -i "s/LAMBDA: .*  #.*/LAMBDA: $ent_lambda  # search/" "$TMP_CONFIG"
    sed -i "s/MAX_EPOCH: 20/MAX_EPOCH: 5/" "$TMP_CONFIG"
    sed -i "s/USE_CLIP: False/USE_CLIP: True/" "$TMP_CONFIG"
    sed -i "s/USE_PROMPT_TOKEN: False/USE_PROMPT_TOKEN: True/" "$TMP_CONFIG"
    sed -i "s/VAL_PHASE_SCHEDULE: .*/VAL_PHASE_SCHEDULE: [[5, 1]]/" "$TMP_CONFIG"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python main_train_0.py --config "$TMP_CONFIG" 2>&1 | tee "${SEARCH_DIR}/${name}.log"
    echo ">>> Done: $name"
done

echo "Phase 2 完成！"
