#!/bin/bash
# V2 超参搜索脚本 — Phase 1: Floor sweep
# 在 mini split 上快速测试，每轮 5 epochs (~15 min)
# 用法: bash scripts/search_phase1.sh [GPU_ID]

set -e
GPU_ID=${1:-3}
BASE_CONFIG="./configs/cfg_rl_3df_gate_mini_noclip.yml"
SEARCH_DIR="./logs/search_phase1_floor"
mkdir -p "$SEARCH_DIR"

echo "============================================"
echo " Phase 1: BRANCH_WEIGHT_FLOOR sweep"
echo " GPU: $GPU_ID | 5 epochs each | mini split"
echo "============================================"

# 搜参组合: floor ent_lambda gate_floor
declare -a RUNS=(
    "floor_015|0.15|0.05|0.25"
    "floor_020|0.20|0.05|0.25"
    "floor_025|0.25|0.05|0.25"
)

for run_info in "${RUNS[@]}"; do
    IFS='|' read -r name floor ent_lambda gate_floor <<< "$run_info"
    echo ""
    echo ">>> Running: $name (floor=$floor, ent=$ent_lambda, gate=$gate_floor)"
    
    # 创建临时配置
    TMP_CONFIG="${SEARCH_DIR}/cfg_${name}.yml"
    cp "$BASE_CONFIG" "$TMP_CONFIG"
    
    # 修改参数
    sed -i "s/BRANCH_WEIGHT_FLOOR: .*/BRANCH_WEIGHT_FLOOR: $floor/" "$TMP_CONFIG"
    sed -i "s/GATE_RESIDUAL_FLOOR: .*/GATE_RESIDUAL_FLOOR: $gate_floor/" "$TMP_CONFIG"
    sed -i "s/LAMBDA: .*  #.*/LAMBDA: $ent_lambda  # search/" "$TMP_CONFIG"
    sed -i "s/MAX_EPOCH: 20/MAX_EPOCH: 5/" "$TMP_CONFIG"
    
    # 启用 CLIP（搜参阶段先保持 CLIP 开启）
    sed -i "s/USE_CLIP: False/USE_CLIP: True/" "$TMP_CONFIG"
    sed -i "s/USE_PROMPT_TOKEN: False/USE_PROMPT_TOKEN: True/" "$TMP_CONFIG"
    
    # 每轮验证
    sed -i "s/VAL_PHASE_SCHEDULE: .*/VAL_PHASE_SCHEDULE: [[5, 1]]/" "$TMP_CONFIG"
    
    # 运行
    CUDA_VISIBLE_DEVICES=$GPU_ID python main_train_0.py --config "$TMP_CONFIG" 2>&1 | tee "${SEARCH_DIR}/${name}.log"
    
    echo ">>> Done: $name"
done

echo ""
echo "Phase 1 完成！结果保存在 $SEARCH_DIR/"
echo "查看分支权重: grep 'Branch Weight Monitor' ${SEARCH_DIR}/*.log"
