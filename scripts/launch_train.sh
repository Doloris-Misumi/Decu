#!/bin/bash
# V2 训练启动器（tmux 会话保持，SSH 断开不死）
# 用法: bash scripts/launch_train.sh [GPU_ID] [CONFIG]
#       bash scripts/launch_train.sh 2 ./configs/cfg_rl_3df_gate.yml
#       bash scripts/launch_train.sh 3 ./configs/cfg_rl_3df_gate_mini_search.yml

GPU_ID=${1:-2}
CONFIG=${2:-./configs/cfg_rl_3df_gate.yml}
SESSION="v2_train_gpu${GPU_ID}"

# 先杀掉旧会话
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

echo "启动训练会话: $SESSION"
echo "GPU: $GPU_ID | Config: $CONFIG"

# tmux 里直接用全路径 python，避免 conda activate 失败
PYTHON=/home/hongsheng/miniconda3/envs/rl_3dod/bin/python
tmux new-session -d -s "$SESSION" \
    "cd /home/hongsheng/v2 && export HF_HUB_OFFLINE=1 && CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON main_train_0.py --config $CONFIG 2>&1 | tee ./logs/train_gpu${GPU_ID}_\$(date +%m%d_%H%M).log"

sleep 2

# 验证是否启动成功
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "✅ 训练已在后台启动"
else
    echo "❌ 会话未启动，检查环境"
    exit 1
fi

echo ""
echo "常用操作:"
echo "  tmux attach -t $SESSION"
echo "  脱离: Ctrl+B 然后 D"
echo "  杀掉: tmux kill-session -t $SESSION"
