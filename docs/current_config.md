# V2 当前主要参数配置 (v4 轻正则)
日期: 2026-07-07

## 分支控制
- BRANCH_WEIGHT_FLOOR: 0.15 (轻兜底，防死亡不绑架)
- GATE_RESIDUAL_FLOOR: 0.20 (适度保留)
- BRANCH_ENTROPY.LAMBDA: 0.01 (轻推，不强迫均匀)
- BRANCH_ENTROPY.TARGET_RATIO: 0.75 (允许不均匀路由)

## 条件编码器
- USE_CLIP: True
- USE_PROMPT_TOKEN: True
- OT_ENABLED: True

## 优化器
- LR: 0.0005
- BATCH: 4
- EPOCHS: 20
- SCHEDULER: CosineAnnealingLR

## 历史版本
- v3 (heavy_reg): floor=0.20, gate=0.25, ent=0.05, ratio=0.85
  → mini过均匀(熵0.997), full崩塌(熵0.894)
- v2 (original): floor=0.12, gate=0.15, ent=0.005, ratio=0.78
  → 全量epoch3崩塌到雷达100%
