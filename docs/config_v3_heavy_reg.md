# V2 配置存档 — v3 重正则化版本
日期: 2026-07-07
实验: exp_260707_182558 (全量 GPU2 运行中)

## 分支控制
- FLOOR: 0.20
- GATE: 0.25
- ENT_LAMBDA: 0.05
- TARGET_RATIO: 0.85

## 现象
- Mini: 熵→0.997 过于均匀，AP 从 24.94 跌到 6.06
- Full: 熵→0.894 雷达垄断，LiDAR/Camera 贴地板

## 结论
ent_lambda=0.05 在 mini 和 full 上走向两极，不可用
