"""
Visualization script for v2 mini experiment results.
Generates: weather-branch heatmap, training curves, loss comparison.
"""
import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

EXP_DIR = '/home/hongsheng/v2/logs/exp_260706_231418_RL_3df_gate'
OUT_DIR = '/home/hongsheng/v2/logs/exp_260706_231418_RL_3df_gate/analysis_plots'
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════
# 1. Weather-Sensor Branch Weight Heatmap
# ═══════════════════════════════════════
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# Parse KITTI per-weather results
kitti_file = os.path.join(EXP_DIR, 'test_kitti', 'none', '0.3', 'complete_results.txt')
weather_ap = {}
if os.path.isfile(kitti_file):
    with open(kitti_file) as f:
        current_weather = None
        for line in f:
            if line.startswith('Conf thr:') and 'Condition:' in line:
                current_weather = line.split('Condition:')[1].strip()
            elif '3d   AP:' in line and current_weather:
                parts = line.split(':')
                if len(parts) >= 2:
                    ious = [float(x) for x in parts[1].strip().split(',')]
                    if len(ious) >= 3:
                        weather_ap[current_weather] = ious[2]  # IoU=0.3

# Parse branch weights from CSV
branch_csv = os.path.join(EXP_DIR, 'branch_monitor', 'branch_epoch_summary.csv')
epochs_data = []
if os.path.isfile(branch_csv):
    with open(branch_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs_data.append({
                'epoch': int(float(row['epoch'])),
                'lidar': float(row['branch_lidar']),
                'radar': float(row['branch_radar']),
                'camera': float(row['branch_camera']),
                'entropy': float(row['entropy_mean']),
            })

# ═══════════════ FIGURE 1: Branch Weight Evolution ═══════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

if epochs_data:
    epochs = [d['epoch'] for d in epochs_data]
    ax1.plot(epochs, [d['lidar'] for d in epochs_data], 'b-o', label='LiDAR', markersize=4)
    ax1.plot(epochs, [d['radar'] for d in epochs_data], 'r-s', label='Radar', markersize=4)
    ax1.plot(epochs, [d['camera'] for d in epochs_data], 'g-^', label='Camera', markersize=4)
    ax1.axhline(y=0.12, color='gray', linestyle='--', alpha=0.5, label='Floor (0.12)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Branch Weight')
    ax1.set_title('Branch Weight Evolution')
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [d['entropy'] for d in epochs_data], 'purple', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Branch Entropy')
    ax2.set_title('Branch Entropy (higher = more balanced)')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

fig.suptitle('v2 Branch Routing Dynamics (mini dataset)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_branch_weight_evolution.png'))
plt.close()
print('✓ Figure 1: branch weight evolution')

# ═══════════════ FIGURE 2: Weather-AP Bar Chart ═══════════════
if weather_ap:
    weather_order = ['all','normal','overcast','fog','rain','sleet','lightsnow','heavysnow']
    available = [w for w in weather_order if w in weather_ap]
    values = [weather_ap[w] for w in available]
    colors = ['gray','#3498db','#2ecc71','#f39c12','#e74c3c','#9b59b6','#1abc9c','#ecf0f1']
    bar_colors = [colors[weather_order.index(w) % len(colors)] for w in available]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(available, values, color=bar_colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('3D AP (%) @ IoU=0.3')
    ax.set_title('v2 Per-Weather 3D Detection Performance (mini dataset)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '02_weather_ap_bars.png'))
    plt.close()
    print('✓ Figure 2: weather AP bar chart')

# ═══════════════ FIGURE 3: AP Comparison v1+OT vs v2 ═══════════════
v1_ap = {
    'all': 59.97, 'normal': 52.92, 'overcast': 60.33, 'fog': 83.01,
    'rain': 49.11, 'lightsnow': 79.24, 'heavysnow': 82.45,
}
v2_ap = {k: v for k, v in weather_ap.items()}

# Only compare weathers both have
common_weathers = sorted(set(v1_ap.keys()) & set(v2_ap.keys()) - {'all'},
                         key=lambda w: v2_ap.get(w, 0))

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(common_weathers))
width = 0.35
bars1 = ax.bar(x - width/2, [v1_ap[w] for w in common_weathers], width, label='v1+OT', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, [v2_ap[w] for w in common_weathers], width, label='v2', color='#2ecc71', alpha=0.8)

for bar, val in zip(bars1, [v1_ap[w] for w in common_weathers]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', fontsize=8)
for bar, val in zip(bars2, [v2_ap[w] for w in common_weathers]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', fontsize=8)

# Highlight improvements
for i, w in enumerate(common_weathers):
    diff = v2_ap[w] - v1_ap[w]
    if diff > 1:
        ax.annotate(f'+{diff:.1f}', (x[i] + width/2, v2_ap[w] - 8),
                    ha='center', fontsize=10, fontweight='bold', color='green')
    elif diff < -1:
        ax.annotate(f'{diff:.1f}', (x[i] + width/2, v2_ap[w] - 8),
                    ha='center', fontsize=10, fontweight='bold', color='red')

ax.set_xticks(x)
ax.set_xticklabels(common_weathers)
ax.set_ylabel('3D AP (%) @ IoU=0.3')
ax.set_title('v1+OT vs v2: Per-Weather 3D AP Comparison (mini dataset)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_v1_vs_v2_ap_comparison.png'))
plt.close()
print('✓ Figure 3: v1 vs v2 AP comparison')

# ═══════════════ FIGURE 4: AP Radar Chart ═══════════════
weather_radar = ['normal', 'overcast', 'fog', 'rain', 'lightsnow', 'heavysnow']
available_radar = [w for w in weather_radar if w in v1_ap and w in v2_ap]
n = len(available_radar)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
v1_vals = [v1_ap[w] for w in available_radar] + [v1_ap[available_radar[0]]]
v2_vals = [v2_ap[w] for w in available_radar] + [v2_ap[available_radar[0]]]
ax.fill(angles, v1_vals, alpha=0.25, color='#e74c3c', label='v1+OT')
ax.fill(angles, v2_vals, alpha=0.25, color='#2ecc71', label='v2')
ax.plot(angles, v1_vals, 'o-', color='#e74c3c', linewidth=2)
ax.plot(angles, v2_vals, 'o-', color='#2ecc71', linewidth=2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(available_radar, fontsize=12)
ax.set_ylim(0, 95)
ax.set_title('Weather-Robustness Radar Chart', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_weather_radar.png'))
plt.close()
print('✓ Figure 4: weather radar chart')

print(f'\nAll plots saved to: {OUT_DIR}')
print(f'Files:')
for f in sorted(os.listdir(OUT_DIR)):
    print(f'  {f}')
