"""
Complete visualization suite for v2 mini experiment.
Generates all figures needed for the paper.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import os, json, csv, glob
from collections import defaultdict

EXP = '/home/hongsheng/v2/logs/exp_260706_231418_RL_3df_gate'
OUT = os.path.join(EXP, 'paper_figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'figure.dpi': 200})

# ─── Color schemes ───
C_LIDAR  = '#3498db'  # blue
C_RADAR  = '#e67e22'  # orange
C_CAMERA = '#2ecc71'  # green
CMAP_BEV = plt.cm.viridis

# ═══════════════════════════════════════════════════
# FIGURE 1: Shared vs Residual Decomposition
# ═══════════════════════════════════════════════════
def load_sample(epoch=9, sample=0):
    d = os.path.join(EXP, 'feature_vis', f'epoch_{epoch:03d}', f'sample_{sample:02d}')
    data = np.load(os.path.join(d, 'feature_maps.npz'))
    with open(os.path.join(d, 'meta.json')) as f:
        meta = json.load(f)
    return data, meta

data, meta = load_sample(9, 0)

# Magnitude maps (abs mean across channels, 32x180)
shared_L = data['shared_lidar']    # LiDAR's contribution to shared
shared_R = data['shared_radar']    # Radar's contribution to shared
shared_C = data['shared_camera']   # Camera's contribution to shared
resid_L  = data['residual_lidar']
resid_R  = data['residual_radar']
resid_C  = data['residual_camera']

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle(f'Shared vs Residual BEV Decomposition\nEpoch 9, {meta.get("climate","?")}, {meta.get("prompt","?")}',
             fontsize=14, fontweight='bold')

plot_pairs = [
    (axes[0,0], shared_L, 'Shared - LiDAR', C_LIDAR),
    (axes[0,1], shared_R, 'Shared - Radar', C_RADAR),
    (axes[0,2], shared_C, 'Shared - Camera', C_CAMERA),
    (axes[1,0], resid_L, 'Residual - LiDAR', C_LIDAR),
    (axes[1,1], resid_R, 'Residual - Radar', C_RADAR),
    (axes[1,2], resid_C, 'Residual - Camera', C_CAMERA),
]

for ax, arr, title, color in plot_pairs:
    im = ax.imshow(arr, cmap=CMAP_BEV, aspect='auto', origin='lower')
    ax.set_title(title, color=color, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)

# Bottom row: final comparison
axes[2,0].imshow(data['calibrated_lidar'], cmap=CMAP_BEV, aspect='auto', origin='lower')
axes[2,0].set_title('Calibrated LiDAR', fontweight='bold')
axes[2,0].set_xticks([]); axes[2,0].set_yticks([])

axes[2,1].imshow(data['calibrated_radar'], cmap=CMAP_BEV, aspect='auto', origin='lower')
axes[2,1].set_title('Calibrated Radar', fontweight='bold')
axes[2,1].set_xticks([]); axes[2,1].set_yticks([])

axes[2,2].imshow(data['final_bev'], cmap='hot', aspect='auto', origin='lower')
axes[2,2].set_title('Final Fused BEV', fontweight='bold')
axes[2,2].set_xticks([]); axes[2,2].set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1_shared_residual.png'), bbox_inches='tight')
plt.close()
print('✓ Fig 1: Shared/Residual decomposition')

# ═══════════════════════════════════════════════════
# FIGURE 2: BEV Spatial Dominance (where each sensor rules)
# ═══════════════════════════════════════════════════
weighted_L = data['weighted_lidar']
weighted_R = data['weighted_radar']
weighted_C = data['weighted_camera']

# Stack and find dominant sensor at each BEV position
stacked = np.stack([weighted_L, weighted_R, weighted_C], axis=0)  # [3, 32, 180]
dominance = np.argmax(stacked, axis=0)  # [32, 180], 0=LiDAR, 1=Radar, 2=Camera
total_energy = np.sum(stacked, axis=0) + 1e-9
energy_L = stacked[0] / total_energy
energy_R = stacked[1] / total_energy
energy_C = stacked[2] / total_energy

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('Spatial Branch Dominance Map\n(which sensor contributes most at each BEV position)',
             fontsize=13, fontweight='bold')

# Dominance map
cmap_dom = plt.cm.colors.ListedColormap([C_LIDAR, C_RADAR, C_CAMERA])
im0 = axes[0].imshow(dominance, cmap=cmap_dom, aspect='auto', origin='lower', vmin=0, vmax=2)
axes[0].set_title('Dominant Sensor\n(LiDAR/R/C)', fontweight='bold')
axes[0].set_xticks([]); axes[0].set_yticks([])
legend_elements = [mpatches.Patch(color=C_LIDAR, label='LiDAR'),
                   mpatches.Patch(color=C_RADAR, label='Radar'),
                   mpatches.Patch(color=C_CAMERA, label='Camera')]
axes[0].legend(handles=legend_elements, loc='lower right', fontsize=8)

# Per-sensor contribution maps
for ax, arr, name, color in [
    (axes[1], energy_L, 'LiDAR Contribution', C_LIDAR),
    (axes[2], energy_R, 'Radar Contribution', C_RADAR),
    (axes[3], energy_C, 'Camera Contribution', C_CAMERA),
]:
    im = ax.imshow(arr, cmap='hot', aspect='auto', origin='lower', vmin=0, vmax=1)
    ax.set_title(name, color=color, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2_bev_dominance.png'), bbox_inches='tight')
plt.close()
print('✓ Fig 2: BEV spatial dominance')

# ═══════════════════════════════════════════════════
# FIGURE 3: Weather-Sensor Branch Weight Heatmap
# ═══════════════════════════════════════════════════
# Parse from KITTI results and branch CSV
kitti_file = os.path.join(EXP, 'test_kitti', 'none', '0.3', 'complete_results.txt')
weather_ap = {}
if os.path.isfile(kitti_file):
    current = None
    with open(kitti_file) as f:
        for line in f:
            if 'Condition:' in line:
                current = line.split('Condition:')[1].strip()
            elif '3d  :' in line and current:
                parts = line.split(':')[1].strip().split()
                if len(parts) >= 3:
                    weather_ap[current] = float(parts[2])

# Get branch weights from last epoch
branch_csv = os.path.join(EXP, 'branch_monitor', 'branch_epoch_summary.csv')
epoch_19_data = {}
if os.path.isfile(branch_csv):
    with open(branch_csv) as f:
        for row in csv.DictReader(f):
            if int(float(row['epoch'])) == 19:
                epoch_19_data = {
                    'lidar': float(row['branch_lidar']),
                    'radar': float(row['branch_radar']),
                    'camera': float(row['branch_camera']),
                }
                break

# Weather AP bar chart (Figure 3a)
weather_order = ['normal','overcast','fog','rain','lightsnow','heavysnow','all']
available = [w for w in weather_order if w in weather_ap]
colors = [C_LIDAR, C_RADAR, C_CAMERA, '#e74c3c', '#1abc9c', '#9b59b6', '#34495e']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Weather-Conditioned Detection Performance', fontsize=14, fontweight='bold')

vals = [weather_ap[w] for w in available]
bar_colors = [colors[available.index(w) % len(colors)] for w in available]
bars = ax1.bar(available, vals, color=bar_colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, vals):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f'{val:.1f}%',
             ha='center', fontsize=10, fontweight='bold')
ax1.set_ylabel('3D AP (%) @ IoU=0.3')
ax1.set_title('Per-Weather Detection Performance')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 95)

# Branch weight gauge (Figure 3b)
if epoch_19_data:
    sensors = ['LiDAR', 'Radar', 'Camera']
    weights = [epoch_19_data['lidar'], epoch_19_data['radar'], epoch_19_data['camera']]
    gauge_colors = [C_LIDAR, C_RADAR, C_CAMERA]
    bars = ax2.barh(sensors, weights, color=gauge_colors, edgecolor='black')
    for bar, w in zip(bars, weights):
        ax2.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2, f'{w*100:.1f}%',
                 va='center', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_title('Final Branch Weights (epoch 19)')
    ax2.set_xlabel('Weight')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3_weather_ap_and_branch.png'), bbox_inches='tight')
plt.close()
print('✓ Fig 3: Weather AP + branch gauge')

# ═══════════════════════════════════════════════════
# FIGURE 4: Per-Distance AP
# ═══════════════════════════════════════════════════
def compute_per_distance_ap(pred_dir, gt_dir, dist_bins=[0,20,40,60,80]):
    """Compute recall per longitudinal distance bin."""
    pred_files = glob.glob(os.path.join(pred_dir, '*.txt'))
    gt_files = glob.glob(os.path.join(gt_dir, '*.txt'))
    common = set(os.path.basename(f) for f in pred_files) & set(os.path.basename(f) for f in gt_files)
    
    bin_gt = [0] * (len(dist_bins)-1)
    bin_matched = [0] * (len(dist_bins)-1)
    
    for fname in sorted(common):
        gts = []
        with open(os.path.join(gt_dir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 14:
                    # KITTI: x(longitudinal)=parts[13], z(height)=parts[11]
                    forward_dist = float(parts[13])
                    gts.append(forward_dist)
        preds = []
        with open(os.path.join(pred_dir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 16:
                    forward_dist = float(parts[13])
                    score = float(parts[15])
                    if score > 0.3:
                        preds.append(forward_dist)
        
        for gt_dist in gts:
            for bi in range(len(dist_bins)-1):
                if dist_bins[bi] <= gt_dist < dist_bins[bi+1]:
                    bin_gt[bi] += 1
                    match = any(abs(gt_dist - pd) < 3.0 for pd in preds)
                    if match:
                        bin_matched[bi] += 1
                    break
    
    bin_ap = []
    for i in range(len(dist_bins)-1):
        if bin_gt[i] > 0:
            bin_ap.append(bin_matched[i] / bin_gt[i] * 100)
        else:
            bin_ap.append(0)
    return bin_ap, bin_gt

# Try to compute for v2 epoch 19
epoch19_kitti = os.path.join(EXP, 'test_kitti', 'epoch_19_subset', '0.3')
if os.path.isdir(epoch19_kitti):
    bin_ap, bin_gt = compute_per_distance_ap(
        os.path.join(epoch19_kitti, 'pred'),
        os.path.join(epoch19_kitti, 'gt'),
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    dist_labels = ['0-20m','20-40m','40-60m','60-80m']
    x = np.arange(len(dist_labels))
    bars = ax.bar(x, bin_ap, color=[C_LIDAR, C_RADAR, C_CAMERA, '#e74c3c'], edgecolor='black')
    for bar, val, gt in zip(bars, bin_ap, bin_gt):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.1f}%\n(GT:{gt})',
                ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(dist_labels)
    ax.set_ylabel('Approx. AP (%)')
    ax.set_title('Per-Distance Detection Performance (v2, epoch 19)')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig4_per_distance_ap.png'), bbox_inches='tight')
    plt.close()
    print('✓ Fig 4: Per-distance AP')
else:
    print('⚠ Fig 4 skipped: no epoch_19 KITTI data')

# ═══════════════════════════════════════════════════
# FIGURE 5: Combined Training Dynamics Dashboard
# ═══════════════════════════════════════════════════
# Read branch epoch summary
epochs, l_w, r_w, c_w, ent = [], [], [], [], []
if os.path.isfile(branch_csv):
    with open(branch_csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(float(row['epoch'])))
            l_w.append(float(row['branch_lidar']))
            r_w.append(float(row['branch_radar']))
            c_w.append(float(row['branch_camera']))
            ent.append(float(row['entropy_mean']))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('v2 Training Dynamics Dashboard', fontsize=15, fontweight='bold')

# Plot 1: Branch weight curves
ax = axes[0,0]
if epochs:
    ax.plot(epochs, l_w, 'o-', color=C_LIDAR, label='LiDAR', markersize=4, linewidth=2)
    ax.plot(epochs, r_w, 's-', color=C_RADAR, label='Radar', markersize=4, linewidth=2)
    ax.plot(epochs, c_w, '^-', color=C_CAMERA, label='Camera', markersize=4, linewidth=2)
    ax.axhline(y=0.12, color='gray', linestyle='--', alpha=0.5, label='Floor (0.12)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Branch Weight')
ax.set_title('Branch Weight Trajectory'); ax.legend()
ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)

# Plot 2: Entropy
axes[0,1].plot(epochs, ent, 'purple', linewidth=2.5)
axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Entropy')
axes[0,1].set_title('Branch Entropy (balance indicator)')
axes[0,1].set_ylim(0, 1.05); axes[0,1].grid(True, alpha=0.3)

# Plot 3: v1 vs v2 AP comparison
v1_ap = {'normal':52.92,'overcast':60.33,'fog':83.01,'rain':49.11,'lightsnow':79.24,'heavysnow':82.45}
v2_ap = {k:v for k,v in weather_ap.items()}
compare_w = sorted(set(v1_ap)&set(v2_ap), key=lambda w: v2_ap.get(w,0))
x3 = np.arange(len(compare_w)); w = 0.35
axes[1,0].bar(x3-w/2, [v1_ap[w] for w in compare_w], w, label='v1+OT', color='#e74c3c', alpha=0.85)
axes[1,0].bar(x3+w/2, [v2_ap[w] for w in compare_w], w, label='v2', color=C_CAMERA, alpha=0.85)
for i, wname in enumerate(compare_w):
    diff = v2_ap[wname] - v1_ap[wname]
    c = 'green' if diff>0 else 'red'
    axes[1,0].annotate(f'{diff:+.1f}', (x3[i], max(v1_ap[wname],v2_ap[wname])+3),
                       ha='center', fontsize=9, fontweight='bold', color=c)
axes[1,0].set_xticks(x3); axes[1,0].set_xticklabels(compare_w)
axes[1,0].set_ylabel('3D AP (%)'); axes[1,0].legend()
axes[1,0].set_title('v1+OT vs v2: Per-Weather 3D AP'); axes[1,0].grid(True, alpha=0.3, axis='y')

# Plot 4: Radar chart (only if we have data)
if len(compare_w) > 0:
    n = len(compare_w)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist() + [0]
    ax4 = fig.add_subplot(2,2,4, polar=True)
    v1v = [v1_ap[w] for w in compare_w] + [v1_ap[compare_w[0]]]
    v2v = [v2_ap[w] for w in compare_w] + [v2_ap[compare_w[0]]]
    ax4.fill(angles, v1v, alpha=0.2, color='#e74c3c')
    ax4.fill(angles, v2v, alpha=0.2, color=C_CAMERA)
    ax4.plot(angles, v1v, 'o-', color='#e74c3c', linewidth=2, label='v1+OT')
    ax4.plot(angles, v2v, 'o-', color=C_CAMERA, linewidth=2, label='v2')
    ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(compare_w, fontsize=10)
    ax4.set_ylim(0, 95); ax4.set_title('Weather Robustness Radar', pad=20, fontweight='bold')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig5_training_dashboard.png'), bbox_inches='tight')
plt.close()
print('✓ Fig 5: Training dashboard')

# ═══════════════════════════════════════════════════
# Print summary tables
# ═══════════════════════════════════════════════════
print('\n' + '='*70)
print('TABLE 1: Per-Weather 3D AP @ IoU=0.3')
print('='*70)
print(f'{"Weather":<14} {"v1+OT":>8} {"v2":>8} {"Δ":>8}')
print('-'*40)
for w in weather_order:
    if w in v1_ap and w in weather_ap:
        print(f'{w:<14} {v1_ap[w]:>7.2f}% {weather_ap[w]:>7.2f}% {weather_ap[w]-v1_ap[w]:>+7.2f}')
    elif w in weather_ap:
        print(f'{w:<14} {"N/A":>7}  {weather_ap[w]:>7.2f}%')

print('\n' + '='*70)
print('TABLE 2: Ablation Components')
print('='*70)
print(f'{"Component":<30} {"v1+OT":<12} {"v2":<12}')
print('-'*55)
components = [
    ('Camera-LiDAR cross-attn', '✗', '✓'),
    ('Enhanced weather head', '✗', '✓'),
    ('OT warmup schedule', '✗', '✓'),
    ('Branch entropy reg', '✗', '✓ (lambda=0.005)'),
    ('Branch weight floor', '(0.1, mix only)', '0.12 (dual)'),
    ('Weather balanced sampler', '✓', '✓'),
]
for comp, v1s, v2s in components:
    print(f'{comp:<30} {v1s:<12} {v2s:<12}')
cam_alive = epoch_19_data.get('camera', 0) * 100
print(f'Camera branch alive: v1=0% v2={cam_alive:.0f}%')

print(f'\nAll figures saved to: {OUT}')
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f'  {f} ({size/1024:.0f} KB)')
