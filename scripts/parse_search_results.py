#!/usr/bin/env python3
"""解析搜参日志，提取分支权重演化，输出对比表格。"""
import os, re, sys, glob

def parse_branch_monitor(log_path):
    """从训练日志提取 Branch Weight Monitor 行。"""
    results = []
    with open(log_path) as f:
        for line in f:
            if 'Branch Weight Monitor' not in line:
                continue
            # 解析 epoch
            m_epoch = re.search(r'\[Epoch (\d+)\]', line)
            epoch = int(m_epoch.group(1)) if m_epoch else -1
            
            # 解析各分支 mean
            m_l = re.search(r'Lidar\s+mean=([\d.]+)', line)
            m_r = re.search(r'Radar\s+mean=([\d.]+)', line)
            m_c = re.search(r'Camera\s+mean=([\d.]+)', line)
            m_ent = re.search(r'Entropy\(mean/std\)=([\d.]+)', line)
            
            if all([m_l, m_r, m_c, m_ent]):
                results.append({
                    'epoch': epoch,
                    'lidar': float(m_l.group(1)),
                    'radar': float(m_r.group(1)),
                    'camera': float(m_c.group(1)),
                    'entropy': float(m_ent.group(1)),
                })
    return results

def score_branch_health(results):
    """评分：熵越高越好，最低分支权重越高越好，分支不贴地板越好。"""
    if not results:
        return 0.0
    last = results[-1]
    # 平均熵
    avg_entropy = sum(r['entropy'] for r in results) / len(results)
    # 最后一轮最低分支权重
    min_weight = min(last['lidar'], last['radar'], last['camera'])
    # 地板接近度惩罚（最后一个 epoch）
    floor_hit_penalty = sum(
        1.0 for w in [last['lidar'], last['radar'], last['camera']] if w < 0.22
    )
    score = avg_entropy * 0.5 + min_weight * 1.5 - floor_hit_penalty * 0.2
    return score

def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else './logs'
    log_files = sorted(glob.glob(f'{log_dir}/**/*.log', recursive=True))
    
    # 按搜索阶段筛选
    search_logs = [f for f in log_files if 'search_phase' in f]
    if not search_logs:
        # fallback: 找最近的实验日志
        search_logs = sorted(glob.glob('./logs/exp_*/train_epoch/*'), reverse=True)[:5]
        print("未找到 search_phase 日志，显示最近实验：")
    
    print(f"{'Run':<30} {'E0 L/R/C':>18} {'E2 L/R/C':>18} {'E4 L/R/C':>18} {'Score':>8}")
    print("-" * 95)
    
    scored = []
    for logf in search_logs:
        results = parse_branch_monitor(logf)
        if not results:
            continue
        name = os.path.basename(logf).replace('.log', '')[:28]
        score = score_branch_health(results)
        
        def fmt_epoch(e):
            for r in results:
                if r['epoch'] == e:
                    return f"{r['lidar']:.2f}/{r['radar']:.2f}/{r['camera']:.2f}"
            return "N/A"
        
        e0 = fmt_epoch(0)
        e2 = fmt_epoch(2)
        e4 = fmt_epoch(4)
        
        print(f"{name:<30} {e0:>18} {e2:>18} {e4:>18} {score:>8.3f}")
        scored.append((name, score, results))
    
    if scored:
        scored.sort(key=lambda x: -x[1])
        print(f"\n🏆 Best: {scored[0][0]} (score={scored[0][1]:.3f})")

if __name__ == '__main__':
    main()
