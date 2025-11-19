#!/bin/bash
# 检查所有12张HLSMAC地图的训练状态
# 使用Python脚本提取准确的训练指标

cd "$(dirname "$0")"

# 调用Python脚本进行详细检查
python3 << 'PYEOF'
import os
import re
from datetime import datetime

MAPS = {
    "adcc": "暗度陈仓",
    "dhls": "调虎离山", 
    "fkwz": "反客为主",
    "gmzz": "关门捉贼",
    "jctq": "金蝉脱壳",
    "jdsr": "借刀杀人",
    "sdjx": "声东击西",
    "swct": "上屋抽梯",
    "tlhz": "偷梁换柱",
    "wwjz": "围魏救赵",
    "wzsy": "无中生有",
    "yqgz": "欲擒故纵"
}

def extract_metrics(log_file):
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    metrics = {}
    for i in range(len(lines)-1, -1, -1):
        if "Recent Stats" in lines[i] and "t_env:" in lines[i]:
            t_env_match = re.search(r't_env:\s+(\d+)', lines[i])
            if t_env_match:
                metrics['t_env'] = int(t_env_match.group(1))
            
            for j in range(i, min(i+30, len(lines))):
                line = lines[j]
                if 'test_battle_won_mean' in line:
                    match = re.search(r'test_battle_won_mean:\s+([\d.]+)', line)
                    if match:
                        metrics['win_rate'] = float(match.group(1))
                if 'td_error_abs' in line:
                    match = re.search(r'td_error_abs:\s+([\d.]+)', line)
                    if match:
                        metrics['td_error'] = float(match.group(1))
                if 'test_reward_mean' in line:
                    match = re.search(r'test_reward_mean:\s+([\d.]+)', line)
                    if match:
                        metrics['reward'] = float(match.group(1))
            
            if 't_env' in metrics:
                break
    
    return metrics if metrics else None

def check_process(map_name):
    import subprocess
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if f'map_name={map_name}' in line and 'python3' in line and 'main.py' in line:
                parts = line.split()
                if len(parts) >= 11:
                    return {
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'time': parts[9]
                    }
    except:
        pass
    return None

print("=" * 60)
print("HLSMAC 12张地图训练状态汇总")
print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print()

total_running = 0
total_stopped = 0
completed = []

for map_code, map_name in MAPS.items():
    print("-" * 60)
    print(f"地图: {map_code:4s} ({map_name})")
    print("-" * 60)
    
    process = check_process(map_code)
    if process:
        print(f"状态: ✅ 训练中 (PID: {process['pid']}, CPU: {process['cpu']}%, MEM: {process['mem']}%, 运行: {process['time']})")
        total_running += 1
    else:
        print("状态: ⏸️  未运行")
        total_stopped += 1
    
    log_file = f"RLalgs/dTAPE/results/sacred/{map_code}/ow_qmix_env=4_adam_td_lambda/1/cout.txt"
    metrics = extract_metrics(log_file)
    
    if metrics:
        t_env = metrics.get('t_env', 0)
        progress = (t_env / 2005000) * 100 if t_env > 0 else 0
        print(f"进度: {t_env:,} / 2,005,000 ({progress:.1f}%)")
        
        if progress >= 99:
            completed.append(map_code)
        
        if 'win_rate' in metrics:
            win_rate_pct = metrics['win_rate'] * 100
            print(f"胜率: {metrics['win_rate']:.4f} ({win_rate_pct:.2f}%)")
        if 'td_error' in metrics:
            print(f"TD误差: {metrics['td_error']:.4f}")
        if 'reward' in metrics:
            print(f"奖励: {metrics['reward']:.4f}")
    else:
        if os.path.exists(log_file):
            print("日志: ✅ 存在但无最新指标")
        else:
            print("日志: ❌ 未找到")
    
    print()

print("=" * 60)
print("汇总统计")
print("=" * 60)
print(f"正在训练: {total_running}/12")
print(f"未运行: {total_stopped}/12")
if completed:
    print(f"接近完成 (≥99%): {len(completed)}/12 - {', '.join(completed)}")
print("=" * 60)
PYEOF

