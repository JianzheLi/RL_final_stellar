# HLSMAC 自定义地图添加教程

本教程将指导你如何在 HLSMAC 框架中添加一个我们制作的新的 StarCraft II 自定义地图，并使其支持强化学习训练。

## 目录
1. [准备工作](#1-准备工作)
2. [步骤一：放置地图文件](#步骤一放置地图文件)
3. [步骤二：配置地图注册表](#步骤二配置地图注册表)
4. [步骤三：编写环境代码](#步骤三编写环境代码)
5. [步骤四：注册环境类](#步骤四注册环境类)
6. [步骤五：测试与运行](#步骤五测试与运行)

---

## 1. 准备工作

确保你已经制作好了 `.SC2Map` 地图文件。
* **我方单位**：Owner 必须设为 **Player 1**。
* **敌方单位**：Owner 必须设为 **Player 2**。
* **单位属性**：确认你知道所有单位的 `Unit Type ID`（可在编辑器查看，或通过代码调试打印）。

---

## 步骤一：放置地图文件

将你的地图文件（例如 `payy.SC2Map`）放置在星际争霸安装目录下的 `Maps/Tactics_Maps/`(或者适合你对应地图文件夹) 文件夹中。

**路径示例：**
* Windows: `~\StarCraft II\Maps\Tactics_Maps\pzyy.SC2Map`
* Linux: `/home/user/StarCraftII/Maps/Tactics_Maps/mymap.SC2Map`

> **注意**：必须放在 `Tactics_Maps` 子文件夹内，否则代码无法找到。

---

## 步骤二：配置地图注册表

打开文件：`smac/env/sc2_tactics/maps/sc2_tactics_maps.py`

在 `map_param_registry` 字典中添加你的地图配置。

```python
    "pzyy": {
        "n_agents": 11,          # 10 枪兵 + 1 寡妇雷
        "n_enemies": 25,         # 22 跳虫 + 1 眼虫
        "limit": 10,            # 时间限制，可根据需要调整
        "a_race": "T",           # 我方是人族
        "b_race": "Z",           # 敌方是虫族
        "unit_type_bits": 5,     # 这是一个掩码，稍微大点没关系
        "map_type": "pzyy",      # 对应下面要写的环境类
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "n_actions": 12,     
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
                "BurrowDown": 2095, # 寡妇雷埋地 ID
                "BurrowUp": 2097,   # 寡妇雷出地 ID
            },
            "unit_id_dict": {
                "marine": 48,
                "widow_mine": 498,
                "widow_mine_burrowed": 500,
                "zergling": 105,
                "overseer": 129
            },
        },
    },
    "ldtj": {
        "n_agents": 7,           # 5 异龙 + 2 孢子爬虫
        "n_enemies": 5,          # 4 寡妇雷 + 1 架起的坦克
        "limit": 120,            # 游戏时间限制
        "a_race": "Z",           # 我方虫族
        "b_race": "T",           # 敌方人族
        "unit_type_bits": 3,     # 预留足够的 bit 给映射后的 ID (0, 1)
        "map_type": "ldtj",      # 对应下面要新建的类名
        "support_info": {
            "n_actions_no_attack": 6,
            "n_actions_move": 4,
            "action_set": {
                "move": 16,
                "attack": 23,
                "stop": 4,
            },
            "unit_id_dict": {
                "mutalisk": 108,
                "spore_crawler": 98,
                "widow_mine": 498,
                "siege_tank_sieged": 33
            }
        },
    },
```
## 步骤三：配置地图注册表

在 `smac/env/sc2_tactics/` 中，把 `star36env_pzyy.py` 与 `star36env_ldtj.py` 复制进去

## 步骤四：注册环境类

这里需要告诉你的算法目录，对应的地图在哪里，这里以dTAPE为例。
打开文件：`RLalgs/dTAPE/src/envs/__init__.py`
导入你的新环境类：
```python
from smac.env.sc2_tactics.star36env_pzyy import SC2TacticsPZYYEnv
from smac.env.sc2_tactics.star36env_ldtj import SC2TacticsLDTJEnv
```
在 `env_fn` 或判断逻辑中添加分支：
```python
    elif kwargs.get("map_name") == "pzyy":  # <--- 新增这个判断
        return SC2TacticsPZYYEnv(**kwargs)
    elif kwargs.get("map_name") == "ldtj":   # <--- 新增
        return SC2TacticsLDTJEnv(**kwargs)
```
## 步骤五：运行
```bash
$env:SC2PATH="~\StarCraft II"  #你的游戏
$env:PYTHONPATH="$PWD;$PWD\RLalgs\dTAPE\src$PWD\smac" #你的环境
python RLalgs/dTAPE/src/main.py --config=d_tape --env-config=sc2te --capture=sys with env_args.map_name=ldtj`#运行训练
```