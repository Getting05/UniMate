# Astro P2：从文本生成机器人动作数据

此接入使用 UniMate 官方 `unimate_uniml3d_f60_v3` 预训练模型，不需要重新训练。
输入英文动作描述，先生成 P2 骨架动画，再按 P2 的真实关节轴和关节限位进行全身拟合，输出 30 自由度机器人轨迹。

**输出属于运动学候选数据。** UniMate 不知道电机速度/力矩限制、支撑稳定性和控制器能力。限位拟合不能保证动力学可执行；应先检查视频和质量报告，再做跟踪仿真或数据集筛选。

## 1. 已部署的 4090 主机

| 项目 | 路径 / 值 |
|---|---|
| SSH | `ssh getting@10.40.1.45` |
| 代码 | `/data/getting/UniMate` |
| Python 环境 | `/data/getting/.venvs/unimate`，Python 3.10 |
| P2 原始模型 | `/data/getting/unimate-assets/astro_p2/` |
| 使用的 MJCF | `mjcf/astro_p2_30dof_primitive_collision.xml` |
| 预训练模型 | `outputs/unimate_uniml3d_f60_v3/` |
| Hugging Face 缓存 | `/data/getting/unimate-cache/huggingface` |
| P2 条件和映射 | `outputs/rig/astro_p2/` |
| 安装和验证日志 | `/data/getting/unimate-logs/` |

模型来自本机提供的 `menagerie_x/assets/astro_p2`，原始 MJCF/网格整体复制到远端。代码仓库保存接入代码和语义配置；原始模型、权重和生成数据保留在主机，不随 Git 提交。

每次登录后：

```bash
cd /data/getting/UniMate
source /data/getting/.venvs/unimate/bin/activate
export HF_HOME=/data/getting/unimate-cache/huggingface
export OMP_NUM_THREADS=1
export MUJOCO_GL=egl
```

首次下载或更新权重时，若直连失败，可使用主机已有的代理：

```bash
export HTTPS_PROXY=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
```

缓存完整后可用 `export HF_HUB_OFFLINE=1` 离线生成；首次下载时不要设置它。

## 2. 一条命令生成、转换和录制

```bash
NUM_REPETITIONS=3 SEED=10 bash scripts/run_p2.sh \
  outputs/p2/walk_001 "An object walks forward."
```

同时生成多个动作：

```bash
NUM_REPETITIONS=3 SEED=20 bash scripts/run_p2.sh outputs/p2/batch_001 \
  "An object walks forward." \
  "An object raises its right arm." \
  "An object squats and then stands up."
```

使用短英文句子，每条描述一个动作。沿用训练描述的 `An object ...` 风格。
这些描述是使用示例，不表示每次都能生成对应的高质量机器人动作。

默认每个描述生成 3 次、`cfg_scale=3`、GPU 批大小 4，输出视频。
可用 `NUM_REPETITIONS`、`SEED`、`BATCH_SIZE`、`CFG_SCALE` 覆盖。
`RENDER=0` 跳过视频；减少 `BATCH_SIZE` 可降低显存占用。
每次使用新的输出目录，脚本拒绝混入已有结果。

```text
outputs/p2/walk_001/
├── manifest.json        UniMate 资产、动作和提示词关联
├── motions/*.npy        原始生成特征，保留用于重新拟合
└── robot/
    ├── *.npz           30 自由度机器人数据
    ├── *.json          拟合、关节速度、穿地和自碰撞报告
    └── *.mp4           P2 网格运动学回放
```

官方模型每段输出 60 帧。本流程按 30 FPS 导出，即每段约 2 秒。
修改 `--fps` 仅改变时间解释和速度计算，不会增加帧数，也不是重采样。
需要较长动作时可以先用上游 `--motion_expand` 生成，再用下节的转换命令逐个转换；接缝和累积漂移需要额外检查。

## 3. 分步骤执行

### 重建机器人条件

更换或修改 MJCF 后必须重新构建，不要直接沿用旧映射。

```bash
python -m unimate.robots.p2 prepare \
  --mjcf /data/getting/unimate-assets/astro_p2/mjcf/astro_p2_30dof_primitive_collision.xml \
  --output outputs/rig/astro_p2
```

`summary.json` 保存模型签名、语义标签、尺寸比例和源 MJCF 路径。
`configs/robots/astro_p2.json` 是可审查的标签和末端辅助点配置。
语义标签改变会影响生成分布；修改后重新 prepare 并生成对照样例。

### 只生成原始动画

```bash
python -m unimate.inference.sample \
  --exp_dir outputs/unimate_uniml3d_f60_v3 \
  --asset outputs/rig/astro_p2 \
  --prompt "An object walks forward." \
  --num_repetitions 3 --seed 10 --batch_size 4 \
  --only_save_motion --output_dir outputs/p2/raw_walk
```

### 将一个生成文件转为 P2 数据

将 `MOTION.npy` 替换为上一步 `motions/` 中的具体文件。

```bash
python -m unimate.robots.p2 convert \
  --asset outputs/rig/astro_p2 \
  --motion outputs/p2/raw_walk/motions/MOTION.npy \
  --output outputs/p2/converted/walk.npz --ground
```

`--ground` 根据两脚碰撞几何的全片最低点做一次恒定高度平移；它保留跳跃和根轨迹形状，不逐帧贴地，不修复滑脚，也不建立接触标签。
省略该参数可检查生成器原始根高度。
`--temporal` 默认 0.02，抑制相邻帧拟合关节变化；设为 0 可关闭。
`--max-nfev` 默认 60；报告若显示未收敛帧，可提高此值重试，并比较误差和视频。

### 回放为视频

```bash
MUJOCO_GL=egl python -m unimate.robots.p2 render \
  --asset outputs/rig/astro_p2 \
  --motion outputs/p2/converted/walk.npz \
  --output outputs/p2/converted/walk.mp4
```

回放使用 MuJoCo 正向运动学，不是控制器跟踪仿真。
若移动了模型目录，给 `convert` / `render` 增加 `--mjcf 新路径`；模型签名必须一致。

## 4. 数据字段和关节顺序

```python
import numpy as np
x = np.load("outputs/p2/converted/walk.npz", allow_pickle=False)
print(x["dof_pos"].shape)  # (T, 30)
print(x["joint_names"].tolist())
```

| 字段 | 含义 |
|---|---|
| `model_signature` | 本次导出所用模型的运动学签名 |
| `fps` | 每秒帧数 |
| `qpos` | `(T,37)`；根位置 XYZ + 根四元数 **WXYZ** + 30 关节角 |
| `root_pos` | `(T,3)`，米；X 向前、Y 向左、Z 向上 |
| `root_rot` | `(T,4)`，**XYZW** |
| `dof_pos` / `dof_vel` | `(T,30)`，弧度 / 弧度每秒 |
| `joint_names` | 30 个关节的明确列顺序，以 MJCF 顺序为准 |
| `qvel` | `(T,36)`，MuJoCo `mj_differentiatePos` 定义的广义速度 |
| `body_pos` / `body_rot` | 所有非 world body 的世界坐标 / **XYZW** 四元数 |
| `body_names` | body 字段的列顺序 |
| `projection_error` | `(T,36)`；生成目标到拟合后机器人 FK 点的距离，米，含 5 个辅助点；在恒定地面对齐之前计算 |
| `foot_clearance` | `(T,2)`；左右脚碰撞几何距 Z=0 的最低高度，米 |
| `self_collision_count` | 每帧 MuJoCo 检出的穿透超过 1 mm 的自碰撞 contact 数 |
| `max_self_penetration` | 每帧上述接触中最大穿透深度，米 |

速度由相邻帧差分计算，最后一帧复用前一帧速度。
不能把 WXYZ 和 XYZW 混用，也不能按别的机器人/数据集列顺序直接读取 `dof_pos`。
这不是 ProtoMotions `MotionLib .pt` 文件；接入该训练框架时还需按目标 skeleton、顺序和分割规则转换。

## 5. 如何筛选结果

先看 `*.json` 和视频，再决定是否保留：

- `projection_mean_m` / `projection_max_m`：过大说明生成动画无法很好满足 P2 关节约束。
- `solver_success_frames` 应等于 `solver_total_frames`；收敛不代表误差足够小。
- `max_joint_speed_rad_s`：应结合你的实际电机/控制器能力判断；当前代码没有电机速度或力矩上限。
- `self_collision_frames` / `max_self_penetration_m`：检查手穿躯干、腿交叉等；检测覆盖所选 MJCF 的碰撞几何和排除规则，不能代表全部网格表面。
- `raw_minimum_foot_clearance_m` 是对齐前最低高度；`minimum_foot_clearance_m` 和 `constant_ground_shift_m`：判断高度是否异常。恒定对齐后最低点为零不代表每一帧都有合理支撑。
- 视频中检查左右动作是否正确、滑脚、抖动、漂浮、躯干姿态和转身。

拟合器固定生成的根平移和根朝向，对全部 30 个关节统一优化位置/朝向残差并施加真实关节限位。
它不约束足部接触、质心、碰撞、关节速度或动力学；这些问题会被部分报告，但不会自动消除。
`status=kinematic_candidate_not_dynamics_validated` 是明确的输出状态，不要仅因为脚本成功就把结果当成可直接上机的数据。

## 6. 安装和复现

4090 主机已经建立专用环境；需要重装时：

```bash
cd /data/getting/UniMate
export P2_MJCF=/data/getting/unimate-assets/astro_p2/mjcf/astro_p2_30dof_primitive_collision.xml
export UNIMATE_VENV=/data/getting/.venvs/unimate
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
bash scripts/setup_p2.sh
```

脚本使用 `uv`、Python 3.10、上游完整依赖和 `requirements-p2.txt`，下载主权重及 T5。
源模型是外部输入，不会从 GitHub 自动取得。

运行接入测试：

```bash
P2_MJCF=/data/getting/unimate-assets/astro_p2/mjcf/astro_p2_30dof_primitive_collision.xml \
  OMP_NUM_THREADS=1 python -m pytest -q tests/test_p2.py
```

测试用独立 MuJoCo FK 姿态验证坐标手性、全部末端关节可观测性、特征往返、限位拟合、模型签名和非法输入拒绝。
未设置 `P2_MJCF` 时会跳过模型集成测试；跳过不算验证通过。

## 7. 本次实测（2026-10-07）

4090 上离线加载官方 EMA 权重，seed=10、cfg_scale=3、batch_size=2，分别生成走路和抬右手各两次。输出在 `outputs/p2/demo_20261007/`，每个 60 帧，共 240 帧。

| 样例 | 平均拟合误差 | 最大拟合误差 | 自碰撞帧 | 最大关节速度 |
|---|---:|---:|---:|---:|
| 抬右手 rep 0 | 7.31 mm | 35.13 mm | 0 / 60 | 9.88 rad/s |
| 抬右手 rep 1 | 6.96 mm | 51.96 mm | 0 / 60 | 9.90 rad/s |
| 走路 rep 0 | 4.72 mm | 28.63 mm | 0 / 60 | 9.39 rad/s |
| 走路 rep 1 | 3.90 mm | 22.51 mm | 3 / 60 | 9.88 rad/s |

四个样例的 240 帧拟合均收敛，导出角度在真实限位内，全部四元数归一化；走路 rep 1 最大自碰撞穿透约 3.35 mm。此批数据未验证动力学或真机跟踪，不能作为已合格数据集。原始报告：`outputs/p2_validation/demo_report.json`。

独立已知姿态的编码/解码/拟合往返最大点误差约 2.2 微米；4 项集成测试通过。真实 CUDA 矩阵运算、T5/UniMate 推理、MuJoCo EGL 网格视频录制均完成。视频抽帧检查确认抬臂和步行动作可见。

权重 SHA256 已与 Hugging Face 文件元数据核对：`f991ddd87262c251f4681d18297a6163ab4f139daa28c86d5ef0a66771df7bc6`。完整环境版本位于 `outputs/p2_validation/environment-freeze.txt`。

上游 Blender 4.0 wheel 在本机 `pip check` 会提示平台标签不匹配，但 `import bpy` 实测成功并返回 4.0.0；P2 流程使用 MuJoCo，不依赖 Blender 渲染。此说明不代表上游全部 Blender 数据处理功能已验证。

## 8. 设计说明和许可

P2 保留 30 个 hinge 加 1 个 free root。UniMate 使用 36 个节点：31 个机器人 body 加左右手、左右脚、头部 5 个辅助末端点。
上游把父节点旋转编码在子节点，因此辅助点用于保留末端关节旋转；它们不进入 `dof_pos`。
模型空间是 Y-up、+Z-forward、骨架树直径 2；导出时恢复米制 Z-up。

接入代码沿用仓库代码许可。原始机器人模型保留来源权利，未通过此接入重新许可。
官方 UniMate 检查点采用 **CC BY-NC 4.0**；使用范围请查看 [官方模型许可](https://huggingface.co/Linzhan/UniMate/blob/main/LICENSE)。
