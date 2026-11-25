# RobustHFL-Thesis

OpenI 托管任务的 V2 入口为 `scripts/openi_entry.py`。默认运行环境检查，网页字段与单种子预实验见 [OpenI V2 部署指南](docs/research/OPENI_DEPLOY_GUIDE.md)。

硕士课题《面向图像损坏的模型异构联邦鲁棒学习方法研究》的 PyTorch 实验仓库。研究点一验证私有训练样本信号与组级边际训练价值之间的关系。研究点二验证公共代理上的无标签指标能否识别学生依赖的正迁移。独立运行两组探针后，可用 A/B/C/D 比较两个原型的主效应与交互效应。

## 环境与数据

使用 Python 3.10 或更新版本。若新建环境，先按本机 CUDA 环境安装配对版本的 PyTorch 和 torchvision，再安装其余依赖。默认 `device: auto` 有可用 CUDA 时使用 GPU。显式传入 `--device cuda` 可在 CUDA 构建不可用时立即退出。本机 Anaconda base 是 CPU 版 PyTorch，已有的 `pytorch` 环境是通过 RTX 3050 验证的 CUDA 构建。完整版本见 [审计报告](results/audit_report.md)。

```powershell
python -m pip install -r requirements.txt
.\scripts\clone_references.ps1
python -m core.splits --config configs/base.yaml
python -m scripts.smoke --with-adapters
```

正式配置会自动下载 CIFAR-10 和 CIFAR-100 到 `data/` 并校验官方归档及读取成员的 MD5。也支持把官方归档或官方解包目录放在同一数据根目录，或给各入口传入 `--data-root D:\datasets\cifar`。第一次运行冒烟时需要已下载的数据，冒烟入口本身关闭下载并设有 170 秒看门狗。可对克隆脚本传 `-Proxy` 设置 Git HTTP 代理。运行时不导入参考仓库。

切分使用源限定键 `(dataset/split, index)`。CIFAR-10 官方训练集先留出同域代理，再按 IID 或 `split.partition: dirichlet` 分配客户端，最后从各客户端独立留出验证。两套代理只返回无标签像素。CIFAR-10 测试索引存入清单，测试图像和标签只在最终评测入口读取。改种子、切分策略或客户端数量时，给新配置指定新的 `--manifest`。冒烟与正式配置各自冻结一份清单。

训练损坏成员由客户端与数据角色种子固定。损坏施加于 `[0,1]` 像素副本，之后执行共享训练常量标准化。验证报告同一批本地留出索引的干净精度、配置比例下的混合精度和客户端指定类型及强度下的全损坏精度。后者作为 `corrupt_acc` 和 ΔR 的鲁棒端点，并用于客户端宏平均及最差客户端统计。高斯噪声和散焦模糊使用 CIFAR-C 生成脚本的五档参数。散焦用 SciPy 实现同半径和抗锯齿核。运行时保留浮点像素。

## 正式运行

在仓库根目录执行。以下配置使用固定种子 20261124。对其他机器，把 Conda 环境路径替换成本机路径。

```powershell
$pytorchEnv = 'D:\anaconda3\envs\pytorch'
$env:Path = "$pytorchEnv\Library\bin;$pytorchEnv\Scripts;$env:Path"
$pytorchPython = Join-Path $pytorchEnv 'python.exe'
& $pytorchPython -m probes.probe1_local_value --config configs/probe1.yaml --out results/runs/probe1 --device cuda
& $pytorchPython -m probes.probe2_freq_delta_r --config configs/probe2.yaml --prepare-checkpoints --out results/runs/probe2 --device cuda
& $pytorchPython -m methods.joint_framework --config configs/abcd.yaml --groups A B C D --out results/runs/abcd --device cuda
```

`--prepare-checkpoints` 先按同一协议训练标准 KD，再读取生成的四个客户端 checkpoint。复用同协议、同视图签名的 checkpoint 时，改用 `--checkpoints <客户端 checkpoint 所在目录>`。每次训练在 `results/checkpoints/<实验>/<run_id>/` 保存客户端权重和优化器状态。日志中的 `checkpoints` 事件记录目录和单文件哈希。Local-CE、FedMD-style KD、RAHFL、AugHFL 各自支持 `python -m baselines.<模块名> --config configs/base.yaml`。PowerShell 入口为 `.\scripts\run_experiments.ps1 -Experiment <smoke|probe1|probe2|abcd|fedmd|local|rahfl|aughfl>`，可用 `-Python` 指定 Conda 环境解释器，用 `-Device cuda` 要求 GPU。

### 研究点一

同一客户端的 EMA 原型和跨轮损失记录只访问私有训练数据。两轮 CE 预热之后，计算余弦相似度、弱扰动预测 JS、逐样本 CE、跨轮 CE 斜率和随机对照。原始信号值与客户端内百分位均记录。候选质量取相似度与反向 JS 两项的等权平均。原始模式用固定理论区间映射，百分位模式使用训练样本的经验秩。正式配置在相同架构及相同损坏类型的两个客户端上对照 0% 与 60%。统计表按损坏成员划分，固定质量线 0.5 只用于报告接受率。

每段独立抽取等量样本，从同一预热模型和带动量的优化器状态启动保留、软降权和替换三臂。替换优先同标签并排除该分位段。若同标签样本不足，则从剩余候选无放回抽取，并记录实际匹配率。干预臂步数和每步样本数相同。五种训练损失对照的初始状态、批顺序与步数相同。随机和损失降权仅调节 CE，固定一致性与候选方法加入 JS。可在 YAML 对 `local.consistency_weight` 选用 `const_1`、`linear_q`、`bell_4q1mq`。

| 输出 | 内容 |
| --- | --- |
| `signal_marginal_intervention.csv` | 三臂样本数、索引摘要、替换匹配率、相对初值和替换臂的干净及鲁棒变化 |
| `signal_statistics.csv`、`quality_acceptance.csv` | 每客户端、尺度、损坏成员的分布与固定质量线统计 |
| `signal_ranking.csv`、`local_comparisons.csv` | 信号与保留相对替换收益的统计及五种本地损失对照 |

### 研究点二

每个学生、教师、频带和重复组合都从完全相同的学生权重与优化器起步。每次重复在两域内各自锁定一串公共 batch，记录源内索引哈希。同一域及重复的所有臂共享原始 batch、学习率和更新步数。全频、低频、高频均先在像素空间构造，模型使用同一标准化。默认 `eps=0.3`、截止频率 `0.17` cycles/pixel。`frequency.eps` 可调整干预强度。候选指标只使用公共预测，跨视图一致性由相对其余两个视图的平均 JS 计算。两个客户端时没有第三个模型可形成跨教师共识，共识列为空。

`student_teacher_band_gain.csv` 记录 ΔR、两端 checkpoint 哈希和五项候选信号。`gain_ranking.csv` 分域、分学生以及汇总报告 Spearman、AUROC、AUPRC 和最高评分组的正收益比例。正收益定义为 ΔR > 0。无排序方差或缺少二分类标签时，相应统计留空。六种 KD 对照共享学生起点与更新步数，每步都执行三个视图的教师和学生前向、一次反向，单视图对照只给选中的视图非零权重。原型 C/D 的频带权重根据教师 MI 与学生–教师 JS 差值计算，至少 `gamma_full` 的总权重分配给全频。

### A/B/C/D 与日志

四组各自初始化且共享客户端种子、数据视图和优化器更新预算。A/B 使用单视图标准 KD，C/D 使用三视图频率 KD。B/D 在同一轮次开启本地质量调制。`groups.csv` 逐组列出步数、耗时、显存和通信量，`interaction.csv` 对四项验证指标计算 `(D−B)−(C−A)`。

各 CSV 行有匹配的 JSONL 记录。指标精度以比例表示。中间轮日志的 `ckpt_hash` 是模型状态摘要，最终轮是文件摘要，组级字段是有序客户端摘要的再哈希。探针记录所用起点的文件摘要。GPU 时间是 CUDA stream 区间，显存是 PyTorch 峰值分配量。CPU 运行的两项 GPU 字段为空。通信字节按模拟上传和下发的张量负载计算。用不同的 `--out` 可并行运行多个进程。

### 最终评测

冻结方案和 checkpoint 后，使用 [官方 CIFAR-10-C](https://zenodo.org/records/2535967) 中的解包文件。最终入口对完整官方测试集和 CIFAR-10-C 的五个强度进行评测，并验证标签顺序。可以使用 `--corruptions` 列出预先选定的损坏类型。该入口独立于训练、阈值设定和教师路由。

```powershell
python -m core.final_eval --config configs/abcd.yaml --checkpoints PATH_TO_FROZEN_GROUP --cifar-c data/CIFAR-10-C --out results/runs/final
```

本轮验证见 [GPU 冒烟日志](results/smoke_cuda_console.log)、[GPU 产物表头及 SHA-256](results/smoke_cuda_artifact_manifest.json) 和 [CPU 冒烟日志](results/smoke_verified_console.log)。参考实现的差异及环境记录见 [审计报告](results/audit_report.md)。
