# RobustHFL-Thesis 在 OpenI 网页运行

这个项目按 OpenI 平台托管任务部署。先上传 CIFAR 数据集，把源码放进 OpenI 能选到的代码仓库，再在网页创建 GPU 训练任务。第一次只提交 CUDA 冒烟。该流程不需要 SSH、JupyterLab 或服务器登录信息。[OpenI 官方指南](https://openi.pcl.ac.cn/guide/index.html)

## 1. 上传数据，准备代码仓库

**数据集。** 在 OpenI 创建一个数据集，建议显示名为 `RobustHFL CIFAR 官方归档 v1`。按自己或组织的权限选可见性，上传 `robusthfl-cifar-v1.zip`，等待页面显示处理完成。记录实际数据集页面及 `owner/repo_id`，用于任务选择。这个 ZIP 根目录只含下面两个官方归档：

```text
robusthfl-cifar-v1.zip
├── cifar-10-python.tar.gz
└── cifar-100-python.tar.gz
```

上传的是外层 ZIP。保留里面两个原始 TAR.GZ，脚本会校验各自的官方 MD5 并记录 SHA-256。CIFAR-10-C 无需上传到训练任务。Windows 源文件、训练结果与凭据无需放进数据集。[CIFAR 官方页面](https://cave.cs.toronto.edu/kriz/cifar.html) · [官方归档校验](https://zenodo.org/records/10089977)

如需重新生成同一版上传文件，从本地仓库根目录执行，输出目录应为新位置：

```powershell
python -m scripts.openi.data --data-dir data --manifest openi-transfer/cifar.json --bundle openi-transfer/robusthfl-cifar-v1.zip
```

**代码。** 将源码提交到 OpenI 能选择的仓库和分支。可从现成的源码 ZIP 解包获得干净的目录，再按目标仓库流程提交。ZIP 本身是源码传输包。网页的启动文件字段应选仓库内实际存在的 `openi_web.py`。本地尚无可选择的远端或 HEAD 提交，仓库准备、数据上传、网页作业创建均由用户操作。生成源码包的本地命令如下：

```powershell
python -m scripts.openi.source_bundle --out openi-transfer/robusthfl-openi-web-source.zip
```

## 2. 在网页建第一次任务

创建训练任务，选择对应仓库、分支、已上传的 CIFAR 数据集，以及分配一张 V100 的 PyTorch GPU 镜像。镜像需要 Python 3.10 或更新版本、兼容 V100 的 PyTorch/torchvision、c2net 和 `requirements-openi.txt` 中的实验依赖。当前没有经过实际 OpenI 任务验证的镜像标签。先以平台可选镜像试冒烟，从日志确认依赖和驱动。环境兼容性以实际 CUDA 反向、FFT 和 torchvision CUDA 算子测试为准。[PyTorch 官方版本列表](https://pytorch.org/get-started/previous-versions/) · [c2net SDK 接口](https://pypi.org/project/c2net/)

| 网页字段 | 第一次填写 |
| --- | --- |
| 任务名 | `robusthfl-v100-smoke-01` |
| 代码仓库及分支 | 你实际提交源码的仓库及分支 |
| 数据集 | 刚才上传的 CIFAR 数据集 |
| 算力 | 一张 V100 |
| 启动文件 | `openi_web.py` |
| 参数名 `run_id` | `v100-smoke-01` |
| 参数名 `stage` | `smoke` |

其余参数默认省略。入口调用 `c2net.prepare()` 获取数据和输出路径。在挂载根及两层子目录内，支持平台解包后的同目录 CIFAR 双归档，也支持尚未解包的 `robusthfl-cifar-v1.zip`。如挂载里有多个版本，给网页参数 `dataset_subdir` 指定相对挂载根的那一份。启动器拒绝从挂载外取数据。网页镜像自动分配的 GPU 以逻辑 CUDA 设备使用，无需填写设备号或 Windows 路径。

等任务完成后查看控制台，并下载 `RobustHFL_v100-smoke-01_*_complete_*.tar.gz` 及同名 `.sha256`。查看包内 `status.json` 和 `preflight_*.json`，确认完成状态、实际 GPU 型号、驱动、显存、磁盘和依赖版本。两个客户端训练两轮，第一轮保存 checkpoint 后从该状态恢复，再执行两个小探针。项目给冒烟工作进程设有 170 秒上限。平台数据准备或结果发布超时也会导致任务失败，届时先看日志判断发生在哪个阶段。首次冒烟通过之后再创建正式任务。

## 3. Phase 7 和 Phase 8 正式任务

这里沿用现有定义，Phase 7 是两组正式探针，Phase 8 是 A/B/C/D。每个网页任务填写一个种子。用同一仓库提交、数据版本和已验证镜像。

| 任务 | 启动文件 | 网页参数 |
| --- | --- | --- |
| Phase 7 研究点一 | `openi_web.py` | `run_id=p7-probe1-s20261124`、`stage=probe1`、`seed=20261124` |
| Phase 7 研究点二 | `openi_web.py` | `run_id=p7-probe2-s20261124`、`stage=probe2`、`seed=20261124` |
| Phase 8 A 组 | `openi_web.py` | `run_id=p8-A-s20261124`、`stage=abcd`、`group=A`、`seed=20261124` |

B/C/D 各用一个任务，把 `group` 和 `run_id` 改成对应组。网页若显示参数表，每项分行填名称和值。可选填 `git_commit` 为任务实际读取的完整提交，平台只传代码快照时将标作声明值。启动器同时独立计算规范化源码 SHA-256。

正式网页任务会先用实际分配的设备做一次有界 batch 校准，并记录在 `calibration.json`。校准使用四个 CIFAR 客户端，包含一个本地步、一个多视图 KD 步和验证。候选训练/验证 batch 为 32/64、16/32、8/16、4/8，实测峰值保留显存不超过初始空闲显存的 75% 才采用。分配到不同规格时，A/B/C/D 合并工具会拒绝不同配置或步数的结果。Phase 7 的 Probe 2 会先完成同协议 FedMD checkpoint，再运行频率分析。正式实验结果由下载后的审计确认。

## 4. 产物与恢复

`run.json` 记录 run ID、代码、环境、种子、配置和两份官方数据的身份。包内有 `resolved_config.json`、`splits_manifest.json`、CSV、JSONL、控制台和文件哈希。训练与模型选择只使用私有训练、本地验证和无标签代理。官方 CIFAR-10 测试及 CIFAR-10-C 只在独立的 `core.final_eval` 入口读取。

| 路径 | 内容 |
| --- | --- |
| `training/rounds/<round>/state.pt` | 最近已提交轮的客户端模型、优化器、B/D 原型和损失历史、随机状态和预算 |
| `training/checkpoints/client_*.pt` | 可供后续探针和最终评测使用的最终权重 |
| `probe1/metrics/` | 正式研究点一的汇总表 |
| `probe2/attempts/<id>/` | 研究点二的 CSV、JSONL 和 FFT 拼图 |
| `ARTIFACTS.json` | 归档中每个文件的字节数及 SHA-256 |

恢复归档和校验文件从 OpenI 任务输出下载。下面用独立目录验证和解包，`RECOVERY_ARCHIVE` 为下载的归档路径，sidecar 放在旁边：

```bash
python -m scripts.openi.artifacts verify --archive "$RECOVERY_ARCHIVE" --unpack-to "$OPENI_RUNS/p7-probe2-s20261124"
```

FedMD 和 A/B/C/D 从最后完整轮续训，Probe 1 跳过完成的客户端，Probe 2 可用已完成 bootstrap 权重重跑分析。若用网页新任务恢复，挂载原 CIFAR 数据集及恢复归档，保证 TAR.GZ 与其 `.sha256` 同目录，并保留归档文件。如果平台会自动解包 TAR.GZ，可将该对文件装入一层 ZIP，并让平台只解外层。沿用原任务的 `run_id`、stage、seed、代码和组别，把 `recovery_file` 填成 TAR.GZ 相对 c2net 数据集挂载根的路径。入口会验证哈希再恢复，沿用归档内的批量校准。首次冒烟省略该字段。

已持有解包运行目录时，也可用通用入口。变量均取实际路径：

```bash
python openi_startup.py --stage probe2 --run-id p7-probe2-s20261124 --seed 20261124 \
  --require-v100 --runs-root "$OPENI_RUNS" --data-dir "$CIFAR_ROOT" \
  --sync-dir "$OPENI_SYNC" --calibration "$OPENI_RUNS/p7-probe2-s20261124/calibration.json" --resume
```

Linux 镜像准备脚本为 `scripts/openi/init_linux.sh`，它创建独立 venv。按需选择驱动和 Volta 兼容的 torch/torchvision，使用 `TORCH_VERSION`、`TORCHVISION_VERSION`、`TORCH_WHEEL_INDEX` 显式指定 wheel。依赖、环境报告和 Git 信息不写入现有 Windows 环境。Windows 的绝对 Conda 路径只出现在本地 README 示例内。OpenI 路径通过 SDK 或通用入口参数传入，训练无 Wand/ImageMagick 或 OpenCV 运行依赖。

## 当前核验状态

已在本地 RTX 3050 验证通用 CUDA 入口、归档校验和断点恢复，恢复与原运行的两个最终 checkpoint SHA-256 一致。此前两轮冒烟用时 48.01 秒。网页入口和两种输入挂载形式通过本地测试。OpenI 网页实际仓库、远端数据集、镜像、V100 资源、云端结果回传以及完整正式实验均待首次网页作业核验。
