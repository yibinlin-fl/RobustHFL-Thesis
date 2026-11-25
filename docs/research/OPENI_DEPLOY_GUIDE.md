# OpenI V2 网页托管部署

第一次在 OpenI 网页上传 V2 数据 ZIP、选择私有代码仓库和 PyTorch GPU 镜像，启动 `scripts/openi_entry.py`。仓库默认 `mode: env_check`。无参数提交时，任务只校验环境、官方数据、划分和实测小批吞吐。环境验收和预算确认后，在网页参数中选正式模式，或在专用分支提交修改后的 YAML 再创建新任务。平台通过 c2net 传入代码、数据、模型和输出位置。 [OpenI 代码路径文档](https://openi.pcl.ac.cn/docs/index.html#/cloudbrain/codepath) · [OpenI 官方指南](https://openi.pcl.ac.cn/guide/index.html) · [c2net 发布文档](https://pypi.org/project/c2net/)

## 数据集与代码仓库

| 数据集字段 | 建议 |
| --- | --- |
| 英文名 | `robusthfl-cifar-official-v2` |
| 中文名 | `RobustHFL CIFAR 官方训练协议 V2` |
| 可见性 | 私有，限课题成员。以本人或有写权限的课题组织为所属空间 |
| 文件 | `robusthfl-cifar-v2.zip`，包含两份官方原始 TAR.GZ、`input_manifest.json` 与 `reference_split.json` |
| 内容简介 | CIFAR-10 私有训练、本地留出、同域代理。CIFAR-100 用作跨域无标签代理 |
| 验收 | 等平台显示上传处理完成，记录实际数据集名称和 ID。新建任务时能搜索并选到该数据集，再核对文件名与大小 |

数据归档带官方原包结构。训练径路按官方 MD5 核验并仅解析训练成员。参考切分含官方测试索引保留项，没有测试像素或标签的导出。不同种子由运行时重新生成冻结清单。CIFAR-10-C 单独保存，研究阶段无需上传。上传包的 SHA-256 sidecar 保存在本地。 [官方 CIFAR 数据](https://cave.cs.toronto.edu/kriz/cifar.html) · [数据归档 MD5](https://zenodo.org/records/10089977)

GitHub 负责私有版本管理，`main` 用于稳定源码，`openi-v2` 用于 OpenI 任务。发布分支的 SHA 可以用 `git rev-parse openi-v2` 查看。`origin` 指向 GitHub 私有仓库。OpenI 官方指南确认可以向 OpenI 项目推送 Git 分支，再在计算任务中选择该项目。GitHub 私有仓库的直接授权导入还没有为本项目验收。 [OpenI 官方 Git 上传说明](https://openi.pcl.ac.cn/guide/index.html)

如果 OpenI 账号界面明确提供私有 GitHub 仓库授权迁移，可在平台授权界面仅授权该仓库，选定目标为私有，完成后核对 `openi-v2` 的 SHA。以界面实际提供的权限为准。也可直接在 OpenI 创建空的私有项目，复制该项目的 HTTPS 克隆地址，在本地用独立的 `openi` 远端同步同一个分支：

```powershell
$openiRemote = Read-Host '粘贴私有 OpenI 项目的 HTTPS 克隆地址'
git remote add openi $openiRemote
git push openi openi-v2
git ls-remote --heads openi refs/heads/openi-v2
```

如果已经有名为 `openi` 的远端，先核对其目标再推送。按本机 Git 凭据提示认证。确认远端返回的 SHA 与本地一致，再到 OpenI 新建任务。训练容器只使用平台提供的项目代码。提交范围为源码、配置、测试、依赖和部署文档。数据、checkpoint、本地结果和导师材料留在本地。

### 第三方参考

`third_party/` 中的 RAHFL、AugHFL、fl_data_quality 是独立参考仓库。训练使用本项目的 `core/`、`baselines/` 和 `methods/`。三个固定参考提交的 Git 树里均未找到独立的 LICENSE、COPYING 或 NOTICE 文件，因此源码发布只保留引用与固定提交获取脚本。需要核对原参考时可在本地使用 `scripts/clone_references.ps1`。脚本会检查取回的提交，OpenI 运行无需安装这些参考仓库。 [RAHFL](https://github.com/FangXiuwen/RAHFL/tree/ba21203ca200ecdaa765539ba6aeacd05d2cbc99) · [AugHFL](https://github.com/FangXiuwen/AugHFL/tree/ac4c64fbef0c6ccefae3bf796781e5097b590437) · [fl_data_quality](https://github.com/fardatalab/fl_data_quality/tree/c9a5d584ce0805a5fcda70707f8ca640aabfc9a8)

## 新建计算任务

实际平台按钮和镜像标签以你打开的账户页面为准。下表针对用户提供的 A100 规格填写，尚无云端实际验收记录。

| 网页字段 | 第一次建议填写 |
| --- | --- |
| 计算资源 | 英伟达 GPU |
| Internet 访问 | 关闭。输入数据使用已关联挂载，依赖预装在镜像中。按平台页面的 c2net 服务要求调整 |
| GPU 资源规格 | `1 × NVIDIA A100 40 GB`，用户截图为 8 CPU / 50 GB RAM。实际资源以后续 `nvidia-smi` 为准。兼容的 V100 也可做环境检查 |
| 任务名称 | `robusthfl-v2-env-01` |
| 任务描述 | `V2 环境、数据隔离与吞吐验收，单卡小批量，无科研结论` |
| 镜像 | 选择提供 Python 3.10+、CUDA 版 PyTorch、匹配 torchvision 和可用 c2net 的 PyTorch 镜像。需 NumPy、SciPy、PyYAML、Pillow、scikit-learn、matplotlib |
| 数据集 | 选择前述已上传的 V2 数据集 |
| 模型 | 留空，基础模型由 CE 从头训练。恢复可按下节关联结果 |
| 项目 | 你提交 V2 源码的私有项目 |
| 代码分支 | 实际已提交的分支，建议 `openi-v2` |
| 启动文件 | `scripts/openi_entry.py` |
| 可选运行参数 | 第一次留空。默认 `env_check`，自动运行 ID，不用填 GPU 编号或容器绝对路径 |
| 时限／积分 | 如果页面允许，首次选择 15 分钟短任务并核对页面显示的实际积分。代码内自检上限 170 秒，从开始校验数据与基准算起，另给镜像启动、挂载及回传留时间 |

选择支持仓库相对路径的 Python 启动文件字段即可直接执行。若当前页面只列出根目录文件，选择源码已附带的 `openi_managed.py`。它调用同一入口与配置。实际平台是否显示嵌套入口尚待网页验收。

镜像选型看任务实际日志。检查 torch CUDA build、torchvision GPU 算子、FFT 反向、驱动、显存和磁盘。无需匹配本地 2.8.0+cu126。若现有镜像缺依赖，更换满足上述条件的预装镜像。Linux 环境的依赖列表在 `requirements-openi.txt`，`scripts/openi/init_linux.sh` 是已有环境构建辅助入口。 [PyTorch 官方匹配版本](https://pytorch.org/get-started/previous-versions/)

## 首次 A100 结果验收

从 OpenI 任务详情的输出或结果下载入口，下载完整的 `RobustHFL-V2/<env_check-run-id>/`，一并留存任务编号和日志。以下四项全部通过后才选择正式任务：

| 输出 | 核对项 |
| --- | --- |
| `environment.json` | 实际 `nvidia_smi` 与 PyTorch GPU 信息均为 A100，CUDA 和 torchvision 内核检查通过，torch/torchvision/CUDA 版本与镜像匹配，c2net 入口成功，挂载及解析的数据位置正确 |
| `budget_estimate.json` | 四客户端实图前向输出形状为 `[32, 10]`（默认 batch），训练来源读取速率、样本批准备速率、CE/KD 小批步耗时均有实测值，`by_mode` 含各独立模式估算 |
| `run_manifest.json` | `env_check_only=true`、`execution_status=complete`、`formal_optimizer_updates=0`、`checkpoint_count=0`，Git SHA 或源码摘要、数据与切分匹配 |
| `file_hashes.json` | 校验全部下载文件。记录任务日志中的 `C2NET upload_output() returned`，再以实际可下载、哈希匹配为验收依据 |

吞吐测量使用临时模型，每个客户端各做 1 次本地 CE 与 1 次 KD 更新，共 8 次基准更新，另外独立执行真实 CIFAR 输入的最小前向。该路径在正式训练之前直接退出，从未生成正式 checkpoint。正式模式须新建 run ID。页面以 ZIP 提供代码且 Git 信息缺失时，若可填参数，填 `git_commit=<实际分支提交全 SHA>`，便于验收命令核对。

```powershell
python scripts/openi_entry.py --verify-output DOWNLOAD_ENV_CHECK_DIRECTORY
python -m scripts.openi.transfer_v2 plan-a100 --run-dir DOWNLOAD_ENV_CHECK_DIRECTORY --max-seconds 7200 --max-points 18 --points-per-hour 9 --reserve-seconds 900
```

把费率和任务硬限制换成网页实际值。若云端报告含实际提交，可追加 `--commit FULL_GIT_SHA`。计划器要求下载件显示受托管 A100，验证数据切分和 SHA-256，并输出能在一项任务里完成的 `recommended_single_mode`。推荐顺序为研究一、异质损坏、频率，只有整项预算合适才推荐。若没有可完整运行的模式，先调整已授权时限／积分，再重新估算。15 分钟是默认启动与回传估算预留，若实际耗时更大需调高。默认 2 小时和 18 积分、9 积分/小时对应训练估算窗口 105 分钟。已有 RTX 3050 文件和 mock 结果不能用于 A100 排期。

## 正式模式与 checkpoint 输入

在网页看到 `env_check` 通过且首次下载核验成功后，先比较 `budget_estimate.json` 内的预计更新量、实测步耗时及费用估计，再提交一个正式模式。截图约 9 积分/小时是费用估算输入，实际计费以网页为准。默认预算上限为该配置 18 积分或 7200 秒取小值，任务排队、输入准备、归档回传及退出清理开销需另留余量。预算过紧先选单个模式，保持模型质量门槛。

| 模式 | 配置、数据与 checkpoint 输入 | 新任务未恢复时的训练 |
| --- | --- | --- |
| `env_check` | 同一 V2 配置与 CIFAR 上传包，无 checkpoint | 仅临时吞吐基准 |
| `work1_precheck` | 同一 V2 配置与 CIFAR 上传包。复用时需要基础训练完整目录、warm-up、analysis 及相邻轮状态 | 从头训练四个 2×2 基础模型。Mixed CE 使用基础模型，另训练 Clean CE、高 Loss 降权、随机降权 |
| `work2_heterogeneity` | 同一 V2 配置与 CIFAR 上传包。可恢复基础训练和已提交的各场景训练状态 | 仍从头执行四个 2×2 基础模型及质量门槛，再独立执行五场景 × Local-CE 和 Standard KD。场景模型不由 Work 1 终态初始化 |
| `work2_frequency` | 同一 V2 配置与 CIFAR 上传包，默认两个公共代理域。复用时需要兼容的基础训练完整目录、已通过的质量门槛和四个最终客户端 checkpoint | 先重新训练并验收 2×2 基础模型，再作配对 ΔR。无需研究一或异质场景的模型 |
| `precheck_all` | 同一 V2 配置、CIFAR 上传包和可选兼容恢复包 | 按优先级复用一份 2×2 基础模型。本次只提交 `env_check` |

若页面参数表可用，填写参数名 `mode`、值如 `work1_precheck`。否则只修改 `configs/openi_precheck.yaml` 的 `mode`，提交后在网页选择该分支。正式模式仍为同一个种子。各阶段模型质量与预算条件都通过后才汇总。数据、代码、运行时或科学配置变化后需新运行 ID。本设置不启动最终 A/B/C/D。

平台以代码 ZIP 提供源码时可能没有 Git 元数据。可通过网页 `git_commit` 参数提供实际选中提交的完整 SHA。入口区分 checkout 实测与声明值，并始终记录源码摘要。

预算估计基于一次四客户端 CE、一次四客户端 KD 与验证的小批实测，带可配置裕量。正式协议执行全配置更新数。模型 gate 使用干净留出准确率、验证多数类基线、预测类别数及最大单类占比。门槛是分析资格门槛，不是科学 GO 判定。Formal OOM 后更改 batch 属于新协议，保留旧失败状态。默认仅在退出尝试传全部结果，按需开启阶段或间隔回传。每个训练臂在磁盘上只保留最近一轮状态和必要的基础模型快照，旧轮 CSV/JSONL 留存。

本地 RTX 3050 正式配置小批校准给出的外推如下。每个独立模式含其基础训练。按用户提供的 9 积分/小时换算，仅为了估预算，未产生平台计费。首次 A100 作业会按实测写出新的 `by_mode` 估计。预算不足时保留能完成的阶段前缀和 checkpoint，并在 `invocation_budget.json` 记录延期阶段，不将部分完成标为全部完成。

| 模式 | 估计更新上界 | 本地外推 | 估计积分 |
| --- | ---: | ---: | ---: |
| `work1_precheck` | 16,800 | 1.46 小时 | 13.09 |
| `work2_heterogeneity` | 44,640 | 2.69 小时 | 24.19 |
| `work2_frequency` | 4,272 | 0.29 小时 | 2.62 |
| `precheck_all` | 58,032 | 3.95 小时 | 35.59 |

合并模式的 checkpoint 估计空间约 4.07 GB，另需为数据、日志、平台输出和下载留余量。显存、挂载读取速度及可用输出容量以运行报告为准。

## 日志、回传和继续运行

打开任务详情的运行日志区域。完成后在任务提供的输出或结果下载区域获取平台输出树下 `RobustHFL-V2/<run_id>/`。如果实际标签不同，以页面显示为准。

- `run_manifest.json`、`environment.json`、`budget_estimate.json`、`execution.log` 记录身份、资源、预算和日志。
- `file_hashes.json` 记录输出文件字节数和 SHA-256。终端中的 `C2NET upload_output() returned` 以及 `upload_status.json` 可核对调用结果。SDK 正常返回后仍需下载验证。下载目录应与已有本地 `results` 隔离。
- `checkpoints/base_2x2/` 包含 Warm-up、Analysis 状态及最终客户端权重。其他训练臂在各阶段训练子目录。 `work1/`、`work2_heterogeneity/`、`work2_frequency/` 存 CSV，`figures/` 存分析 PNG，`reports/` 存研究摘要。
- 任务错误会保留 `failure.json` 和 traceback。在同一个持久输出目录内可用原 run ID 加 `--resume`。另起 OpenI 任务时按下面步骤恢复。

### 跨任务持久化与恢复

1. 从上一任务的结果页下载完整运行目录到独立位置。确认 `file_hashes.json` 上传件和可恢复 checkpoint 都能实际下载。环境验收任务不产 checkpoint。保存下载件 SHA-256 和原始 run ID。
2. 在本地生成可重新挂载的恢复 ZIP。该包保留基础模型相邻轮、模型与优化器、分片日志和阶段收据。不要用只含单个 `.pt` 的文件替代它。

   ```powershell
   python -m scripts.openi.transfer_v2 pack-recovery --run-dir DOWNLOAD_RESEARCH_RUN_DIRECTORY --out openi-transfer/recovery-ORIGINAL_RUN_ID.zip
   ```

3. 将该 ZIP 持久化到有权限的私有 OpenI 模型附件或私有数据集。查看页面对实际包大小和文件类型的限制。如果任务提供模型附件，将包关联到新任务并设 `restore_mount=model`，入口从 `pretrain_model_path` 取包。若使用数据集则设 `restore_mount=dataset`，并保证新任务也能访问 CIFAR，必要时填写 `dataset_subdir` 显式选 CIFAR。选择压缩包本身或平台解包后包含 `file_hashes.json` 的目录。容器之间没有隐式共享的上次临时输出。
4. 同一提交、同一科学配置、种子、数据和兼容运行环境下，填写 `run_id=原任务ID`、`mode=要完成的模式`、`restore_mount=model`（或 `dataset`）、`restore_subdir=相对该挂载根的ZIP或目录路径`。恢复入口自动开启 resume，逐文件验证后复制到新任务输出空间。下一任务已有目标运行目录时拒绝覆盖。使用 Git ZIP 的任务若上次指定了 `git_commit`，续跑保持一致。
5. 校验成功的阶段会被跳过，轮级训练从最后有效提交续跑。未完成的干预／频率分析会重算该阶段。若无法重新挂载私有恢复 ZIP、下载哈希失败，或两次任务环境不兼容，保持单个研究模式在一个任务内完整结束。

默认以提交后正常退出或可捕获失败时的已审计回传为持久化点。容器被强制销毁时不能保证回传。可选阶段或间隔发布只有下载件通过哈希核验后才能算恢复点。预算选型需为结果回传预留时间。

本地用以下命令复核运行或再画图。OpenI 创建任务本身无需执行这些命令。

```powershell
python -m pytest -q tests
python scripts/openi_entry.py --local --mode env_check --data-dir data --output-dir openi-local --device cpu
python scripts/openi_entry.py --local --mode env_check --data-dir data --output-dir openi-local --device cuda
python scripts/openi_entry.py --verify-output DOWNLOAD_RUN_DIRECTORY
python -m scripts.openi.precheck_figures --run-dir DOWNLOAD_RUN_DIRECTORY --out REGENERATED_FIGURES_DIRECTORY
```

V2 与既有 RTX 3050 冒烟分开保存。OpenI 镜像、平台托管运行及实际远端回传等待首次网页作业核验。
