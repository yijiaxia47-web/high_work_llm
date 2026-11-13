# 高处作业安全识别：四源输入与消融实验

本项目将施工现场原图、目标检测编号图、人员中心场景图和规则初判四类证据交给视觉语言模型，输出人员的高处作业状态、防护装备状态及风险等级。项目支持原图基线、完整四源和单源／多源消融，并分别使用真实标注框与 YOLO 预测框构建证据，以研究输入证据对推理结果的影响。

## 项目结构

```text
.
├── main/                         # 可复用主流程；python -m main 启动
│   ├── __main__.py               # 主入口
│   ├── cli.py                    # 共用命令行参数与输入预检
│   ├── runner.py                 # 运行调度、调用记录与结果保存
│   ├── conditions.py             # full、base、消融条件
│   ├── manifest.py               # 样本清单与路径解析
│   ├── prompting.py              # 按条件组合提示词与图片
│   ├── evidence.py               # 检测框 → 编号图、场景图、规则初判
│   ├── metrics.py                # JSON 校验、评分、一致性与配对统计
│   └── config.py                 # 环境变量与模型配置
├── prompts/                      # 业务提示词；各实验条件共用任务定义
│   ├── system.txt
│   ├── safety_task.txt
│   ├── gold_draft.txt
│   └── gold_draft_system.txt
├── scripts/                      # 实验与数据处理命令
│   ├── run_ablation.py           # 统一实验入口
│   ├── prepare_evidence_from_yolo_labels.py
│   ├── prepare_evidence_from_yolo_predictions.py
│   ├── train_yolo.py
│   ├── draft_gold_labels.py
│   ├── build_review_queue_from_drafts.py
│   ├── fill_local_gold_draft.py
│   ├── finalize_gold_from_review.py
│   ├── attach_gold_to_manifest.py
│   ├── offline_rescore_oracle_ablation.py
│   ├── migrate_experiments.py     # 迁移预检、复制与哈希核验
│   ├── migrate_harness_dataset.py # 独立四类安全带数据集迁移与核验
│   └── run_*.cmd                 # Windows 快捷入口
├── llm/                          # 已有模型调用层与 Provider
├── data/                         # 本地数据；目录约定见下文
│   ├── datasets/work_at_height/   # DST2189：六类，12,712 张原图
│   ├── datasets/harness_wearing/  # 安全带佩戴检测：四类，1,884 张原图
│   ├── evidence/                 # oracle/、oracle_001/、predicted/ 四源证据
│   ├── gold/                     # 人工复核后的参考标签
│   ├── review/legacy/            # 历史 gold.zip 解包副本
│   └── weights/                  # 迁入的检测器权重
├── outputs/                      # 实验、训练、标注草稿结果
│   ├── legacy/                   # 原始实验文件与旧脚本的逐字节归档
│   ├── gold_draft_legacy/        # 路径已更新的草稿与人工复核 CSV
│   ├── migration_experiments/    # 第一批迁移报告与逐文件 SHA-256 清单
│   └── migration_harness_wearing/ # 独立安全带数据集迁移报告
├── tests/                        # 无真实 API 调用的检查
├── requirements.txt              # 主流程与证据处理依赖
├── requirements-yolo.txt         # YOLO 训练／推理依赖
├── .env.example                  # 模型服务配置模板
├── experiments/                  # 迁移后保留的原件，核验通过后可手工删除
└── input_demo/                   # 原始输入目录，保留不动 *
```

`scripts/` 调用 `main/` 的公共逻辑，`main/` 通过 `llm/` 调用模型；新流程不依赖 `experiments/`。后续业务修改放在 `main/` 与 `prompts/`，实验编排和数据工具放在 `scripts/`。

## 环境准备

使用 Python 3.10 或以上版本。以下命令在项目根目录的 Linux／Bash 中执行；Windows 可使用相同的单行 Python 命令，并按本机方式创建、激活虚拟环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

在根目录 `.env` 中选择视觉服务商、填写其 API key、接口地址和视觉模型名。例如已有调用层支持以下变量组合：

| 服务商 | Provider | 配置变量 |
|---|---|---|
| OpenAI 或兼容接口 | `VLM_PROVIDER=openai` | `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_VLM_MODEL` |
| DashScope | `VLM_PROVIDER=dashscope` | `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`DASHSCOPE_VLM_MODEL` |

使用支持图片输入的模型。新入口默认读取根目录 `.env`，也可用 `--env-file /path/to/config.env` 指定；进程中已有的同名环境变量优先。`--temperature` 和 `--max-tokens` 可覆盖环境设置，也支持 `none` 以省略相应请求参数；未配置时分别使用 `0` 和 `1024`。`VLM_REQUEST_STREAM=1` 用于要求 SSE 的接口，`VLM_TRUST_ENV=1` 用于读取系统代理配置。

仅在训练 YOLO 或由预测框生成证据时安装额外依赖：

```bash
python -m pip install -r requirements-yolo.txt
```

## 数据与四源输入

四源输入是以下四个字段，消融参数也使用这些名称：

| 字段 | 内容 | 来源 |
|---|---|---|
| `original_image` | 现场原图 | 数据集 |
| `numbered_visualization` | 带目标框、类别及编号的图片 | 标注框或预测框 |
| `scene_graph` | 人员、目标及空间关联 JSON | 检测框的确定性转换 |
| `rule_result` | 候选高处作业、装备存在等初判 JSON | 场景图上的规则 |

现有场景图使用框包含关系和归一化中心距离关联目标。规则层提供候选证据，挂钩连接与安全带规范使用仍保留 `unclear`，交由图像核查；“检测到安全带”不能直接等同于“规范使用安全带”。

数据目录按以下布局准备；实际图像、YOLO 标签、参考标签和历史结果均为本地文件，受 `.gitignore` 排除，不随 Git 仓库分发。本地迁移已复制主数据集的 12,712 张原图及对应 YOLO 标签，未发现已有检测权重：

```text
data/datasets/work_at_height/
├── data.yaml
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

同一图片和 YOLO 标签使用相同文件主名，例如 `images/valid/a.jpg` 对应 `labels/valid/a.txt`。DST2189 的标注类别顺序固定为 `mewp, person, harness, helmet, scaffolding, ladder`；迁移时保留顺序，避免类别编号错位。

### 独立的安全带佩戴检测数据集

`高空作业安全带佩戴检测` 独立迁入 `data/datasets/harness_wearing/`，与 DST2189 分开存放。共 1,884 对图像／YOLO 标签：train 1,507 对、val 188 对、test 189 对。原有文件名、`val` 目录名和 `labels/labels.txt` 原样保留，另补充自己的 `data.yaml`。

| 类别编号 | 原始类别名称 |
|---|---|
| 0 | `aqd_cdgf` |
| 1 | `aqd_gfsy` |
| 2 | `aqd_wpd` |
| 3 | `aqd_wzqpd` |

```text
data/datasets/harness_wearing/
├── data.yaml                 # nc: 4，保留上述类别顺序
├── images/{train,val,test}/
└── labels/
    ├── labels.txt            # 原始类别定义文件
    └── {train,val,test}/
```

训练时显式选择本数据集配置与独立运行目录：

```bash
python scripts/train_yolo.py --data data/datasets/harness_wearing/data.yaml --model yolo11s.pt --project outputs/yolo/harness_wearing --name yolo11s_harness_wearing

# 仅检查这份数据的原图基线输入，不调用 API
python scripts/run_ablation.py --image-dir data/datasets/harness_wearing/images/val --mode base --outdir outputs/harness_wearing/check_base --limit 20 --dry-run
```

当前四源证据生成器使用 DST2189 的六类语义；标注框入口会检查 `data.yaml` 的类别及顺序，并拒绝这份四类数据，防止误读编号。本数据集暂无四源证据或人工推理参考标签；现有 `data/evidence/oracle/`、`oracle_001/` 和 `data/gold/` 均属于 DST2189。训练及实验时分别选择对应数据集的配置、输入和输出目录。

本次迁移保留 `input_demo/originals/高空作业安全带佩戴检测/` 原件。全部 3,769 个原文件（1,884 张图像、1,884 份标签及 1 份类别定义）复制后均比较 SHA-256；报告和逐文件清单位于 `outputs/migration_harness_wearing/`。可执行：

```bash
python scripts/migrate_harness_dataset.py             # 只读预检
python scripts/migrate_harness_dataset.py --apply     # 复制并逐文件校验，遇到内容冲突时报错
python scripts/migrate_harness_dataset.py --verify    # 根据保存的清单重新核验
```

### 由标注框准备证据（oracle）

```bash
python scripts/prepare_evidence_from_yolo_labels.py --dataset-root data/datasets/work_at_height --split valid --outdir data/evidence/oracle --limit 20 --min-classes 2
```

生成 `manifest.json`、`numbered_visualization/`、`scene_graph/` 和 `rule_result/`。默认选择包含人员且至少有两类目标的样本；正式实验应固定清单，记录这一筛选条件。oracle 使用真实标注框，适合隔离证据组织与推理机制的影响。

### 由 YOLO 预测框准备证据（predicted）

```bash
python scripts/train_yolo.py --data data/datasets/work_at_height/data.yaml --model yolo11s.pt --epochs 100 --imgsz 640 --batch 8 --device 0 --project outputs/yolo --name yolo11s_rebuild
python scripts/prepare_evidence_from_yolo_predictions.py --weights outputs/yolo/yolo11s_rebuild/weights/best.pt --image-dir data/datasets/work_at_height/images/valid --outdir data/evidence/predicted --limit 20 --conf 0.25 --device 0
```

新 `data.yaml` 模板的 `path: .` 由训练脚本按 YAML 文件所在目录解析，`val` 对应 `images/valid`；按上述目录迁入数据即可使用。训练脚本会在输出目录保存 `dataset_resolved.yaml`，同名训练目录已存在时使用 `yolo11s_rebuild2` 等新名称，后续 `--weights` 相应调整。无 GPU 时用 `--device cpu`，显存不足时降低 `--batch`。已有迁移权重可直接用 `--weights data/weights/best.pt`。

预测框用于评估端到端性能，与 oracle 结果分开报告。两个准备脚本的样本筛选不同，比较两种设置前应按样本 ID 对齐，不能仅凭相同 `--limit` 认为使用了同一批图片。

### Manifest 格式

Manifest 是 JSON 数组，或包含 `samples` 数组的对象。路径相对于 manifest 所在目录解析，也支持绝对路径。下例假定文件位于 `data/evidence/oracle/manifest.json`：

```json
[
  {
    "id": "a",
    "original_image": "../../datasets/work_at_height/images/valid/a.jpg",
    "numbered_visualization": "numbered_visualization/a.png",
    "scene_graph": "scene_graph/a.json",
    "rule_result": "rule_result/a.json",
    "gold": "../../gold/a.json"
  }
]
```

`gold` 可省略；只要求所选实验条件实际使用的输入源存在。人员级评分按 `person_id` 对齐，人工标签中的人员编号应与证据一致；原图基线没有编号证据，仍需检查模型输出与参考标签的人员对应关系，当前评分器没有基于框的人员匹配功能。

## 运行主流程与消融

先用 `--dry-run` 检查清单和输入，不调用模型，也不需要 API key：

```bash
python -m main --manifest data/evidence/oracle/manifest.json --outdir outputs/check_full --dry-run
```

数据与 API 配置就绪后运行完整四源方案：

```bash
python -m main --manifest data/evidence/oracle/manifest.json --outdir outputs/full --repeats 3 --temperature 0
```

`python -m main` 默认运行 `full`，`scripts/run_ablation.py` 默认运行 `leave_one_out`。统一实验入口通过 `--mode` 选择条件：

| `--mode` | 运行内容 | 条件数 |
|---|---|---:|
| `full` | 完整四源 | 1 |
| `base` | 仅原图，使用与 full 相同的任务提示词和评分逻辑 | 1 |
| `ablation` | 移除 `--drop` 指定的一个或多个输入源 | 1 |
| `leave_one_out` | full、四个逐源移除条件及 base | 6 |
| `all_subsets` | 四个输入源的全部非空组合 | 15 |

```bash
# 完整四源
python scripts/run_ablation.py --manifest data/evidence/oracle/manifest.json --mode full --outdir outputs/ablation_full

# 原图基线：复用相同 manifest，便于与 full 配对比较
python scripts/run_ablation.py --manifest data/evidence/oracle/manifest.json --mode base --outdir outputs/base

# 只有原图时也可以直接运行 base
python scripts/run_ablation.py --image-dir data/datasets/work_at_height/images/valid --mode base --outdir outputs/base_images --limit 20

# 消融一个输入源
python scripts/run_ablation.py --manifest data/evidence/oracle/manifest.json --mode ablation --drop rule_result --outdir outputs/minus_rule

# 同时消融多个输入源
python scripts/run_ablation.py --manifest data/evidence/oracle/manifest.json --mode ablation --drop scene_graph rule_result --outdir outputs/minus_graph_rule

# full + 四个逐源消融 + base，共六个条件
python scripts/run_ablation.py --manifest data/evidence/oracle/manifest.json --mode leave_one_out --outdir outputs/oracle_leave_one_out --repeats 3 --temperature 0

# 全部十五种非空组合，包括纯文本条件
python scripts/run_ablation.py --manifest data/evidence/oracle/manifest.json --mode all_subsets --outdir outputs/oracle_all_subsets --repeats 3
```

`--drop` 仅用于 `ablation`，至少保留一个输入源；`--image-dir` 仅用于 `base`，不能和 `--manifest` 同时使用。图片目录会递归扫描，应指向原图所在目录，避免将编号图混入基线；可用 `--gold-dir data/gold` 为目录输入挂接同名参考标签。

消融仅改变送入模型的输入源，不重新计算其余证据：例如移除场景图后，仍可保留之前由场景图生成的规则初判。因此这里评估的是输入源消融，不是删除整个上游算法模块后的重新训练。

运行预测框实验时，将清单替换为 `data/evidence/predicted/manifest.json`，并使用新的输出目录。调用次数为“样本数 × 条件数 × repeats”；20 张图、6 个条件、重复 3 次共 360 次调用。已有关键结果文件的输出目录不会被覆盖，每次实验使用新的 `--outdir`。

可用 `python -m main --help` 或 `python scripts/run_ablation.py --help` 查看完整参数。业务提示词集中在 `prompts/`，支持通过 `--system-prompt-file` 和 `--task-prompt-file` 指定文件；比较不同条件时保持提示词、模型、采样参数和样本清单一致。

## 人工参考标签

每个样本对应 `data/gold/<sample_id>.json`，顶层为 `overall_risk_level` 和 `persons`；人员条目包含 `person_id` 及六个状态字段。模型生成的草稿必须经过人工检查后才能作为参考标签。

```bash
# 生成模型辅助草稿与待复核 CSV，会调用 API
python scripts/draft_gold_labels.py --manifest data/evidence/oracle/manifest.json --outdir outputs/gold_draft --limit 20

# 人工编辑 outputs/gold_draft/gold_review_queue.csv 的 human_ 列，
# 并将逐行 review_status 改为 checked 后，导出参考标签
python scripts/finalize_gold_from_review.py --csv outputs/gold_draft/gold_review_queue.csv --outdir data/gold

# 将参考标签挂接至清单
python scripts/attach_gold_to_manifest.py --manifest data/evidence/oracle/manifest.json --gold-dir data/gold --out data/evidence/oracle/manifest_with_gold.json

# 使用人工参考标签运行完整消融与评分
python scripts/run_ablation.py --manifest data/evidence/oracle/manifest_with_gold.json --mode leave_one_out --outdir outputs/oracle_with_gold --repeats 3 --temperature 0
```

没有 API 时，也可以直接根据本地场景图／规则创建复核 CSV；人工核查流程相同，将上述导出命令中的 CSV 改为此文件即可：

```bash
python scripts/fill_local_gold_draft.py --manifest data/evidence/oracle/manifest.json --out outputs/gold_draft/gold_review_queue_local.csv
```

`build_review_queue_from_drafts.py` 可从已有 `draft_json/` 重建 CSV；`fill_local_gold_draft.py` 也可通过 `--csv` 读取已有复核队列。它们只辅助人工复核，不能替代人工标注。`offline_rescore_oracle_ablation.py --run <历史运行目录> --out <新目录> --gold-dir data/gold` 用于检查历史输出中的字段／枚举别名修复，修复后结果应与原始评分分开保存和报告。

## 主要结果与指标

本地历史 `raw_results.jsonl`、`summary.json`、比较表及复评分结果已逐字节迁入 `outputs/legacy/<原目录名>/`。历史配置和评分口径保持原样，不将它们视为当前主流程的新实验结果，也不在此据此填写“full 优于 base”等结论。当前未发现 YOLO 训练评估文件或检测权重。

迁入 `data/gold/` 的 20 个标签包含 36 个人员条目：`gold_review_queue_local.csv` 的全部 36 行均标为 `checked`，用新导出脚本重建的内容与历史 `gold/`、`gold.zip` 一致，且人员编号与原证据一致。复核 CSV 工作副本位于 `outputs/gold_draft_legacy/`，原编码、原路径版本及压缩包保留在 `outputs/legacy/gold_draft_001/`。原始队列和 API 草稿队列仍为 `pending`，迁移保留其状态。

正式运行输出如下：

| 文件 | 内容 |
|---|---|
| `config.json` | 实际模型与参数、提示词及哈希、样本清单、输入源组合 |
| `manifest.json` | 本次实际选用的样本及其输入路径快照 |
| `raw_results.jsonl` | 每次调用的原始输出、解析结果、错误、耗时、结构校验与参考标签评分 |
| `summary.json` | 各条件的解析率、结构合法率、一致性、延迟、参考标签指标及配对差异 |

`--dry-run` 只生成配置、清单快照和 `dry_run.json` 请求预览，不产生模型响应或评分。

没有 `gold` 时，只能报告解析率、结构合法率、重复调用一致性和延迟。有人工参考标签时，脚本计算完全匹配率与逐字段准确率；人员状态先在单张图内按人员编号计分，再对调用结果取平均。配对差异先在每张图内平均重复调用，再按图像进行 bootstrap；`mean_full_minus_ablated` 为正表示该字段上 full 的准确率更高。宏平均 F1、检测 mAP 和人工错误分类不由当前主流程自动计算。

评分采用严格结构要求：任一结构校验失败，该次预测的全部字段准确率记为 0；API 失败与无效调用仍保留在相应分母中。无效输出不视为重复一致的有效证据，单次调用没有重复一致性指标。不同实验保留 oracle／predicted 设置、样本筛选条件及 gold 来源，避免将教师输出一致率当作真实安全识别准确率。

## 本地无损迁移与后续清理

迁移只复制文件，不删除原件，不覆盖内容不同的目标。两批 oracle 证据直接保留原有编号图、场景图、规则和样本编号，仅更新工作 manifest 的路径，避免重新生成导致历史实验样本发生变化。新路径对应关系如下：

| 原位置 | 新位置 |
|---|---|
| `input_demo/originals/Work at Height Safety Datase(DST2189)/images/`、`labels/` | `data/datasets/work_at_height/images/`、`labels/` |
| `experiments/runs/oracle_evidence_002/` | `data/evidence/oracle/`（20 个样本，含 gold 清单） |
| `experiments/runs/oracle_evidence_001/` | `data/evidence/oracle_001/`（2 个样本，单独保留） |
| `experiments/runs/gold_draft_001/gold/` | `data/gold/`（已通过复核记录一致性检查） |
| `experiments/runs/gold_draft_001/gold.zip` | 原包归档至 `outputs/legacy/gold_draft_001/gold.zip`，解包副本放入 `data/review/legacy/gold/` |
| `experiments/runs/gold_draft_001/` 的 CSV、`draft_json/` | `outputs/gold_draft_legacy/`（更新图片路径，CSV 统一 UTF-8 BOM，保留人工字段与复核状态） |
| `experiments/runs/` 的全部文件，包括历史清单、配置、原始响应、日志、独立比较表 | `outputs/legacy/<原相对路径>/`（原始字节不变） |
| `experiments/` 顶层脚本、示例清单和缓存 | `outputs/legacy/scripts/`（原件存档；日常使用 `scripts/` 与 `main/`） |

`input_demo/` 中的 VOC 标注、原始压缩包以及两套数据集的原件继续保留原位，不在本次清理范围。安全带佩戴检测数据集的独立迁移见前文；DST2189 的六类名称顺序及现有 `data.yaml` 模板保持不变，未迁移或改写 `.env`。本地没有 predicted 证据和 YOLO 权重，后续按前文命令生成。

迁移工具以当前本地这批目录为对象；缺少源文件、复核记录不一致或目标内容冲突时会报错，不自动猜测处理。以下命令在项目根目录执行，支持 Windows 和 Linux：

```bash
# 只读预检：核对复核记录、计算哈希并检查全部目标冲突
python scripts/migrate_experiments.py

# 执行复制并自动校验；重复执行可复用内容相同的文件
python scripts/migrate_experiments.py --apply

# 删除 experiments 前再次核验；删除后也可用此命令检查迁入副本
python scripts/migrate_experiments.py --verify

# 只用迁入后的文件检查完整六条件，不调用 API
python scripts/run_ablation.py --manifest data/evidence/oracle/manifest_with_gold.json --mode leave_one_out --outdir outputs/check_migrated --dry-run
```

核验结果记录在 `outputs/migration_experiments/report.json` 与 `files.csv`；后者逐项列出原路径、新路径、字节数和 SHA-256。`--verify` 同时检查原目录是否新增或修改文件，以及工作清单是否仍依赖 `experiments/` 或 `input_demo/`。日后修改工作文件会使历史哈希核验失败，这是预期行为；如需删除原目录，先核验并保留报告和完整归档。迁入文件和归档均受 `.gitignore` 排除，需要随项目另行备份，不能仅依靠 Git。

`outputs/migration_experiments/validation.json` 记录额外离线检查：禁止读取两个旧目录后，20 个样本的六条件 dry-run 通过；三份复核队列共 92 行的人工字段、复核状态和备注保持一致，20 份草稿的原始响应保持一致；三个 split 的图像与标签主名全部配对。两个实验 `.cmd` 入口也已通过 dry-run，未调用 API。

旧 Python 功能已由 `scripts/` 同名工具和 `main/` 公共模块承接；`four_source_ablation_demo.py` 对应 `scripts/run_ablation.py`，`raw_only_baseline.py` 对应其 `--mode base`。旧参数 `--conditions leave_one_out` 对应新参数 `--mode leave_one_out`。历史脚本保留用于溯源，存档中的绝对路径不修改、不作为运行入口。

Windows 快捷入口为 `scripts/run_oracle_ablation.cmd`、`scripts/run_raw_baseline.cmd` 和 `scripts/run_prepare_oracle_demo.cmd`，使用相对项目位置及根目录 `.env`。前两者会调用 API，可追加 `--dry-run` 预检；证据准备入口默认输出到 `data/evidence/oracle_regenerated/`，保留迁入的证据。运行结果目录已有文件时需另选 `--outdir`。

## 本地验证

```bash
python -m unittest discover -s tests -v
python -m main --help
python scripts/run_ablation.py --help
```

自动检查使用临时数据与模拟响应，不调用真实 API。完整实验、检测器训练及性能评估需要迁入真实数据并配置可用模型服务后执行。
