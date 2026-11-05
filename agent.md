# General_SERS_Analysis：研究背景与协作约定

更新日期：2026-11-04。本文汇总用户已明确的目标、数据规范与当前工程约定，供后续开发接续。实现状态以实际代码和检查结果为准；研究设想不等于已验证结论。

## 1. 研究定位

- 从儿茶酚胺类神经递质的 SERS 检测出发，研究可以推广至其他结构相似物的光谱化学分析方法。
- 用户已有实验基础：NTA–Fe 修饰颗粒结合衍生化，可检测较低浓度范围的儿茶酚胺类神经递质。具体实验条件和性能数值必须回查实验记录，不能自行补写。
- 核心关注算法和数据；算力不是本项目的创新重心。
- 弱化临床应用导向。尿液检测只是一个可选应用实例，是否成功不决定核心方法创新是否成立。
- 曾讨论 Nature Machine Intelligence、JACS、Angew 等期刊，但期刊目标不应成为夸大普适性或性能的依据。

## 2. 两条长期研究主线

### 2.1 自动构建光谱分析流程

从已有工具库中选择、组合子模块，形成适合具体化学任务的分析流程。最终希望学习“怎样选择和连接算法”，并能说明选择依据，而非仅比较若干人为固定的模型。

待研究的问题包括：化学任务信息如何指导流程选择；流程结构如何控制误差传播；补充什么独立数据最能改善结果；选择理由能否通过干预实验检验。这些是候选贡献，尚未证明。

模块组合拟用图表示：节点是工具，边记录传递的数据和路径启用条件。自动图搜索、训练策略、解释方法尚未定案。待分析物结构是否作为输入、是否采用 GNN（图神经网络）提取结构特征，均属后续研究问题；当前不实现深度学习。

仅有儿茶酚胺一个实验任务不足以训练或验证通用流程设计器，因此需要广泛收集有原始光谱、标签和任务说明的独立数据集，优先高水平论文的公开原始数据。论文作者原算法可作为参考基线，不应把它当作“最优流程”的真值。

### 2.2 数据需求分析

希望回答新任务需要采集多少独立样本，以及提前估计这个数量需要哪些信息。最终关心的是建议采集量及其依据，而不仅是给定样本量后预测误差。

“足够好”的性能如何客观评价、是否存在可以复用的评价准则，仍需研究，不可宣称已经解决。重复采集的谱数与独立实验样本数必须区分。

两条主线暂放同一仓库，预留独立子包；当前仅构建共享工具基础，不开展流程搜索和数据量分析。

## 3. 路径与已有资料

- 目标代码仓库：`E:\myCode\Python\General_SERS_Analysis`。
- 现有数据库：`E:\Research\国自然\data\General_SERS_Analysis_Database`。
- 研究讨论与实验记录：`E:\Research\莙政\讨论`，尤其要参考实验记录。
- 最近发表论文所在目录：`E:\Research\莙政\纸基底CA\Manuscripts\publication`。
- 当前开发暂存目录：`E:\Research\国自然\tmp\general_sers_scaffold`。它是部署前的暂存副本，不是第二个正式仓库。

本文件只记录资料位置，不表示已重新核验上述资料的全部内容。后续引用实验结论或论文细节时，应阅读对应原始文件。

## 4. 数据库约定

### 总任务

总任务是 Classification 与 Quantification（数值预测亦可称 Regression）。当前 content.csv 的具体 task 值保留用户给定的三种：

- `Classification`
- `Single-Component Quantification`
- `Multi-Component Quantification`

先分类后定量等内部子任务组合由后续流程设计器决定，不在总任务字段中预先规定。

### content.csv 字段

| 字段 | 含义 |
| --- | --- |
| Data_id | 数据集/任务编号，通常一个任务一个编号 |
| task | 上述三种任务值之一 |
| n_classes | 任务涉及的物质/分子数，不必等于分类器输出类别数 |
| molecules | 按固定顺序以 `_` 分隔的分子名称 |
| quant_unit | 定量单位，如 uM；结构属性回归也可以是 carbons；分类填 null |
| quant_datatype | 定量标签数据类型，如 int、double；分类填 null |
| n_group | 样本组数量；例如同一浓度组合的 50 条重复谱可以构成一组 |
| n_spectra | 总光谱条数 |
| data_source | 数据来源，应尽可能保留论文 DOI、仓库地址和出处 |
| modality | 用户已要求补充的测量类型，如 Raman、direct-SERS、indirect-SERS |

modality 的完整受控词表、字段在现有索引中的实际完整度尚须检查，不能默认已经补齐。

### 数据布局

```text
General_SERS_Analysis_Database/
  content.csv
  data/
    <Data_id>/
      <group>/
        <spectrum>.csv
  molecules/
    <molecule structure file>
```

- 每条谱单独保存为 CSV，列名为 `RamanShift`、`intensity`，第一列波数，第二列强度。
- 分类组名按 `Class_<presence1>_<presence2>_..._<presenceN>_<index>` 表达；presence 为 0/1，顺序对应 molecules。
- 定量组名按 `Quan_<value1>_<value2>_..._<valueN>_<index>` 表达；单位、顺序和数据类型对应索引。
- 用户原始组名描述中的分隔符存在歧义，上述是其语义整理；开发读取器前应核对实际文件夹示例，不能直接假定所有历史目录均符合这一拼写。
- molecules 文件名需与索引中的分子名称对应。用户倾向统一使用 SMILES：它是用文本表示原子、键与连接关系的分子结构记法；并不自带实验光谱或全部三维构象。具体文件扩展名、立体化学、盐型/质子化态、命名冲突处理尚未定案。
- 数据集组不天然等同独立实验单位。未来应记录制备批次、重复关系等信息，避免重复谱跨训练/测试导致结果虚高；这些尚未成为当前固定索引字段。

## 5. 当前明确授权的实现范围

用户要求先完成以下基础，之后停下来供检查：

1. 合理目录、`.gitignore`、英文 `README.md`、`requirements.txt`。
2. `data/database` 和 `data/test_data` 空目录。数据库由用户自己迁入，不自动移动原数据库。
3. `logs`、`outputs`、`visualizations` 保存未来运行记录、输出和图形。
4. 节点 Node 和边 Edge 两类基础对象；所有工具继承 Node。
5. 下述经典算法工具。不涉及深度学习。

| 类型 | 用户要求的工具 |
| --- | --- |
| 前处理 | SG 滤波、airPLS 去基线、插值、光谱裁剪、峰值归一化、min-max 归一化、标准化 |
| 分类 | RF、PLS-DA、SVM、XGBoost、Logistic Regression |
| 聚类/可视化 | k-means、t-SNE |
| 回归 | PLSR、RF、SVR、XGBoost、Digital SERS、线性回归 |
| 解混 | CLS、NNLS |
| 聚合/特征提取 | PCA、峰提取、因子分析、均值、中位数、最大值、最小值、光谱作差 |

**初版范围不实现**命令行入口、数据库读取器、流程执行器、自动搜索/Agent、自动数据划分、学习曲线或数据量预测。后续新增的可视化入口见第 11 节；该后续授权覆盖初版中有关 CLI 和可视化读取的限制。目录预留不代表这些功能已完成。用户曾提出的命令行示例 `--task classification --molecules A_B_C --data_dir data/test --output_dir model` 是未来意图，当前不可作为可运行命令。

## 6. 接口与科学含义

- `fit(...)`：从训练数据学习参数并保存状态，返回工具本身。它不必是神经网络预训练；计算训练集均值、拟合回归系数也属于 fit。
- `transform(...)`：使用固定规则或已学状态，得到处理后的光谱、特征或其他表示。
- `predict(...)`：使用已学状态输出类别、数值目标或系数。各工具只提供有科学含义的接口，不强行实现所有动作。
- `Spectra` 保存二维强度矩阵（行是谱、列是波数点）、严格递增的共享波数轴和可选样本编号。需显式对齐波数，不把列索引当作真实波数。
- 各模型超参数由构造参数传入；sklearn/XGBoost 包装器主要使用 `parameters` 字典。
- feature 范围的标准化只在训练集 fit；不能在测试集重新学习统计量。多模块级联未来需要组级验证和折外预测，当前工具库没有自动保证这些流程约束。
- 普通单标签分类与多分子存在/不存在的多标签分类不同；后者使用二进制目标矩阵和支持该模式的工具。
- PLS-DA 输出分数不是校准概率。k-means 簇编号不自动等于化学身份。t-SNE 是嵌入可视化，不是分类器，当前没有新样本 transform。
- CLS/NNLS 需要已知参考组分谱；输出系数与重构结果，不声称发现未知纯谱或直接得到浓度。
- Digital SERS 当前以阈值判定事件、按组统计阳性比例并用占据模型校准；需独立、可比较的检测事件。全阳性饱和不能输出有限估计。其化学适用性、阈值和误差仍需真实实验验证。
- 回归输出保持二维（样本数 × 目标数）；组聚合后应使用组级标签，不能仍沿用逐谱标签。
- Edge 当前是连接描述和结构化条件，不运行任意字符串，也尚无整图调度、循环检测和完整端口检查。

## 7. 当前实现状态与验证纪律

已在暂存目录编写 src 布局、上述工具、英文 README、中文 `docs/tool_library.md` 及合成数据测试。`workflow_design` 与 `data_planning` 仅预留子包。

2026-11-04：依赖安装完成，暂存代码的合成数值/接口测试为 **19 passed**。scikit-learn 1.9.1 对 SVC 的 probability 参数发出一项未来弃用提示，当前测试通过；升级库时应检查概率校准接口。工程文件已部署到目标仓库，共 30 个文件逐一通过 SHA256 副本校验；保留原 `.git`，未移动原数据库。data/database 与 data/test_data 仅含空目录占位文件。

检查应关注训练状态、波数轴、标签形状、多输出行为、参考谱恢复、聚合分组、Digital SERS 饱和与边条件。合成测试通过只证明接口和部分数值行为，不证明实际 SERS 效果或研究普适性。不能编造实验结果、样本量或文献结论。

## 8. 文档与协作偏好

- README 用英文；面向用户的说明默认中文。
- 用户机器学习基础有限：专业术语在出现处解释，避免只堆缩写。数学公式应逐个解释符号，并提供来源定理或研究；自行设定的模型应明确标注。
- 用户希望信息集中，优先维护现有文档，避免不断产生内容重叠的报告。
- 后续规划需先尊重用户已有实验基础；不能以临床验证取代算法和数据的核心问题。
- 在当前授权范围内完成工作后停下供用户检查，不自行扩展下一阶段功能。

## 9. 本地 Python 环境

2026-11-04 已在工程根目录创建 `.venv`，Python 3.12.15，基础解释器来自 `E:\DevTools\Anaconda3\envs\lxt_basic\python.exe`；虚拟环境不继承基础环境的第三方包。已安装 requirements.txt 全部依赖，并以 editable 模式安装本工程。

环境解释器：`E:\myCode\Python\General_SERS_Analysis\.venv\Scripts\python.exe`。requirements 兼容性检查、全部工具模块导入及线性回归数值检查通过。此前 19 项 pytest 测试是在开发检查环境中运行；本次未向新 `.venv` 安装 pytest 开发依赖，也未在其中重跑该测试集。

已安装运行依赖版本：numpy 2.5.4、scipy 1.18.1、scikit-learn 1.9.1、xgboost 3.4.1、pybaselines 1.2.1、joblib 1.6.0。`.venv` 已由 .gitignore 排除，不应提交。编辑器应选择工程 `.venv` 的解释器。

## 10. 数据版本管理约定

用户明确要求 data 不应被 ignore。已删除 .gitignore 中针对 data 的目录排除规则，data/database 与 data/test_data 中的数据文件应允许 Git 跟踪；常规缓存和环境排除规则仍适用。未自动执行 git add 或提交。

## 11. 按 Data_id 绘制组平均光谱

2026-11-04 用户授权在根目录创建 main.py 作为所有任务的主接口，目前仅实现 `visualize` 子命令；不增加训练、流程搜索或数据量分析任务。

- 命令：`.\.venv\Scripts\python.exe main.py visualize --data-id GSAD0004`。
- 数据默认读取 `data/database/content.csv` 与 `data/database/data/<Data_id>/<group>/*.csv`；数据库已由用户迁入，初版空目录描述仅为历史状态。
- 每个 group 中重复谱按原始强度作逐点算术平均，每个 group 一条曲线，不合并不同目录、不做前处理。
- 组内波数轴须相同；递减轴显式反转，其他差异报错，不自动插值。不同 group 可以保留各自波数轴。
- 默认保存 `visualizations/<Data_id>_group_means.png`；`--show` 额外弹窗，`--groups` 选择组，`--database-dir` 和 `--output-dir` 改变输入和输出位置。同 ID 重跑会覆盖同名图片。
- 超过 20 组时绘制全部组均值但省略图例；建议用 --groups 查看有标签的子集。索引计数不符会警告，实际读取数量以文件为准。
- 新增 `src/general_sers_analysis/dataset_visualization.py` 和对应测试；requirements 新增 matplotlib。没有实现其他总任务。

验证结果：新增 4 项 unittest 检查通过；真实 GSAD0004 的 33 条光谱绘制为 11 条组平均谱，图片已人工检查，pip check 通过。matplotlib 3.11.2 已安装。真实 GSAD0005 同组混含 466–2500 与 2628–4000 cm^-1 两个不重叠波段，当前按设计拒绝直接平均，未改动原数据；若要支持该数据，需要先明确分段数据的重复关系与汇总方式。

## 12. GSAD0004 example 工作流（2026-11-05）

用户授权新增 main.py example：使用现有 SpectralCut、Interpolate、AirPLS、MeanAggregator、PLSRRegressor 与 Node/Edge API。可视化采用新增 GroupSpectrumPlot(Node) 适配器，调用已有均值工具后原样传递数据。transform_path 只执行明确指定的无条件默认端口路径，不是通用图执行器；训练在最终边之后显式调用模型 fit。

默认先 cut 600–1800 cm^-1，再按训练数据裁剪公共范围内 2 cm^-1 整数刻度线性插值对齐，不作峰位校准或外推。之后逐谱 airPLS，绘制组平均谱，逐谱训练三输出 PLSR（2 个成分、训练内缩放）。按 group 固定随机划分，8 组训练、3 组测试，测试按组均值计算每组分误差；11 组条件不代表独立制备，不能宣称跨基质验证。默认前两个组分 R2 为负，保留原始预测和指标。

运行命令：`.\.venv\Scripts\python.exe main.py example`；可加 --show。输出 summary.json、test_predictions.csv、workflow.joblib 到 outputs/GSAD0004_example，去基线图到 visualizations/GSAD0004_example_baseline_corrected.png。实际运行和模型重载、分组隔离、三输出与 Node/Edge 类型检查通过；新增 test_example_workflow.py 检查连接执行和外推拒绝。英文用法见 README。原先“不增加训练任务”的限制被本次明确授权覆盖，其他 AutoML、通用图调度和数据量分析仍未实现。


## 13. 通用可视化工具与输出图（2026-11-05）

用户要求可视化节点正式放进 tools，不再在 example 中定义。现已新增 tools/visualization.py：GroupSpectrumPlot（组平均谱）、PredictionPlot（多目标 pred-vs-true 子图）、ConfusionMatrixPlot（单标签混淆矩阵或多标签按组分二分类子图）。均继承 Node，transform 保存图后原样返回输入；预测图以 X 传预测、y_true 传真值，可按 groups 聚合。混淆矩阵行是真值、列是预测，normalize 可选 true/pred/all，不能输入概率矩阵冒充标签。

example 已改为导入工具库节点，添加 PLSR 到 PredictionPlot 的 target_values Edge。每次运行将平均谱和测试集定量图写入 outputs/GSAD0004_example/figures，并复制到 visualizations；summary.json 的 figures 字段记录两图路径。定量图每个分子一个 subplot，共用一张图，显示与组均值评估一致的 RMSE/R2。GSAD0004 不含分类模型，故该例不生成伪造的混淆矩阵。

已重跑完整例子，数值结果未变化；6 项可视化节点检查和 2 项工作流连接检查通过，模型重载与图片路径检查通过。可视化类已从 example 中移除，旧的模块属性通过导入仍可解析。以上约定更新第 12 节关于本地可视化适配器的位置描述。


## 14. 用户更正输出目录含义（2026-11-05，优先于第 13 节）

outputs 中的“图”指整个算法流程 graph，不是光谱/定量等业务 plot。正式约定：outputs/<run>/ 保存模型、数值结果与 workflow_graph.png/.svg/.json；visualizations/<run>/ 保存各可视化 Node 生成的光谱、pred-vs-true、混淆矩阵等图，按 outputs 的运行相对目录镜像组织，不再把业务图复制到 outputs。

example 已据此修改并重跑：outputs/GSAD0004_example/ 中生成实际节点和连接的流程图，visualizations/GSAD0004_example/ 中保存两张业务图。summary.json 分别记录 workflow_graph 与 figures。若 --output-dir 位于项目 outputs 根下，完整保留相对层级；若是外部目录，则使用末级目录名作为 visualization root 下的运行目录。--visualization-dir 指可视化根目录。没有 outputs 运行目录的独立 visualize 指令维持已有显式输出路径语义。

新增 workflow_graph.py，按真实 Node/Edge 拓扑绘制节点名称、主要参数和边数据类型，导出完整 JSON。不是图执行器。运行检查包含目录镜像、无效连接/环拒绝、图文件生成；示例数值不变。已检查流程图外观并删除上一轮放错位置的四张重复图片，原始数据未动。


## 15. 通用工作流执行器（2026-11-05）

用户授权升级执行器，避免 PLSR 特殊处理。新增 workflow.py：NodeSpec 声明工具、输出方法、运行阶段、逐谱映射与堆叠；Workflow 从 Edge 拓扑排序执行 fit/predict/evaluate。fit 克隆工具并训练 requires_fit 节点，预测和评价不再 fit。预测不需要标签，评价可输入真值和分组并画图。外部 input 端口包括 X/y/groups，target_port=input 对应 X，fit:y 只传入训练，fit:X 可覆盖参考谱等训练输入。输出工具端口目前仅 output。支持无条件 DAG 分支与显式命名汇合，不处理自动元数据重排、条件分流或多输出端口提取。预测器之后再训练学习器需要 OOF，首版明确拒绝该结构。

example 已删除 transform_path 与手写 PLSR fit/predict，改为统一 Workflow 接口；依然按完整 group 留出。固定插值网格默认 602–1798、步长2；平均谱绘图现在只展示测试组，训练和普通推理不执行绘图。数值与此前一致，前两目标负 R2 保留。输出新增 workflow_spec.json 与 execution_trace.json，joblib 包含已拟合 Workflow；流程图显示实际数据和标签/分组边，业务图仍在 visualizations 对应运行目录。

to_dict/from_dict 可通过显式工具白名单导出/重建声明；重建必须重新 fit，不从 JSON 恢复训练状态。不自动分数据，不进行 AutoML；上层算法应提供组级验证或交叉验证。7 项执行器和 2 项示例映射检查通过，包含训练状态隔离、PLSR/分类器通用性、声明往返、分支汇合、无效结构/不安全级联拒绝。已重跑真实 example 并检查保存模型和阶段日志。新的说明覆盖第 12 节的固定路径执行限制。
