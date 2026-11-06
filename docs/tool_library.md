# 工具库检查说明

工具库实现节点、边和单个工具。上层 workflow.py 提供通用执行器，automl.py 提供经典流程搜索；数据量分析尚未实现。根目录 main.py 提供 visualize、example、automl、predict 子命令，使用方法见英文 README。

## 三个主要动作

- `fit`：学习并保存训练参数；返回工具本身。
- `transform`：把输入变成处理后的光谱或特征。
- `predict`：输出类别、目标值或组分系数。

固定规则工具（如 SG 平滑、裁剪）不必先 fit。标准化、PCA、分类和回归等工具必须先 fit。学习训练数据统计量的工具只能在训练集上 fit。

## 输入格式与连接

`Spectra` 包含 `intensity`（每行一条谱）、`axis`（每列的实际波数）以及可选 `sample_ids`。不允许把列序号当作真实波数，不隐式插值，也不进行超出实测范围的外推。

`Node` 是所有工具的共同父类。`Edge` 只记录连接：起点、终点、输出/输入端口、数据类型、条件。`Condition("presence_probability", "ge", 0.5)` 表示概率达到阈值时启用；它不会执行 Python 字符串。全图循环检查、端口类型检查、路由和节点调度将在后续实现。

## 工具检查要点

| 工具类别 | 主要接口/输出 | 需要注意 |
| --- | --- | --- |
| 前处理 | `transform`，保留光谱轴 | min-max 默认逐谱；标准化默认按训练集逐特征；可显式切换 scope |
| 分类 | `fit(X,y)`，`predict`，部分支持 `predict_proba` | 普通类别向量与多组分存在矩阵不同；后者设置 multilabel=True |
| PLS-DA | one-hot 标签上做 PLS 回归，再选择最高分对应类别 | 分数不是校准概率；当前不支持多标签 |
| k-means | `fit` 后 `predict` 输出簇编号；`transform` 输出到各中心的距离 | 簇编号不是自动确定的化学身份 |
| t-SNE | `fit_transform` 输出坐标 | 仅展示已有数据；不假装具备新样本变换或类别预测能力 |
| 回归 | `predict` 返回二维目标值数组 | 数值可以是浓度或其他连续目标；单位及物质顺序由训练任务规定 |
| Digital SERS | `fit(X,y,groups=...)`；`predict(X,groups=...)` | 需要预先给定阈值与独立检测事件分组；按组输出，全部阳性时拒绝有限浓度估计 |
| CLS/NNLS | `fit(参考组分谱)`；`transform(混合谱)` 返回结果对象 | 输出系数、参考谱、重构谱和残差；不能凭空找到未知纯谱；系数不自动等于浓度 |
| PCA/因子分析 | `fit` 和 `transform` | 学习特征表示，不给出类别或化学浓度 |
| 峰提取 | 固定波段的峰高、峰位或面积，输出固定宽度矩阵 | 参数 statistic 选择特征；不包含峰形拟合或自动化学归属 |
| PeakDetector | 每条谱局部峰的波数和强度列表 | 峰数量不固定，不能直接当作固定维度模型输入 |
| 均值/中位数/最大/最小 | 无 groups：逐行汇总特征；有 groups：按样本汇总各条谱 | 聚合后必须对应组级标签；不同物质或单位不能直接平均 |
| 光谱作差 | `transform(X, reference=...)` | 必须同轴；一条背景可广播；多条背景必须具有匹配的 sample_ids |

固定参数通过构造函数传入，例如 `SGFilter(window_length=11, polyorder=3)`。sklearn/XGBoost 工具使用 `parameters` 字典，例如 `RFClassifier(parameters={"n_estimators": 100, "random_state": 0})`。建议先检查各模块默认值，再确定未来的搜索空间。

## 本版有意保留的边界

1. 没有自动划分训练/验证数据，也没有自动生成折外预测；不能直接把预测级联视为已避免泄漏。
2. 没有自动选择“最好流程”。
3. visualize 可从 content.csv 定位已迁入的数据并读取光谱；仍没有数据库下载、格式转换或训练数据加载流程。
4. 没有训练任何深度学习模型，也没有引入深度学习依赖。
5. 测试仅检查数学操作、输入约束和接口行为，不能证明真实 SERS 分析的准确度。

详细默认值、局限与研究来源参见根目录英文 README 和源码注释。


## 可视化节点

工具位置：`tools/visualization.py`。三个节点均继承 Node，调用 transform 保存图片后原样传回输入，不在绘图过程中训练模型。

- GroupSpectrumPlot：输入 Spectra 和可选 groups，用已有 MeanAggregator 计算组平均谱。
- PredictionPlot：输入预测值 X 和真值 y_true；多分子分别画 subplot，可选 groups 对重复结果先平均。unit 标注单位，molecule_names 指定目标顺序。
- ConfusionMatrixPlot：输入预测标签 X 和真值 y_true；默认画计数，行是真值、列是预测。labels 指定类别顺序，normalize 选择归一化；多标签需显式 multilabel=True。

output_path 指定图片输出位置，show=True 额外弹窗。example 的光谱与定量图只保存在 visualizations/GSAD0004_example 中，与 outputs 的运行目录结构对应；outputs/GSAD0004_example 中保存实际 Node/Edge 流程的 workflow_graph.png/.svg/.json。所有路径写入 summary.json。详见 README 的输入和使用说明。


## 统一执行器

workflow.py 的 Workflow 与 NodeSpec 已取代示例 transform_path。执行器根据 Edge 排序，不按模型名称分支。fit 学习训练参数，predict 复用参数，evaluate 额外接入真值/分组绘图。方法与阶段通过 NodeSpec 声明；标签训练边使用 fit:y。工作流声明可保存为 JSON 并通过工具白名单重建，详细示例和当前限制见 README。阶段图/日志在 outputs，业务图在 visualizations 对应运行目录。需要 OOF 的模型级联、条件路由和自动分组重排尚未实现，不能认为所有图都已可安全执行。
## AutoML 补充（2026-11-05）

统一入口 `main.py automl --data-id <ID|all> --output-dir <新运行目录>`。搜索位于 `automl/search.py`。公共原始数据读取统一为 `core.data.load_dataset`，默认保留逐条谱、原始轴、标签、样本组与划分块；不默认平均或插值。旧组均值基线在 `automl/representation.py` 中使用显式 Interpolate/MeanAggregator 节点准备，仍未搜索聚合位置和策略。完整参数、验证纪律和输出说明见根目录 README。

新增 `tools/model_selection.py`：`RidgeRegressor(parameters)`、`KernelRidgeRegressor(parameters)`、`TargetScaledRegressor(regressor)`、`IndependentPresence(classifier)`、`ConstantPredictor(classification=False)`。它们均继承 Node。目标缩放只学习训练标签，输出恢复原单位；IndependentPresence 对每列单独训练，某训练折只有一种标签时使用常量。ConstantPredictor 为固定基线，不需要推断时提供真值。

`main.py predict` 可对无标签新目录预测，输入为保存网格上的组均值表示；不覆盖已有输出。模型 joblib 仅加载可信本地文件。流程声明支持嵌套 Node 参数的显式白名单重建，不保存拟合状态。
