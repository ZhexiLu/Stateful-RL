# 实验方案：State-Mutating 环境中工具代理状态化自我纠错的隐式恢复学习

## 0. 硬件与算力约束

2x H100（160GB VRAM total）。这决定了：
- 可全参训练 Qwen3-4B；可全参训练 Qwen3-8B（需 DeepSpeed ZeRO-3 或 FSDP）
- 推理/rollout 时可用 vLLM 部署 8B 模型
- 不可全参训练 14B+（AWM 原论文用的 14B 需要更多卡）
- 并行环境实例数受限——AWM 原论文用 1,024 instances，我们需要缩减到 64-128

主要模型选择：**Qwen3-4B**（主实验）+ **Qwen3-8B**（验证 scaling）。与 AWM 原论文使用的 base model 一致（Qwen3 thinking models），保证结果可直接对比。

---

## 1. 数据准备：AWM 环境的 Read/Write 划分

### 1.1 环境获取

从 HuggingFace `Snowflake/AgentWorldModel-1K` 下载全部 1,000 环境。每个环境包含：scenario 描述、SQLite 数据库（schema + sample data）、MCP 接口代码、10 个 task + verification code。

### 1.2 工具操作语义标注

对 AWM 的 35,062 个工具进行操作语义分类。用本地部署的 Qwen3-8B 做分类，每个工具输入其函数签名、参数 schema 和自然语言描述，判断该工具属于以下哪类：

- `read_like`：只读取、查询、搜索、过滤信息，不改变任何外部状态
- `write_like`：创建新记录/实体
- `update_like`：修改已有记录/实体
- `delete_like`：删除记录/实体
- `mixed_or_unknown`：以上无法唯一判定

35,000 个工具，每次输入几百 token，8B 模型在单卡上几小时即可跑完。分类完成后随机抽 200 个工具人工验证准确率（预期 95%+，因为这是一个非常简单的意图分类任务）。

### 1.3 Task 级别标注：`info_seeking` vs. `state_mutating`

Task 类型划分不再只依赖“把 `initial_db_path` 同时作为 `final_db_path` 传入 verifier”这一单一启发式，而采用 **verification-driven 的混合标注策略**，尽量减少标签噪声。

**第一步：verification code 与 predicate 类型分析。**  
对每个 task 的 verification code 做程序分析，优先识别其检查对象是否包含数据库状态变化相关谓词，例如：新行存在、旧行删除、字段值更新、数量/余额/价格变化等。若主要检查的是返回文本、查询结果解释或数据库不变条件，则偏向 `info_seeking`。

**第二步：探针式执行。**  
在可执行层面做两类探针：
- `initial_db == final_db`：若任务在数据库不变时即可满足，支持 `info_seeking` 标签；
- 对少量合成 counterfactual state 做验证：若完成度明显依赖插入/更新/删除等状态变化，支持 `state_mutating` 标签。

**第三步：模糊样本复核。**  
当 verification code 同时包含查询类目标与状态变化目标、或探针结果不一致时，将该 task 标为 `ambiguous`，进入人工复核或单独分析集，而不是直接混入主分组。

这样做的原因是：我们的主结论高度依赖 task type 分组，因此必须把标签噪声控制在方法误差之外，而不是简单地把 verifier 的一个特例调用结果当作唯一依据。


### 1.4 Verification 子条件提取

AWM 的 verification code 本质上是一系列 SQL 查询、状态比较与条件判断。我们的目标不是生成自然语言解释，而是把它转换为 **可执行的中间谓词列表**，用于 step-level progress 计算。

每个子条件包含：
- 查询或检查逻辑；
- 条件类型（行存在、行不存在、字段相等、字段不等、标量接近目标值等）；
- 目标值或目标关系；
- 在中间状态上的执行方式。

提取流程采用“程序分析优先，LLM 辅助补全”的策略：
- 对常见 SQL/比较模板直接做规则抽取；
- 对较复杂的控制流，再用本地部署的 Qwen3-8B 生成候选谓词；
- 所有候选谓词都必须通过可执行性检查和 replay 校验，不可执行或语义不稳定的谓词直接剔除。

例如，一个“为用户创建订单并扣减余额”的 task 可以被拆成：

```
子条件 1: orders 表中存在 user_id=3 的新订单           → 行存在性检查
子条件 2: 该订单的 status 为 'completed'              → 字段值检查
子条件 3: 该订单的 total_price 为 150.0              → 标量目标检查
子条件 4: users 表中 user_id=3 的 balance 为 850.0   → 标量目标检查
```

这是一次性的离线预处理。建议随机抽样 100 个 task，人工核验其谓词列表是否与 verification logic 一致。


### 1.5 训练/测试划分

- 从 526 个 AWM 验证通过的环境（AWM 原论文使用的子集）中，按 scenario 类型分层抽样
- 训练集：400 环境（4,000 tasks）
- 验证集：60 环境（600 tasks）
- 测试集：66 环境（660 tasks）
- 确保训练/测试集的 info_seeking 与 state_mutating task 比例一致

---

## 2. Baselines

所有 baseline 与核心方法使用完全相同的底模、相同的 prompt、相同的最大交互轮数、相同的 invalid-call 终止规则和相同的 history truncation 机制。这样可以把性能差异尽量归因到 reward design，而不是 prompt 或 rollout policy 的变化。

### 2.1 Base Model Evaluation（无训练）

直接用 Qwen3-4B / 8B 的 base thinking model 在 AWM 环境中 rollout，收集：
- 各类 task 的 success rate（按 `info_seeking` / `state_mutating` 分组）；
- 显式错误后的行为模式（重复同一调用、改参数、换工具、先查询再操作、停止等）；
- 自发出现的恢复行为频率。

**目的**：建立自然恢复行为底线，并确认探索分布中是否存在足够的正向恢复样本供 RL 放大。

### 2.2 Baseline B1：Vanilla GRPO（Outcome-Only RL）

复现与 AWM 兼容的 GRPO 训练设置：
- step-level format penalty：无效工具调用格式或参数不可解析时给 -1.0，并立即终止；
- task-level outcome reward：Completed = 1.0，Partially Completed = 0.1，其余 = 0.0；
- history-aware training（sliding window truncation）；
- no-op 检测：连续重复完全相同的调用模式时终止。

该 baseline 对应“只靠终局结果能否学到恢复”。

### 2.3 Baseline B2：Action-Based Step Reward

在 B1 的基础上增加 action-based 的 step-level reward，对应“动作空间奖励”路线。每步 reward 不依赖数据库状态，只依赖动作本身及其直接返回：
- 工具调用格式正确且执行成功 → +0.1；
- 工具调用格式正确但执行返回错误 → 0；
- 调用了与 task 明显无关的工具 → -0.1。

相关工具集合可以通过任务描述、verification logic 和离线 8B 标注联合得到。所有 step reward 再按最大轮数归一化，使其与 outcome reward 处于同一量级。

**目的**：作为本文方法的直接对照，回答“收益是否仅来自任意 step reward”。

### 2.4 可选弱恢复基线（非主对照）

若算力与时间允许，可补充一个弱恢复基线：在显式错误后注入模板化诊断提示，再继续 rollout。该基线不作为主表核心对照，只用于说明“简单的错误提示”与“基于状态进度的训练”并不等价。


## 3. 核心方法：用于隐式恢复学习的 Verifier-Grounded State-Based Progress RL

### 3.1 核心思路

在 stateful 工具环境中，self-correction 的信号主要体现在 **状态变化** 而不是动作表面形式中。模型执行了错误的 write/update/delete 操作，某些目标条件会从满足变为不满足，或与目标状态的距离增大；若模型随后完成补偿或修复，这些条件又会重新恢复。因此，恢复能力可以通过状态进度的下降与回升来刻画，而不需要测试时的外部教师或自由形式的错误诊断文本。

这里的目标是 **隐式恢复学习**：我们不显式要求模型输出“我正在恢复”或某种 recovery type，而是希望它在坏状态出现后，条件性地切换到更有利于修复状态的后续动作。

我们的主张不是“增加任意 step reward”，而是更具体的：在 `state_mutating` 工具环境中，若目标是学习隐式恢复行为，则 step-level reward 应优先定义在 **状态空间**，而不是动作空间。

### 3.2 状态进度函数

对每个 task，设离线提取出的可执行中间谓词集合为

```
C = {c_1, c_2, ..., c_N}
```

定义当前状态 `s_t` 的归一化进度函数：

```
Phi(s_t) = (1/N) * Σ_i phi_i(s_t)
```

其中：
- 对二值谓词，`phi_i(s_t) = 1` 表示当前状态满足该条件，否则为 `0`；
- 对标量目标，使用相对初始状态的归一化距离：

```
phi_i(s_t) = max(0, 1 - |value_i(s_t) - target_i| / (|value_i(s_0) - target_i| + eps))
```

单步状态进度奖励定义为势能差：

```
R_prog(t) = Phi(s_{t+1}) - Phi(s_t)
```

这样定义后：
- 正进度表示当前操作把状态推向目标；
- 负进度表示当前操作破坏了状态；
- “先负后正”的轨迹自然对应隐式恢复行为。

### 3.3 何时计算进度奖励

主进度奖励只在可能改变状态的步骤后计算，即 `write_like / update_like / delete_like` 调用之后。对 `read_like` 步骤默认给 `0`。这样可以降低噪声与额外计算开销。

不过，为了避免模型在犯错后完全跳过状态检查，我们在很窄的恢复窗口中加入一个极小的验证奖励：若前一步出现显式执行错误，或导致 `R_prog(t-1) < 0`，而当前一步调用了与任务相关的 `read_like` 工具来核验状态，则给出一个很小的正奖励 `R_verify(t) = beta`，其中 `beta << lambda`。

### 3.4 Step Reward 与 Turn-Level Return-to-Go

第 `t` 个 tool turn 的即时奖励定义为：

```
r(t) = lambda * R_prog(t) + gamma * R_verify(t) + R_fmt(t) + 1[t=T] * R_outcome
```

其中：
- `R_outcome`：task-level outcome reward，Completed = 1.0，Partially Completed = 0.1，其余 = 0.0，**仅在轨迹终止步 T 出现**；
- `R_prog(t)`：状态进度奖励；
- `R_verify(t)`：错误后小幅验证奖励；
- `R_fmt(t)`：无效格式、参数不可解析、明显无进展循环等已有惩罚。

**为什么不直接对全轨迹求和。** R_prog(t) = Phi(s_{t+1}) - Phi(s_t) 是势能差，直接求和会 telescoping 为 Phi(s_T) - Phi(s_0)，局部的”先负后正”结构完全消失。训练时真正进入优化的只是”最终状态比初始状态好多少”，无法区分哪一步破坏了状态、哪一步修复了状态。

因此采用 **turn-level return-to-go**，对第 `i` 条 rollout 的第 `t` 个 tool turn：

```
G_{i,t} = sum_{u=t}^{T_i} eta^{u-t} * r_i(u)
```

其中 `eta ∈ (0,1)` 为折扣因子（实验中在 0.90–0.97 范围内验证）。取 `eta < 1` 使近端步骤的影响更大，坏写操作不会因为远处的一次恢复而被完全冲淡。

### 3.5 Turn-Level Group-Relative Advantage

标准 GRPO 对同一 task 的 G 条 rollout 各自计算 trajectory-level return，组内归一化后得到 advantage，所有 token 共享同一个 advantage。这一粒度无法区分轨迹内不同步骤的贡献。

本文将归一化粒度从 trajectory 改为 **turn**。对同一 task 下 G 条 rollout，在相同 turn index `t` 上做组内归一化：

```
A_{i,t} = (G_{i,t} - mu_t) / (sigma_t + eps)
```

其中 `mu_t, sigma_t` 是该组内所有在第 `t` 步仍未终止的 rollout 的 return-to-go 均值和标准差。`A_{i,t}` 被分配给第 `i` 条 rollout 中第 `t` 个 tool turn 生成的所有 token。

效果：若某步写操作导致负进度，该步 return-to-go 低于同组其他 rollout 同一 turn 的值，获得负 advantage；若随后的修复步骤成功恢复，那些步骤自身获得正 advantage。这是隐式恢复信号被保留的机制。

### 3.6 训练流程

```
离线预处理（一次性）：
  - 对每个 task 提取可执行中间谓词列表（1.4）
  - 对每个工具做操作语义分类（1.2）
  - 为 B2 / B2+ baseline 构建 relevant tool set 及离线评分

For each training step:
  1. 采样 batch of tasks（保留 info_seeking 与 state_mutating 的自然混合或设定混合比例）
  2. 对每个 task rollout G 条轨迹
     - rollout 中执行 format check / no-op check
     - 每次 state-mutating 调用后计算 R_prog(t)
  3. 终局运行 verifier，得到 R_outcome（仅赋予终止步）
  4. 对恢复窗口内的状态核验步计算 R_verify(t)
  5. 计算每步 r(t)，再计算折扣 return-to-go G_{i,t}
  6. 对同一 task 的 G 条 rollout，在同一 turn index 上做组内归一化得到 A_{i,t}
  7. 用 turn-level advantage 做 policy update
```

### 3.7 与 Action-Based Reward 的关键区别

Action-based reward 评估的是”这一步工具调用本身看起来是否合理”；State-Based Progress Reward 评估的是”这一步操作之后，环境状态是否更接近目标状态”。

前者无法稳定捕捉”状态被改坏后是否成功恢复”；后者则天然把”破坏”和”修复”映射为负进度与正进度，因此更适合训练 state-mutating 场景中的隐式 post-error recovery。此外，turn-level credit assignment 避免了势能差 reward 在 trajectory aggregation 中的 telescoping 退化，保留了 progress signal 的时间结构。


## 4. 评估协议

### 4.1 主评测问题

评测围绕三个问题展开：
1. state-based progress reward 是否优于 outcome-only RL；
2. 相比 action-based step reward，它是否更能提升 `state_mutating` 任务中的恢复能力；
3. 这种提升是否来自真实的隐式恢复，而不是更保守的停止或更少的执行。

### 4.2 主指标

| 指标 | 定义 | 数据源 |
|------|------|--------|
| **Task Success Rate (SR)** | task-level Completed rate | AWM verifier |
| **Post-Error Recovery Rate (RR)** | 轨迹中出现显式错误或负进度后，最终仍完成任务的比率 | 工具返回 + progress + verifier |
| **Dip-and-Recover Rate (DRR)** | 进度曾下降但最终恢复到更高水平并完成任务的轨迹比率 | progress 序列 |
| **Destructive Step Rate (DSR)** | state-mutating 步中导致负进度的比率 | progress 计算 |

所有主指标按 `info_seeking` / `state_mutating` 两类任务分别报告。

### 4.3 副作用与安全性指标

| 指标 | 定义 | 数据源 |
|------|------|--------|
| **Duplicate-Effect Count** | 重复写入、重复提交造成的副作用次数 | 状态差分 + 日志 |
| **Contradictory-Update Count** | 对同一对象写入冲突内容的次数 | 状态差分 |
| **Irreversible-Damage Rate** | 任务失败时状态已被破坏且未恢复的比例 | 状态快照 + verifier |
| **Premature-Stop Rate** | 在仍有恢复空间时提前停止的比例 | 轨迹分析 |

这些指标用于防止模型通过“更少执行”伪造恢复性能提升。

### 4.4 分组报告

所有指标按以下维度分组：
- `info_seeking` vs. `state_mutating`；
- 子条件数量分层（1–2 / 3–5 / 6+）；
- 轨迹中是否出现过显式错误或负进度。

### 4.5 Controlled Recovery Split

这是最关键的补充评测。除了自然 rollout 外，我们额外构造一个 `controlled recovery split`：
- 从测试集 `state_mutating` 任务中采样一批实例；
- 先让一个较弱策略或基线策略执行到第一次显式错误或第一次负进度步；
- 冻结该中间状态；
- 要求待测策略从这个已受损状态继续完成任务。

在这一 split 上至少报告：
- **Recovery Success**：从受损状态继续后最终完成任务的比例；
- **Additional Damage**：恢复过程中新增副作用数量；
- **Verification-First Rate**：进入恢复阶段后的第一步是否先进行状态核验。

该评测直接检验“模型是否学到了隐式恢复”，而不是仅仅在自然分布上成功率更高。

### 4.5.1 如何判断模型学到了隐式恢复

本文不通过模型的自述来判断其是否“意识到错误”，而是通过 **error-conditioned behavior** 来判断：当轨迹进入坏状态（显式执行错误或首次负进度）后，策略是否更频繁地转向状态核验、参数修正、工具切换、补偿性写入或安全停止；并且这种行为变化是否提高了从受损状态回到成功状态的概率。换言之，我们检验的是 **坏状态出现后策略是否发生条件性切换**，而不是模型是否显式声明自己进入了 recovery mode。

### 4.6 外部 Benchmark 泛化测试

在 AWM 训练后，在以下外部 benchmark 上测试辅助泛化：
- **BFCLv3**：function-calling 准确率；
- **τ²-bench**：conversational agent 场景。

需要明确：这些 benchmark 更偏一般 tool-use performance，而不是 stateful recovery 的直接测量，因此外测只作为辅助证据，不替代 AWM 内部的 controlled recovery 结果。


## 5. 消融实验

### 5.1 核心对比：五组主方法

| 方法 | Outcome Reward | Step-Level Reward | Credit Assignment | 描述 |
|------|---------------|------------------|------------------|------|
| Base | 无 | 无 | — | 未训练底模 |
| B1 | task-level | 无（仅 format penalty） | trajectory-level | Vanilla GRPO |
| B2 | task-level | action-based（成功/失败/无关） | turn-level | 粗糙动作空间奖励 |
| B2+ | task-level | action-based（8B 离线有用性评分） | turn-level | 更强动作空间奖励 |
| **Ours** | task-level | state-based progress | turn-level | 状态空间奖励 |

说明：B2 与 B2+ 同样使用 turn-level credit assignment，以确保对比公平——差异仅在 reward 定义（动作空间 vs. 状态空间），而不是 credit granularity。

论文最重要的问题不是”step reward 是否有用”，而是：**在同一环境、同一 prompt、同一 credit assignment 粒度下，状态空间奖励是否比动作空间奖励更适合训练隐式恢复能力。**

预期：在 `info_seeking` task 上几种方法差异有限；在 `state_mutating` task 上，`Ours > B2+ > B2 > B1`，尤其是在 RR、DRR 和 controlled recovery 指标上。

### 5.2 Reward Design vs. Credit Assignment Granularity

| 变体 | Reward | Credit Assignment | 描述 |
|------|--------|------------------|------|
| Ours（Full） | state-based progress | turn-level | 主方法 |
| Ours-TrajAgg | state-based progress | trajectory-level（标准 GRPO） | 消融：相同 reward，回退到 trajectory aggregation |

该实验直接回答：收益来自 reward design、credit assignment granularity，还是两者的结合。若 Ours-TrajAgg 显著弱于 Ours，说明 turn-level credit assignment 对于保留 progress signal 的局部结构是必要的（因为 trajectory aggregation 会导致势能差的 telescoping）。若 Ours-TrajAgg 仍然优于 B1，说明 state-based reward 即使在 trajectory level 也提供了更好的 trajectory ranking。

### 5.3 进度定义方式

| 变体 | 描述 |
|------|------|
| S1 | 只用二值谓词，不使用标量距离 |
| S2（Full） | 二值谓词 + 标量距离 |

该实验回答：标量字段的相对距离奖励是否比简单的通过/未通过判定更有效。

### 5.4 去掉错误后验证奖励

| 变体 | 描述 |
|------|------|
| V1（Full） | 使用恢复窗口中的 `R_verify` |
| V2 | 去掉 `R_verify`，只保留 `R_prog` |

该实验回答：错误后”先核验再修复”的偏好是否需要微弱的额外引导。同时，若 V2 的 task success 与 V1 持平但 Dip-and-Recover Rate 下降，说明 R_verify 确实在帮助恢复而非被 hack。

### 5.5 折扣因子 `eta`

测试 `eta ∈ {0.90, 0.95, 0.97, 1.0}`，观察近端折扣对恢复学习的影响。`eta = 1.0` 是一个重要的 boundary case：此时 return-to-go 不折扣，远处的恢复可以完全冲淡当前坏步骤的负信号。预期 `eta` 略小于 1 时恢复指标最优。

### 5.6 进度奖励系数 `lambda`

测试 `lambda ∈ {0.1, 0.2, 0.3, 0.5}`，观察中间进度优化与最终 task completion 的权衡。预期过大的 `lambda` 可能导致模型过度优化局部谓词而忽略整体完成。

### 5.7 训练分布中 state-mutating 占比

| 变体 | 描述 |
|------|------|
| R1 | 自然比例混合训练 |
| R2 | `state_mutating` 上采样 2x |
| R3 | 仅 `state_mutating` 训练 |

该实验回答：方法收益是否依赖训练分布中 state-mutating 任务的出现频率，以及这种训练偏置是否会损害对其他任务的泛化。


## 6. 行为分析

### 6.1 Progress 轨迹模式分析

对测试集 rollout 绘制 progress 轨迹的统计分布，重点比较 Base / B1 / B2 / Ours 四组在以下模式上的频率：
- **Monotonic Success**：进度基本单调上升并最终成功；
- **Dip-and-Recover Success**：进度下降后恢复并最终成功；
- **Dip-and-Fail**：进度下降后未恢复；
- **Flat / No Progress**：长期空转；
- **Continual Damage**：持续负进度。

若 state-based progress reward 确实在训练隐式恢复能力，则 Ours 的 `Dip-and-Recover Success` 频率应显著高于 B1 与 B2。

### 6.2 错误后下一步动作分布

对测试集中显式错误发生后的下一步动作进行分类：
- **Blind Retry**：同一工具同一参数重试；
- **Argument Revision**：同一工具不同参数；
- **Tool Switch**：切换到其他工具；
- **State Verification**：先调用 `read_like` 工具核验状态；
- **Stop / Abstain**：直接停止。

对比不同方法在这些行为模式上的分布变化，重点观察 Ours 是否在坏状态出现后更倾向于“先验证、再修复”，即是否表现出 error-conditioned policy shift。

### 6.3 Controlled Recovery Case Study

从 controlled recovery split 与自然测试集中各选取若干典型案例，覆盖：
- 单调推进最终成功；
- 先破坏后修复最终成功；
- 先破坏后继续恶化最终失败；
- B2 成功但 Ours 失败的反例；
- Ours 成功但 B2 失败的正例。

每个 case 展示完整轨迹、关键状态快照、局部谓词满足情况与 progress 序列。


## 7. 时间线估算

| 阶段 | 内容 | 时间 |
|------|------|------|
| Phase 0 | AWM 环境搭建 + 工具语义分类 + task type 混合标注 + 中间谓词提取与校验 + Base model evaluation | 3 周 |
| Phase 1 | B1（Vanilla GRPO）+ B2（Action-Based Reward）实现与训练 | 2 周 |
| Phase 2 | 核心方法实现（progress reward + recovery-window verification bonus）并完成主训练 | 2.5 周 |
| Phase 3 | Controlled recovery split 构造 + 主评测 + 副作用指标实现 | 1.5 周 |
| Phase 4 | 核心消融 + 8B 有限 scaling 验证 | 2 周 |
| Phase 5 | 外部 benchmark 评测 + 行为分析 + case study + 写作 | 3 周 |
| **合计** | | **~14 周** |

说明：在 2×H100 条件下，优先保证 4B 主结果、controlled recovery 和核心消融完整跑通；8B 仅做确认性实验，不强行扩展实验面。


## 8. 关键工程风险

**风险 1：task type 划分噪声过高。**  
应对：采用 verification code 分析、探针式执行与人工复核相结合的混合标注；模糊样本单独成组，不混入主结论。

**风险 2：中间谓词不可执行或语义不稳定。**  
应对：程序分析优先；所有 LLM 生成的谓词必须通过 replay 校验；对不稳定谓词直接剔除。

**风险 3：AWM 并行环境实例数不够。**  
AWM 原论文使用 1,024 instances/step，我们在 2×H100 条件下大概率只能支持 64–128。应对：优先完成 4B 主实验，多随机种子比盲目扩大模型更重要；必要时增加 training steps 或采用异步 rollout。

**风险 4：模型学到的是“少做事”，不是“会进行隐式恢复”。**  
应对：引入副作用指标、premature-stop 指标和 controlled recovery split，强制区分“安全停止”和“有效恢复”。

**风险 5：实验面过宽导致主线被稀释。**  
应对：主结果只围绕 Base / B1 / B2 / Ours、自然 rollout 与 controlled recovery 两类评测展开；其余实验只保留服务主假设的最小集合。
