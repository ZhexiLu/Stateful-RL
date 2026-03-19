# 研究计划：面向工具代理状态化自我纠错的隐式恢复学习

## 1. 研究背景与问题定义

大语言模型工具代理已经能够完成多轮工具调用、查询检索和结构化操作，但在自我纠错方面仍然粗糙。多数工作将自我纠错理解为失败后重新生成或反思后重试。然而，在真实工具系统中，错误并不总是停留在文本层或无副作用的查询层——许多工具调用会直接改变外部状态，例如创建记录、更新字段、删除实体、触发工作流或提交事务。一旦错误动作已经执行，后续行为必须基于新的环境状态进行恢复，而不是简单地重写一个正确的函数调用。

因此，本文将工具代理的自我纠错区分为两类：

1. **信息型纠错（information-seeking correction）**：针对只读、查询、检索、搜索、过滤或聚合类 API 的错误，通常可以通过改写参数、替换工具、基于返回结果重新规划等方式修正。
2. **操作型纠错（state-mutating recovery）**：针对 create / update / delete / write 等改变环境状态的 API 的错误，要求模型先确认当前状态，再决定重试、补偿、回滚、改写后续计划还是停止继续操作。

本文的核心观点：**工具代理的自我纠错不应被仅仅理解为失败后的文本反思或再次尝试。对于会改变环境状态的工具调用，更关键的问题是——模型能否在错误动作已经产生副作用之后，基于真实工具反馈和状态变化采取正确的恢复行为。**

本文进一步将这一能力表述为 **隐式恢复学习（implicit recovery learning）**：我们不要求模型显式输出“我犯错了”之类的自我诊断，也不预设离散的 recovery type；相反，我们关注模型是否会在状态偏离目标后, 切换到更稳健的后续行为，例如先核验状态、修改参数、补偿写入、切换工具或停止继续损害。

在这个问题定义下，state-mutating 环境不是对问题的人为收缩，而是更严格、更有区分度的训练与评测场景。本文将使用 AWM（Agent World Model）作为主训练与评测平台，因为 AWM 提供了可执行的代码驱动环境、SQLite 数据库状态管理、统一的 MCP 工具接口和基于代码增强的验证逻辑，适合研究写操作后的恢复行为。

## 2. 研究问题

1. 在多轮工具交互中，当环境状态已经被错误动作改变后，模型能否意识到并做出恢复性行为，而不是盲目重复或继续损害状态？
2. 在 state-mutating 工具环境中，仅靠结果奖励是否足以学习隐式恢复行为，还是需要更细粒度的局部状态监督？

3. 如何在不依赖测试时外部教师、也不显式标注 recovery type 的前提下，使模型学会恢复、补偿、验证、重规划或停止进一步损害？

4. 在可执行环境中，何种训练信号组合最有利于学习隐式恢复：结果奖励、下一状态信号、局部状态进度，还是它们的组合？

## 3. 与现有工作的关系

### 3.1 自我纠错与工具使用 RL 的已有经验

近期工作给出了几个对本文很重要的经验事实。

**On-policy 学习对于自我纠错是必要的。** SCoRe（Kumar et al., 2024）将自我纠错训练为"第一次尝试 + 第二次尝试"的序列决策问题，用多轮在线 RL 直接优化第二次尝试的改正能力。其两阶段训练设计——先在 base model 上做 RL 初始化以避免 collapse，再用 reward bonus 放大自我纠错——说明 SFT 在 offline correction traces 上的 distribution mismatch 是一个根本性问题。Reflect, Retry, Reward（Bensal et al., 2025）进一步表明，仅用二值任务成功反馈就可以通过 GRPO 强化失败后的反思重试行为，且只需要奖励 reflection tokens 而非整条轨迹。这些结果共同说明自我纠错应在模型自己真实出错的分布上学习。

**仅靠终局结果奖励通常不足以学习高质量的工具恢复。** ToolRL（Qian et al., 2025, NeurIPS 2025）系统研究了工具使用 RL 中的奖励设计，发现奖励粒度（correctness reward 的多级设计 vs. 二值）、格式奖励的权重比例、以及 GRPO cold start 等设计选择对性能有显著影响。OpenClaw-RL（Wang et al., 2026）进一步指出，工具输出、用户回复、终止状态和 GUI 变化等"下一状态信号"同时编码了两种信息：evaluative signals（通过 PRM 提取为标量奖励）和 directive signals（通过 Hindsight-Guided On-Policy Distillation 提取为 token 级方向性监督），不应只被压缩成一个标量。

**离线纠错语料和搜索式修正示例有价值，但不能替代在线恢复学习。** Tool-MVR、Agent-R 等方法能够构造反思或修正数据，但仍受限于静态数据分布与搜索成本。

**失败前缀会污染训练。** CLEANER（Xu et al., 2026）的核心发现是：在 agentic RL 中，包含执行错误的轨迹即使最终成功，其错误动作也会被 outcome reward 错误地正向强化（trajectory noise）。CLEANER 通过 Similarity-Aware Adaptive Rollback（SAAR）机制在数据层面消除错误前缀，将"noisy success"轨迹净化为"clean success"用于 policy update。这说明恢复学习中必须认真处理 credit assignment 问题。

**环境错误与策略错误需要显式区分。** CRITICTOOL（Huang et al., 2025, EMNLP 2025）在多个工具评测 benchmark 上分析了 function-calling 过程中的错误类型，明确区分了 internal model-driven errors（模型自身推理或调用错误）与 external environment errors（API 超时、权限不足等），并分别设计了 reflect+correct 和 retry+skip/finish 的评估维度。其实验显示，即使是 GPT-4o 的整体 self-critique 得分也仅为 69.01，说明工具错误恢复对当前模型仍然是一个显著挑战。

**State-mutating 场景中不能只看任务成功率。** Atomix（Mohammadi et al., 2026）为 agent 工具调用提供了 progress-aware transactional semantics——每次调用附带 epoch 标记，通过 per-resource frontier 追踪和 progress predicate 判断是否可以 commit。在含故障注入的真实 workload 上，Atomix 同时报告了 task success、contamination 和 irreversible-effect leakage，说明有副作用的工具使用必须额外关注状态损害。

**可执行环境允许使用程序验证而不是纯文本裁判。** AWM（Wang et al., 2026）采用 SQLite 数据库状态、验证代码与 code-augmented LLM-as-a-Judge（使用 GPT-5）共同形成 task-level reward（Completed=1.0, Partially Completed=0.1, 其他=0.0），并在 step-level 对无效工具调用格式立即惩罚（-1.0）并终止 rollout。其实验在 BFCLv3、τ²-bench 和 MCP-Universe 三个 benchmark 上验证了训练于合成环境的 agent 具有 out-of-distribution 泛化能力。

### 3.2 SimpleTIR 对本文的启发

SimpleTIR（Xue et al., 2025, NeurIPS 2025）研究的是 multi-turn Tool-Integrated Reasoning 的端到端 RL 训练稳定性，而不是 state-mutating 环境中的恢复策略学习。其核心发现是：外部工具反馈会推动生成分布偏离预训练分布，低概率 token 在多轮闭环中不断累积，引发梯度爆炸和错误的 credit assignment。SimpleTIR 的解决方案是过滤包含 void turn（既无 code block 也无 final answer 的 turn）的整条轨迹——将其从 policy update 中剔除但保留在 advantage estimation 中以维持无偏。

对本文的直接启发：

1. **恢复学习之前必须先控制无效轨迹污染。** 如果多轮工具 RL 本身已被格式崩坏和无效动作主导，模型无法学会可靠恢复。
2. **"自我修复"可以作为涌现行为出现，但不应与恢复学习混为一谈。** SimpleTIR 报告的"self-correction"和"cross-validation" pattern 来自作者使用 Claude-3.7-Sonnet 对正确答案轨迹做事后统计分析，而非直接优化目标。这类标签更适合作为行为分析工具而非训练主信号。

### 3.3 MAST 对本文的启发

"Why Do Multi-Agent LLM Systems Fail?"（Cemri et al., 2025, NeurIPS 2025 Datasets Track Spotlight）提出了 MAST failure taxonomy，基于 7 个 MAS 框架、1600+ 标注 traces 归纳了 3 大类 14 种失败模式（specification issues, inter-agent misalignment, task verification），使用 Grounded Theory 方法构建分类体系并达到 κ=0.88 的 inter-annotator agreement。该工作面向多代理系统而非单代理工具恢复，但有两点重要启发：

1. **错误分类必须是结构化的。** MAST 将失败分解为可辨识的 failure modes（reasoning-action mismatch, no/incomplete verification, incorrect verification, task derailment 等），说明错误分析需要显式 taxonomy 而不是仅比较成功率。
2. **LLM-as-a-Judge 更适合作为可扩展标注工具而非主训练信号。** MAST 使用 o1 模型扩展 taxonomy 标注，达到 94% accuracy 和 0.77 Cohen's Kappa，说明对于难以程序化判定的模糊案例可以引入模型辅助裁决。但主训练链路仍应优先依赖环境日志、返回码、状态差分和验证代码。

### 3.4 Fission-GRPO 的局限

Fission-GRPO（Zhang et al., 2026）是与本文研究方向最接近的现有工作。它将失败轨迹"裂变"为新训练实例：用一个 SFT 训练的 Error Simulator 为失败调用生成诊断反馈，拼接到上下文后 on-policy 重采样恢复 rollout。在 BFCL v4 Multi-Turn 上，Fission-GRPO 将 Qwen3-8B 的错误恢复率提升了 5.7%，整体准确率从 42.75% 提升到 46.75%。

但 Fission-GRPO 存在几个方法论上的问题。第一，Error Simulator 依赖 ground truth 构造的训练数据，在评测和训练使用同一 benchmark（BFCL）的情况下存在 teaching-to-the-test 风险。第二，仅在 BFCL v4 一个 benchmark 上评测，缺乏泛化性验证。第三，Error Simulator 生成的是静态诊断文本而非真实环境反馈，无法捕捉 state-mutating 场景中状态已改变的关键信息。第四，绝对数字较低（46.75%），说明该方法尚未解决根本性的恢复能力缺陷。

## 4. 研究定位

本文不主张提出新 benchmark，也不主张建立覆盖所有 agent 系统的统一错误评测体系。研究重点是：

1. 在 AWM 可执行环境中，提出一个适用于工具代理恢复学习的 RL 方法；
2. 让模型在真实交互中学习区分 read-like 错误与 state-mutating 错误；
3. 让模型在状态已改变的条件下学会恢复而不是只学会重试；
4. 使用与研究问题直接相关的评估指标，验证方法是否真正提高 post-error recovery。

定位为：**一篇关于工具代理状态化自我纠错中隐式恢复学习的 RL 方法论文，state-mutating 环境是最主要的训练与评测场景。**

## 5. 方法：用于隐式恢复学习的 Verifier-Grounded State-Based Progress RL

### 5.1 问题设定

我们考虑一个多轮工具交互环境。给定任务描述 \(x\)、初始环境状态 \(s_0\) 和工具集合 \(\mathcal{A}\)，策略 \(\pi_\theta\) 在第 \(t\) 步基于历史 \(h_t=(x,s_0,a_{<t},o_{<t})\) 产生动作 \(a_t\)，环境执行后返回观测 \(o_t\) 并转移到新状态 \(s_{t+1}\)。轨迹终止后，由任务验证器给出最终完成度奖励 \(R_{\text{out}}\)。

本文关注的问题不是一般意义上的 tool-use accuracy，而是 **state-mutating 场景中的 post-error recovery**：当模型的某一步操作已经改变外部状态后，后续策略能否基于新的真实状态继续完成任务，而不是盲目重试、继续污染状态或过早停止。相对于信息型查询任务，这类错误会在环境中留下副作用，因此更适合检验工具代理是否真正学到了恢复能力。

这里的“恢复”是 **隐式恢复**。本文不要求模型显式输出错误诊断标签、恢复阶段标签或 recovery taxonomy，而是将恢复定义为一种 **error-conditioned policy shift**：当状态偏离目标后，策略是否会转向更有利于修复状态的后续动作。

### 5.2 核心思想

本文的核心假设是：在数据库支撑的 stateful 工具环境中，恢复行为的监督信号并不一定需要额外教师，也不一定需要测试时的外部诊断文本。对于大量任务，verification code 本身已经刻画了目标状态应满足的局部条件。若某一步错误操作使当前状态偏离目标状态，则这些条件的满足程度会下降；若模型随后完成补偿或修复，则这些条件又会恢复。也就是说，**恢复能力可以直接通过状态进度的下降与回升来刻画**。

基于这一观察，我们不再把方法表述为宽泛的“Fault-Aware Recovery RL”，而是收束为一个更具体的命题：**利用 verifier 所隐含的中间状态进度信号，学习 state-mutating 工具环境中的隐式恢复策略。** 这一定义将主贡献集中在 reward design 与可验证状态建模上，避免方法边界过宽。

换言之，本文并不显式教授“错误意识”本身，也不要求模型先输出某种恢复类型；本文希望通过环境内生的状态进度信号，让策略在坏状态出现后自然学会更好的后续决策。

### 5.3 从 verification logic 到可执行中间谓词

对于每个任务，我们将其 verification logic 转换为一组可执行的中间状态谓词
\[
\mathcal{C} = \{c_1, c_2, \dots, c_N\},
\]
其中每个 \(c_i\) 用于刻画当前状态是否满足某个局部目标，或当前状态与目标值之间的距离。

这些谓词不依赖逐任务人工编写，而是通过 **程序分析优先、LLM 辅助补全** 的离线流程从 verification code 中提取。对于非标量条件，谓词通常表示行存在性、行不存在性、字段精确匹配、集合包含关系等；对于标量条件，谓词可进一步写成距离形式，例如余额、库存、价格或数量与目标值之间的差异。所有提取结果都要求能够在中间状态上直接执行，并通过小规模人工抽样与 replay 校验其可用性。

### 5.4 状态进度函数

我们定义当前状态 \(s_t\) 的归一化进度函数为
\[
\Phi(s_t) = \frac{1}{N}\sum_{i=1}^{N}\phi_i(s_t),
\]
其中 \(\phi_i\) 是第 \(i\) 个局部条件对应的局部进度分数。

对于二值谓词，我们定义
\[
\phi_i(s_t)=
\begin{cases}
1,& c_i(s_t)\ \text{satisfied},\\
0,& \text{otherwise}.
\end{cases}
\]

对于标量目标，我们定义归一化距离版本
\[
\phi_i(s_t)=\max\left(0,\,1-\frac{|v_i(s_t)-v_i^\star|}{|v_i(s_0)-v_i^\star|+\epsilon}\right),
\]
其中 \(v_i^\star\) 为目标值，\(\epsilon\) 用于避免零分母。

在此基础上，第 \(t\) 步动作的状态进度奖励定义为势能差
\[
R_{\text{prog}}^{(t)} = \Phi(s_{t+1})-\Phi(s_t).
\]

该定义直接统一了“推进目标”“破坏状态”和“错误后修复”三类行为：若动作使状态更接近目标，则奖励为正；若使状态偏离目标，则奖励为负；若轨迹出现先负后正的进度变化，则对应一个显式的恢复过程。

### 5.5 何时计算进度奖励

为降低噪声与额外计算，主进度奖励仅在可能改变环境状态的步骤后计算，即 write-like、update-like、delete-like 调用之后。对 read-like 步骤默认赋值为 0。这样做的原因不是否认查询行为的重要性，而是认为其价值应通过随后更正确的状态修改间接体现。

不过，为避免模型在错误后完全跳过状态检查，我们在很窄的恢复窗口中增加一个极小的 gated verification bonus。若前一步出现显式执行错误，或导致 \(R_{\text{prog}}^{(t-1)}<0\)，而当前一步调用了与任务相关的 read-like 工具以检查状态，则给出一个很小的正奖励 \(R_{\text{verify}}^{(t)}=\beta\)，其中 \(\beta \ll 1\)。该项只用于鼓励“先核验再修复”的行为模式，不替代主进度奖励。

### 5.6 Step Reward 与 Turn-Level Return-to-Go

任务终止后仍保留 AWM 风格的 outcome reward：Completed = 1.0，Partially Completed = 0.1，其余为 0.0。第 \(t\) 个 tool turn 的即时奖励定义为
\[
r^{(t)} = \lambda R_{\text{prog}}^{(t)} + \gamma R_{\text{verify}}^{(t)} + R_{\text{fmt}}^{(t)} + \mathbf{1}[t=T]\,R_{\text{out}},
\]
其中 \(R_{\text{out}}\) 仅在轨迹终止步 \(T\) 出现，\(R_{\text{fmt}}^{(t)}\) 表示无效工具调用格式、参数不可解析或明显无进展循环等已有惩罚项，\(\lambda\) 控制状态进度项权重，\(\gamma\) 远小于 \(\lambda\)。

**为什么不直接对全轨迹求和。** 由于 \(R_{\text{prog}}^{(t)} = \Phi(s_{t+1}) - \Phi(s_t)\) 是势能差形式，若直接在整条轨迹上求和，会产生 telescoping：\(\sum_t R_{\text{prog}}^{(t)} = \Phi(s_T) - \Phi(s_0)\)。此时，progress reward 的”先负后正”局部结构在 trajectory-level aggregation 之后完全消失，训练时真正进入优化的只是”最终状态比初始状态好多少”，无法区分哪一步破坏了状态、哪一步修复了状态。这与本文的核心目标——学习 state-mutating 场景中的隐式恢复行为——直接矛盾。

因此，我们采用 **turn-level return-to-go** 来保留 progress signal 的时间结构。对第 \(i\) 条 rollout 的第 \(t\) 个 tool turn，定义折扣 return-to-go：
\[
G_{i,t} = \sum_{u=t}^{T_i} \eta^{u-t}\, r_{i}^{(u)},
\]
其中 \(\eta \in (0,1)\) 为折扣因子（实验中在 0.9–0.97 范围内验证）。取 \(\eta < 1\) 的原因是：若 \(\eta = 1\)，未来修复带来的正奖励仍会较大程度抵消当前坏步骤的负奖励，使得坏写操作的负信号被远处的恢复冲淡。折扣使近端步骤的影响更大，坏步骤不会因为很远之后的一次恢复而被完全洗平。

### 5.7 Turn-Level Group-Relative Advantage

在标准 GRPO 中，同一 task 的 \(G\) 条 rollout 各自计算一个 trajectory-level return，组内归一化后得到 advantage，同一条 rollout 的所有 token 共享同一个 advantage。这一设计的 credit granularity 很粗，无法区分轨迹内不同步骤的贡献。

本文将归一化粒度从 trajectory 改为 **turn**。对同一 task 下 \(G\) 条 rollout，在相同 turn index \(t\) 上做组内归一化：
\[
A_{i,t} = \frac{G_{i,t} - \mu_t}{\sigma_t + \epsilon},
\]
其中 \(\mu_t, \sigma_t\) 是该组内所有在第 \(t\) 步仍未终止的 rollout 的 return-to-go 均值和标准差。\(A_{i,t}\) 被分配给第 \(i\) 条 rollout 中第 \(t\) 个 tool turn 生成的所有 token。

这一设计保留了 GRPO 的 group-relative flavor，但将 credit assignment 精细化到 turn level。其效果是：若某一步写操作导致负进度，该步的 return-to-go 会低于同组其他 rollout 在同一 turn 的 return-to-go，从而获得负 advantage；若随后的步骤成功修复，那些修复步骤自身会获得正 advantage。这正是隐式恢复信号被保留的机制。

### 5.8 工程控制

实现上保留两个工程性控制措施，但不将其写作主贡献。第一，格式非法、参数无法解析或接口不匹配的轨迹在出错步给予负奖励并终止。第二，连续重复的 no-op 或完全相同的调用模式被提前终止。它们的作用是减少无效轨迹污染、稳定训练，而不是定义本文的方法本体。

### 5.9 与现有工作的区别

本文的方法包含两个互相支撑的设计：**在状态空间而非动作空间定义 step-level reward**，以及 **在 turn level 而非 trajectory level 做 credit assignment**。

与 outcome-only RL 相比，本文不再将整条轨迹压缩成单一终局分数，而是显式利用环境状态的中间变化进行局部 credit assignment。与 action-based step reward 相比，本文不根据动作表面形式打分，而根据 **状态转移后果** 打分，因此能够识别”看起来合法但把状态改坏了”的步骤，也能够识别”前一步犯错、后一步修复”的恢复行为。与标准 GRPO 相比，turn-level advantage 避免了势能差 reward 在 trajectory aggregation 中的 telescoping 退化，保留了 progress signal 的时间结构。

因此，本文的主张不是”任何 step reward 都有帮助”，也不是”任何 dense reward 都改善 credit assignment”，而是更具体的：**在 state-mutating 工具环境中，若目标是学习隐式恢复行为，则 step-level reward 应优先定义在状态空间，并通过 turn-level credit assignment 保留其局部结构。**


## 6. 关于 LLM-as-a-Judge 的立场

本文不以 LLM-as-a-Judge 作为主训练信号来源。主训练链路优先依赖环境可验证信号：工具响应、返回码、状态转移、数据库快照以及 verification code 派生出的中间谓词。模型辅助裁决仅保留两个用途。

第一，用于 **离线标签扩展与模糊案例裁决**。当程序规则无法唯一判定某些边界情况时，例如工具执行成功但是否违背高层任务意图、状态未变化但是否构成合理补偿，才引入 grounded、受约束的模型辅助判别。

第二，用于 **行为分析与错误模式统计**。例如对 case study 做错误类型聚类、对恢复模式做离线归纳等。

因此，本文的方法依赖的是 verifier-grounded state progress，而不是自由形式的 model judging。Judge 只是离线辅助模块，不进入主训练闭环，也不承担主要结论的可重复性基础。


## 7. 实验设计

### 7.1 实验目标

实验要回答三个直接问题：

1. state-based progress reward 是否优于仅依赖 outcome reward 的 vanilla RL；
2. 相比 action-based step reward，它是否更能提升 state-mutating 任务中的 post-error recovery；
3. 该方法学到的是否是隐式恢复能力，而不是更保守地停止、减少执行，或简单地优化中间子条件。

### 7.2 主训练环境与任务分组

主训练与主评测平台使用 AWM。AWM 提供 SQLite 持久状态、可执行 MCP 工具接口、可重置环境以及 verification code，因此能够同时支持终局完成度评估和中间状态检查。

任务按两层标准分组。第一层是主分组：`info_seeking` 与 `state_mutating`。这一标签不再只依赖 `initial_db == final_db` 的单次 verifier 探测，而采用 **verification code 分析 + 可执行谓词类型 + 探针式运行** 的混合策略。若任务目标主要是返回信息、解释查询结果或在初始数据库不变时即可被满足，则归为 `info_seeking`；若谓词列表明确要求数据库状态变化，或探针运行显示任务完成依赖插入、更新、删除等副作用，则归为 `state_mutating`。对于少量模糊样本，进入人工复核集，不直接参与主分组统计。

第二层是复杂度分组：按每个任务提取出的局部谓词数量分桶，用于观察随着目标复杂度增加，状态进度奖励是否更有价值。

### 7.3 基线方法

所有方法共享相同的底模、相同的 prompt、相同的最大交互轮数、相同的 invalid call 终止规则和相同的 history truncation 机制。

我们使用四组对照：

- **Base model**：不做 RL，直接在 AWM 上 rollout，用于建立自然恢复行为底线；
- **Vanilla GRPO**：只使用 outcome reward 与格式惩罚；
- **Action-based step reward**：在 vanilla GRPO 基础上增加工具调用层面的 step reward，评估“动作本身好不好”；
- **Ours: State-Based Progress RL**：本文方法，在状态空间定义进度奖励。

若算力允许，可额外加入一个弱恢复基线，例如错误后加入模板化诊断提示再重采样，但这不作为主对照。主比较对象始终是：**动作空间奖励 vs. 状态空间奖励**。

### 7.4 训练设置

主实验固定为 Qwen3-4B；Qwen3-8B 仅用于有限的 scaling confirmation，而不要求所有设置完整重跑。原因是 2×H100 条件下，4B 更适合做足随机种子与消融实验。主结果建议使用 3 个随机种子报告均值与标准差；若算力紧张，外部 benchmark 与个案分析可使用最佳 checkpoint 做单次评估。

训练超参数沿用 AWM-compatible GRPO 设置：统一的最大轮数、group size、KL penalty、history-aware truncation 和 invalid trajectory handling。状态进度系数 \(\lambda\) 与验证奖励系数 \(\gamma\) 在验证集上调优，但只在少量候选值中选择，避免过度搜索。

### 7.5 主评测指标

主表只保留四个核心指标：

- **Task Success Rate (SR)**：最终任务完成率；
- **Post-error Recovery Rate (RR)**：轨迹中出现显式错误或负进度后，最终仍完成任务的比例；
- **Dip-and-Recover Rate (DRR)**：进度轨迹中出现明显下降后又恢复至更高水平并最终完成的比例；
- **Destructive Step Rate (DSR)**：state-mutating 步中导致负进度的比例。

这四个指标分别对应总体性能、错误后恢复、恢复轨迹证据和破坏性副作用控制。所有主指标都按 `info_seeking` / `state_mutating` 两类任务分别报告。

### 7.6 副作用与安全性指标

针对 `state_mutating` 子集，额外报告以下副作用指标：

- `duplicate_effect_count`：重复写入或重复提交造成的副作用次数；
- `contradictory_update_count`：对同一对象写入互相冲突内容的次数；
- `irreversible_damage_rate`：任务失败时，状态已被破坏且未能在轨迹内恢复的比例；
- `premature_stop_rate`：在仍有恢复空间时提前停止的比例。

这些指标的作用是防止模型通过“更保守、更少执行”来伪造恢复性能提升。

### 7.7 Controlled Recovery Split

除了在自然 rollout 上评估之外，我们额外构造一个 **controlled recovery split**。具体做法是：从测试集中的 `state_mutating` 任务采样一批实例，先让一个较弱策略或基线策略执行到第一次显式错误或第一次负进度步，冻结此时的中间状态，然后要求待测策略从该状态继续完成任务。

这一评测更直接回答“在环境状态已经被改坏后，策略是否学到了隐式恢复”。在该 split 上至少报告三项指标：

- `recovery_success`：从损坏状态继续后最终完成任务的比例；
- `additional_damage`：恢复过程中新增的副作用数量；
- `verification_first_rate`：进入恢复阶段后，第一步是否先执行状态核验。

### 7.7.1 如何判断模型学到了隐式恢复

本文不通过模型自述来判断其是否“意识到错误”，而是通过 **error-conditioned policy shift** 来判断：当轨迹进入坏状态（显式执行错误或首次负进度）后，策略是否更频繁地转向状态核验、参数修正、工具切换、补偿性写入或安全停止；并且这种行为变化是否提高了从受损状态回到成功状态的概率。也就是说，本文检验的是恢复行为是否被学到，而不是模型是否显式输出 recovery label。

### 7.8 外部泛化评测

为验证在 AWM 中学到的恢复偏好不会完全局限于训练环境，我们在外部 benchmark 上做辅助泛化评测。优先保留 BFCLv3 与 τ²-bench，用于观察 function calling 和对话式 agent 场景中的迁移效果。

需要明确的是，这些 benchmark 更偏一般 tool-use performance，而不是 stateful recovery 的直接测量。因此，外部评测只作为辅助证据，不能替代 AWM 内部的 controlled recovery 结论。

### 7.9 消融实验

消融以回答”提升到底来自哪里”为目标，保持精简：

1. **No progress reward**：去掉状态进度项，对应 vanilla GRPO；
2. **Action-based reward**：验证收益是否只是因为引入了任意 step reward；
3. **Action-based reward+ (B2+)**：使用 8B 模型对每步工具调用做离线评分（”这步调用对完成任务是否有帮助？”→ 0-1 分数），作为更强的 action-space baseline，回答”是否仅因为 B2 太弱才赢”；
4. **Trajectory-aggregated progress (方案 A)**：使用相同的 state-based progress reward，但回退到标准 GRPO trajectory-level aggregation，验证 turn-level credit assignment 是否是必要的——即收益来自 reward design、credit assignment granularity，还是两者的结合；
5. **Binary-only progress**：只使用二值谓词，不使用标量距离；
6. **No verification bonus**：去掉恢复窗口中的状态核验奖励；
7. **折扣因子 \(\eta\)**：在 \(\{0.90, 0.95, 0.97, 1.0\}\) 中比较，验证近端折扣对恢复学习的影响；
8. **State-mutating oversampling**：考察训练分布中 state-mutating 任务占比对方法收益的影响。

### 7.10 行为分析

行为分析围绕隐式恢复主假设展开，而不是做泛化的 trace 统计。重点报告：

- progress trajectory 的模式分布：monotonic success、dip-and-recover success、dip-and-fail；
- 显式错误或负进度后的下一步动作分布：blind retry、argument revision、tool switch、state verification、stop；
- 8–10 个代表性 case study，其中至少一半来自 controlled recovery split。


## 8. 预期贡献

1. **问题定义层面**：将工具代理自我纠错进一步收束为 state-mutating 工具环境中的 **隐式恢复学习**，并明确指出该场景比纯查询型纠错更能暴露真实部署风险。
2. **方法层面**：提出 verifier-grounded 的 State-Based Progress RL，包含两个互相支撑的设计——将 verification logic 转换为可执行中间谓词并在状态空间定义 step-level reward，以及采用 turn-level return-to-go 与 turn-level group-relative advantage 保留 progress signal 的时间结构、避免势能差 reward 的 telescoping 退化。
3. **实验层面**：在 AWM 可执行环境中，从自然 rollout 与 controlled recovery 两个角度验证模型是否真正学会在状态已改变的条件下进行隐式恢复，而不仅仅是提高总体成功率。


## 9. 风险与应对

**风险 1：中间谓词提取质量不足，导致 progress reward 噪声过大。**  
优先采用程序分析与模板提取，仅在必要处使用 LLM 辅助；对提取结果进行抽样人工校验与 replay 验证，不可执行或语义不稳定的谓词直接剔除。

**风险 2：模型学到的是保守停止，而非有效的隐式恢复。**  
在主评测外增加副作用与 premature-stop 指标，并构造 controlled recovery split，直接检验从损坏状态出发的恢复能力。

**风险 3：实验主张被“任意 dense reward 都有效”解释掉。**  
将 action-based step reward 设为强基线，并将论文主问题明确写成“状态空间奖励是否优于动作空间奖励”。

**风险 4：task type 划分存在标签噪声。**  
不再依赖单一的 `initial == final` 启发式，而使用 verification code 分析、可执行探针与人工复核相结合的混合标注策略；模糊样本单独成组，不混入主结论。

**风险 5：算力不足导致实验面过宽。**  
主结果固定在 Qwen3-4B，优先完成主对比、controlled recovery 和核心消融；8B 仅做有限的确认性实验。


## 参考文献

1. Kumar, A., Zhuang, V., Agarwal, R., et al. Training Language Models to Self-Correct via Reinforcement Learning (SCoRe). ICLR 2025. arXiv:2409.12917
2. Bensal, S., Jamil, U., Bryant, C., et al. Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning. 2025. arXiv:2505.24726
3. Qian, C., Acikgoz, E.C., He, Q., et al. ToolRL: Reward is All Tool Learning Needs. NeurIPS 2025. arXiv:2504.13958
4. Wang, Y., Chen, X., Jin, X., et al. OpenClaw-RL: Train Any Agent Simply by Talking. 2026. arXiv:2603.10165
5. Wang, Z., Xu, C., Liu, B., et al. Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning. 2026. arXiv:2602.10090
6. Xue, Z., Zheng, L., Liu, Q., et al. SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning. NeurIPS 2025. arXiv:2509.02479
7. Huang, S., Fang, Z., Chen, Z., et al. CRITICTOOL: Evaluating Self-Critique Capabilities of Large Language Models in Tool-Calling Error Scenarios. EMNLP 2025. arXiv:2506.13977
8. Mohammadi, B., et al. Atomix: Timely, Transactional Tool Use for Reliable Agentic Workflows. 2026. arXiv:2602.14849
9. Zhang, Z., Zhao, F., Wang, R., et al. Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors. 2026. arXiv:2601.15625
10. Xu, T., et al. CLEANER: Self-Purified Trajectories Boost Agentic Reinforcement Learning. 2026. arXiv:2601.15141
11. Cemri, M., Pan, M.Z., Yang, S., et al. Why Do Multi-Agent LLM Systems Fail? NeurIPS 2025 Datasets & Benchmarks Track (Spotlight). arXiv:2503.13657
