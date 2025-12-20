下面是一份围绕 **RumorDiffusion** 的完整、高规格研究方案，你可以直接拿去改成开题报告或论文 skeleton，用处比较明确。

---

## 0. 工作标题（可改）

> **RumorDiffusion: Prefix-Conditioned Diffusion for Early Rumor Detection Under Sparse and Incomplete Social Cascades**

中文可写：
**RumorDiffusion：面向稀疏与不完备传播的前缀条件扩散早期谣言检测框架**

---

## 1. 研究目标与问题设定

### 1.1 研究目标

在真实社交平台上，**扩散早期**（仅有极少数转发/评论）就给出消息是否为谣言的高置信度判断，并支持：

* 在极少交互数（如前 5/10/20 条互动）下仍具备可靠判别力；
* 面对严重缺失或噪声化的传播结构（不全的转发树、缺失 follow 图）保持稳健；
* 提供可解释的“不确定性 + 早停策略”，减少不必要的人工审核与延迟。

### 1.2 数据与符号

* 每条消息 $u$ 的传播被表示为一个**事件序列**：

  $$
  \mathcal{E} = \{e_i\}_{i=1}^{N},\quad
  e_i = (\text{user}_i, \text{text}_i, \text{time}_i, \text{parent}_i)
  $$
* 对应的传播结构可表示为一棵或多棵**转发树**或有向图：

  $$
  G = (V, E),\quad V \subseteq \{\text{user}_i\},\; E \subseteq V \times V
  $$
* 每条消息有标签 $y \in \{0,1\}$（非谣言/谣言），以及可选的**传播统计特征**：

  $$
  \text{stats} = (\text{size}, \text{depth}, \text{branch\_factor}, \dots)
  $$

**早期检测设定**：

* 随时间到达的前缀：

  $$
  \mathcal{E}_{\text{prefix}}^{(k)} = \{e_i\}_{i=1}^{k},\quad k \ll N
  $$
* 模型在仅观测到 $\mathcal{E}_{\text{prefix}}^{(k)}$ 的情况下，产出：

  * 谣言概率 $\hat{p}_k = P(y=1|\mathcal{E}_{\text{prefix}}^{(k)}, G_{\text{prefix}}^{(k)})$
  * 不确定性度量 $u_k$（如方差、置信区间）
  * 是否在此时做出最终决策（早停）或继续等待更多交互。

---

## 2. 核心研究问题

1. **前缀 → 全量传播的后验建模**
   如何在仅观测到 early prefix 的情况下，学习一个生成式模型，对“完整传播态势”的潜在表示进行建模，从而提升早期判别力。

2. **早停决策与不确定性估计**
   如何利用生成模型的多样性（多次采样）估计预测不确定性，并据此设计早停策略，在给定误报/漏报/延迟成本的约束下，优化整体决策表现。

3. **结构不完备条件下的鲁棒性**
   在真实平台不能获得完整传播图和社交关系的前提下，如何设计结构扰动和条件编码，使得模型对图缺失和噪声保持鲁棒。

---

## 3. 总体方案概述（RumorDiffusion）

RumorDiffusion 的整体思路：

* 将“完整传播态势”的表示 $\mathbf{x}_0$ 作为**扩散隐变量**；
* 通过 **full encoder** 从完整传播图/序列提取 $\mathbf{x}_0$，通过 **prefix encoder** 从早期前缀提取条件表征 $\mathbf{c}$；
* 在训练时对 $\mathbf{x}_0$ 施加前向高斯噪声，训练一个条件扩散模型在 $\mathbf{c}$ 条件下逆向去噪；
* 同时在 $\mathbf{x}_0$ 上训练谣言分类和“未来传播统计”预测；
* 在推理时，仅用 prefix encoder 得到 $\mathbf{c}$，在其条件下采样多次 $\mathbf{x}_0^{(s)}$，得到谣言概率分布及不确定性，并配合早停策略做决策。

---

## 4. 数学建模与模型结构

### 4.1 Prefix / Full Encoder 设计

#### 4.1.1 Prefix encoder $f_{\text{prefix}}$

输入：前缀事件 $\mathcal{E}_{\text{prefix}}^{(k)}$ 与对应局部结构 $G_{\text{prefix}}^{(k)}$.

1. **文本编码**

   * 使用预训练 Transformer（如 BERT/RoBERTa/DeBERTa）编码每条文本：

     $$
     \mathbf{h}^{\text{text}}_i = \text{Enc}_{\text{text}}(e_i.\text{text})
     $$

2. **用户画像编码**

   * 将用户静态特征（注册时间、粉丝数、认证类型等）映射到向量：

     $$
     \mathbf{h}^{\text{user}}_i = \text{MLP}_{\text{user}}(e_i.\text{user\_feat})
     $$

3. **时间编码**

   * 对时间戳做位置/周期编码或 Time2Vec：

     $$
     \mathbf{h}^{\text{time}}_i = \text{TimeEnc}(e_i.\text{time})
     $$

4. **局部图编码（转发树）**

   * 在前缀子图 $G_{\text{prefix}}^{(k)}$ 上应用 GNN / Graph Transformer：

     $$
     \mathbf{h}^{\text{graph}}_i = \text{GNN}(G_{\text{prefix}}^{(k)}, \{\mathbf{h}^{\text{text}}_i \oplus \mathbf{h}^{\text{user}}_i \oplus \mathbf{h}^{\text{time}}_i\})
     $$

5. **多模态融合 & pooling**

   * 对每个事件节点融合多模态特征：

     $$
     \mathbf{z}_i = \text{Fuse}(\mathbf{h}^{\text{text}}_i, \mathbf{h}^{\text{user}}_i, \mathbf{h}^{\text{time}}_i, \mathbf{h}^{\text{graph}}_i)
     $$
   * 用时间注意力或结构注意力对 $\{\mathbf{z}_i\}_{i=1}^k$ 做聚合：

     $$
     \mathbf{c} = f_{\text{prefix}}(\mathcal{E}_{\text{prefix}}^{(k)}, G_{\text{prefix}}^{(k)}) = \text{AttnPool}(\{\mathbf{z}_i\}_{i=1}^k)
     $$

#### 4.1.2 Full encoder $f_{\text{full}}$

输入：全量事件 $\mathcal{E}_{\text{full}}$ 与全量传播图 $G_{\text{full}}$。结构与 prefix encoder 类似，但在完整图上运行：

$$
\mathbf{h}_0 = f_{\text{full}}(\mathcal{E}_{\text{full}}, G_{\text{full}}) = \text{GraphPool}(\{\mathbf{z}_i\}_{i=1}^N)
$$

通过线性映射得到扩散空间中的初始隐变量：

$$
\mathbf{x}_0 = W \mathbf{h}_0,\quad \mathbf{x}_0 \in \mathbb{R}^d
$$

> 备注：Prefix / Full encoder 可以共享部分参数（如文本编码器），也可以部分解耦；这部分可在消融中验证。

---

### 4.2 前向扩散与逆向去噪（前缀条件扩散）

#### 4.2.1 前向扩散（训练时）

对 $\mathbf{x}_0$ 施加标准 DDPM 噪声：

$$
q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}\left(\sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I}\right),\quad t=1,\dots,T
$$

闭式表达：

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},\quad
\boldsymbol{\epsilon} \sim \mathcal{N}(0,\mathbf{I}),\;
\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)
$$

#### 4.2.2 逆向去噪（前缀条件）

训练参数化分布 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{c})$，其中条件信息仅来源于 **prefix 表征 $\mathbf{c}$**：

* 去噪网络（如 Transformer / MLP）：

  $$
  \hat{\boldsymbol{\epsilon}}_\theta = \text{Model}_\theta(\mathbf{x}_t, t, \mathbf{c})
  $$

* 通过 FiLM 或 cross-attention 注入条件：

  $$
  \mathbf{x}'_t = \text{FiLM}(\mathbf{x}_t, \mathbf{c}) \quad \text{或}\quad
  \mathbf{x}'_t = \text{CrossAttn}(\mathbf{x}_t, \mathbf{c})
  $$

* 逆向均值：

  $$
  \boldsymbol{\mu}_\theta(\mathbf{x}_t,t,\mathbf{c}) =
  \frac{1}{\sqrt{1-\beta_t}}
  \left(
    \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\boldsymbol{\epsilon}}_\theta
  \right)
  $$

* 逆向采样：

  $$
  \mathbf{x}_{t-1} \sim \mathcal{N}\left(\boldsymbol{\mu}_\theta(\mathbf{x}_t,t,\mathbf{c}),\tilde{\beta}_t\mathbf{I}\right)
  $$

---

### 4.3 多任务训练目标

#### 4.3.1 噪声预测损失（扩散主损失）

$$
\mathcal{L}_{\text{noise}} =
\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}, k}
\left[
  \left\|\boldsymbol{\epsilon} -
  \hat{\boldsymbol{\epsilon}}_\theta\bigl(\mathbf{x}_t, t, \mathbf{c}^{(k)}\bigr)
  \right\|_2^2
\right]
$$

其中 $k$ 为随机采样的前缀长度，$\mathbf{c}^{(k)} = f_{\text{prefix}}(\mathcal{E}_{\text{prefix}}^{(k)},G_{\text{prefix}}^{(k)})$。

#### 4.3.2 谣言检测损失

在 $\mathbf{x}_0$ 上接一个分类头 $g_\phi$：

$$
\hat{p} = g_\phi(\mathbf{x}_0),\quad
\mathcal{L}_{\text{cls}} = \text{CE}(\hat{p}, y)
$$

可选地，避免梯度干扰，可对 $\mathbf{x}_0$ 使用 stop-gradient 或分支投影。

#### 4.3.3 未来传播统计预测损失（保证 $\mathbf{x}_0$ 含“未来态势”信息）

在 $\mathbf{h}_0$ 或 $\mathbf{x}_0$ 上接回归头预测未来传播统计：

$$
\hat{\text{stats}} = f_{\text{forecast}}(\mathbf{x}_0),\quad
\mathcal{L}_{\text{forecast}} = \|\hat{\text{stats}} - \text{stats}\|_2^2
$$

#### 4.3.4 总损失

$$
\mathcal{L} = \lambda_1 \mathcal{L}_{\text{noise}} +
              \lambda_2 \mathcal{L}_{\text{cls}} +
              \lambda_3 \mathcal{L}_{\text{forecast}}
$$

$\lambda_1, \lambda_2, \lambda_3$ 为超参数，可通过验证集调整。

---

### 4.4 不确定性估计与早停策略

#### 4.4.1 不确定性估计（多样本扩散）

给定当前前缀 $\mathcal{E}_{\text{prefix}}^{(k)}$，计算 $\mathbf{c}^{(k)}$，从高斯噪声出发进行多次采样：

$$
\mathbf{x}_T^{(s)} \sim \mathcal{N}(0,\mathbf{I}),
\quad
\mathbf{x}_0^{(s)} \sim p_\theta(\mathbf{x}_0|\mathbf{x}_T^{(s)}, \mathbf{c}^{(k)}),\quad s=1,\dots,S
$$

得到一组谣言概率：

$$
p^{(s)} = g_\phi(\mathbf{x}_0^{(s)})
$$

计算：

* 平均概率：

  $$
  \bar{p}_k = \frac{1}{S}\sum_{s=1}^S p^{(s)}
  $$
* 不确定性度量（如方差）：

  $$
  u_k = \frac{1}{S}\sum_{s=1}^S (p^{(s)} - \bar{p}_k)^2
  $$

也可使用分位数估计置信区间。

#### 4.4.2 早停策略

给定阈值 $\tau_0 < \tau_1$ 与不确定性阈值 $\gamma$：

* 若 $\bar{p}_k > \tau_1$ 且 $u_k < \gamma$：高置信度“谣言”，立即报警；
* 若 $\bar{p}_k < \tau_0$ 且 $u_k < \gamma$：高置信度“非谣言”，无需报警；
* 否则：等待更多交互（增大 k），重新决策。

可进一步定义成本函数：

* $C_{\text{FP}}$：误报成本（正常消息被当谣言）
* $C_{\text{FN}}$：漏报成本（谣言未被及时发现）
* $C_{\text{delay}}$：每等待一次新交互的延迟成本

通过验证集网格搜索 $\tau_0,\tau_1,\gamma$，使得期望总成本最小。

---

### 4.5 图不完备鲁棒性设计

1. **结构扰动训练**

在训练阶段，对 $G_{\text{full}}, G_{\text{prefix}}^{(k)}$ 引入随机结构丢弃：

* 边丢弃率 $p_e$：随机删除一定比例边；
* 节点丢弃率 $p_v$：随机屏蔽部分节点特征或节点。

2. **图可观测度编码**

构造一个图完整度标量 $r \in [0,1]$，表示“当前可观测结构的比例”，作为额外条件输入到 prefix encoder / 去噪网络中：

$$
\mathbf{c}' = \text{Fuse}(\mathbf{c}, r)
$$

使模型学习在“图稠密 vs 图稀疏”两种 regime 下不同的决策策略。

3. **实验中显式评估图缺失鲁棒性**

在测试时人为控制图缺失率，从而画出性能随图缺失率变化的曲线，比较 RumorDiffusion 和图基线（BiGCN, HeterGraph Transformer 等）的下降速度。

---

## 5. 训练与推理流程

### 5.1 训练算法（伪代码）

**Algorithm 1: Training RumorDiffusion**

1. 输入：训练集 $\{\mathcal{E}^{(n)}, G^{(n)}, y^{(n)}, \text{stats}^{(n)}\}$
2. 对每个 mini-batch：

   1. 对每条样本随机采样 prefix 长度 $k$（可从集合 $\{5,10,20,\dots\}$ 中采样）；
   2. 构造 $\mathcal{E}_{\text{prefix}}^{(k)}$, $G_{\text{prefix}}^{(k)}$，并施加结构扰动；
   3. 通过 prefix encoder 得到 $\mathbf{c}^{(k)}$；
   4. 通过 full encoder 得到 $\mathbf{h}_0$，线性映射得 $\mathbf{x}_0$；
   5. 采样时间步 $t$、高斯噪声 $\boldsymbol{\epsilon}$，构造 $\mathbf{x}_t$；
   6. 用条件去噪网络预测噪声 $\hat{\boldsymbol{\epsilon}}_\theta$，计算 $\mathcal{L}_{\text{noise}}$；
   7. 在 $\mathbf{x}_0$ 上计算 $\mathcal{L}_{\text{cls}}, \mathcal{L}_{\text{forecast}}$；
   8. 按总损失 $\mathcal{L}$ 反向传播更新 $\theta, \phi$ 等参数。

### 5.2 推理算法（流式早期检测）

**Algorithm 2: Online Early Rumor Detection**

对每条新消息：

1. 维护当前前缀 $\mathcal{E}_{\text{prefix}}^{(k)}$ 与 $G_{\text{prefix}}^{(k)}$；
2. 当有新交互到达时，更新前缀，计算新的 $\mathbf{c}^{(k)}$；
3. 多次采样 $\mathbf{x}_0^{(s)}$，计算 $\bar{p}_k, u_k$；
4. 根据阈值规则决定：报警 / 通过 / 继续等待；
5. 记录当前 k 作为“检测所需交互数”，用于评估早期检测延迟。

---

## 6. 实验设计

### 6.1 数据集与预处理

* **Twitter15 / Twitter16**：英文、多事件谣言数据，带转发树与时间戳；
* **PHEME**：跨事件谣言数据集，含对话结构与时间信息；
* **Weibo（Sina）谣言数据集**：中文社交平台谣言数据，包含转发/评论链与时间；
* 可选：**FakeNewsNet**（Politifact / GossipCop）扩展场景，用前 k 条互动构造早期子集。

**早期划分策略**（统一在各数据集上使用）：

* 按时间排序的前缀长度 $k \in \{5,10,20, \text{all}\}$；
* 或按传播深度的前 10% / 20% 节点子树；
* 对每个 k 评估模型表现，绘制性能–前缀长度曲线。

### 6.2 对比基线

1. **文本单模态**

   * BERT / RoBERTa / DeBERTa + 时间位置编码。

2. **结构 / 序列早期方法**

   * RvNN、TCNN、HAN、dEFEND、SAFE；
   * UPFD / HPN（用户画像 + 图结构）。

3. **图与异质图模型**

   * BiGCN / BiGAT；
   * HeterGraph Transformer / HeterSat；
   * PT-HGA（预训练异质图注意力）。

4. **生成 / 扩散视角对照**

   * 条件于文本/图特征的 DDPM/DDIM + 分类头（无 prefix-conditioning 设计）；
   * 简单时间序列生成模型（GP、VAE）辅助特征。

5. **消融版本 RumorDiffusion**

   * 无扩散：仅 prefix encoder + classifier；
   * 无未来统计辅助任务；
   * 无结构扰动 / 无图完整度编码；
   * 单样本采样（不做 uncertainty & early stopping）。

### 6.3 评测指标

1. **分类性能**

   * Accuracy, Precision, Recall, F1 (macro/micro)；
   * ROC-AUC。

2. **早期检测相关指标**

   * 不同前缀长度 k 下的 F1、AUC 曲线；
   * **Detection latency**：在达到给定性能阈值（如 F1 ≥ 0.8）时所需的平均交互数；
   * **Cost-aware metrics**：在设定 $C_{\text{FP}},C_{\text{FN}},C_{\text{delay}}$ 下的平均成本。

3. **不确定性与校准**

   * Brier score；
   * Calibration curve / Expected Calibration Error (ECE)。

4. **图缺失鲁棒性**

   * 在不同图缺失率下的性能曲线；
   * 对比 RumorDiffusion 与图基线的性能衰减。

5. **效率**

   * 推理时间（每条消息） vs 扩散步数 T；
   * 内存占用，吞吐量。

---

## 7. 消融与分析计划

1. **Prefix-conditioned diffusion 的作用**

   * 比较：无扩散 / 非条件扩散 / prefix 条件扩散；
   * 分析：在极小前缀（k=5）时的性能差异。

2. **未来传播统计预测的作用**

   * 去掉 $\mathcal{L}_{\text{forecast}}$；
   * 比较未来传播规模/深度预测误差与最终分类性能。

3. **不确定性与早停策略**

   * 单样本预测 vs 多样本采样；
   * 是否使用不确定性 early stopping；
   * 绘制“平均检测交互数–F1” trade-off 曲线。

4. **图扰动与完整度编码的作用**

   * 无结构扰动 vs 有扰动训练；
   * 无 $r$ 编码 vs 有 $r$ 编码；
   * 在图严重缺失情形下的表现差异。

5. **不同组件选择**

   * GNN 类型（GCN/GAT/Graph Transformer）；
   * 文本 encoder（BERT vs RoBERTa）；
   * 条件注入方式（FiLM vs Cross-Attention）。

---

## 8. 预期贡献与可能风险

### 8.1 预期贡献

1. 提出一种 **前缀条件扩散框架**，显式学习“early prefix → full cascade latent”的后验映射，用于早期谣言检测；
2. 将扩散模型的多样采样用于 **不确定性估计与早停决策**，在误报/漏报/延迟成本之间提供清晰的可调权衡；
3. 通过结构扰动和图可观测度编码系统性研究 **图不完备场景下的鲁棒性**，贴近真实平台约束；
4. 在多数据集、多早期划分下给出系统实验，为后续 early rumor detection 研究提供统一基准与评测框架。

### 8.2 可能风险与应对

1. **扩散模型训练成本较高**

   * 通过 DDIM / 步数裁剪 / 蒸馏减少推理步数；
   * 先在低维 latent（如 64/128 维）上训练。

2. **早期前缀极端稀疏时提升有限**

   * 结合更强的用户画像与先验知识（如可信媒体列表）；
   * 尝试引入外部知识（例如 fact-checking 结果的粗粒度标签）。

3. **不确定性估计可能偏乐观或偏保守**

   * 对比多种不确定性度量（方差、温度缩放、MC dropout 等）；
   * 用校准方法（如温度缩放、Dirichlet calibration）对 $\bar{p}$ 做后处理。

---

## 9. 进度规划（可按实际时间调整）

* **阶段 1（0–2 个月）**：

  * 完成数据清洗与早期划分 pipeline；
  * 实现强基线（BERT、BiGCN、UPFD 等）并复现实验。

* **阶段 2（2–4 个月）**：

  * 实现 prefix encoder / full encoder & 条件扩散框架；
  * 完成基础版 RumorDiffusion（含 $\mathcal{L}_{\text{noise}} + \mathcal{L}_{\text{cls}}$）。

* **阶段 3（4–6 个月）**：

  * 加入未来传播统计预测头与 $\mathcal{L}_{\text{forecast}}$；
  * 引入结构扰动和图完整度编码；
  * 设计并实现不确定性估计与 early stopping 策略。

* **阶段 4（6–8 个月）**：

  * 系统化消融实验与鲁棒性分析；
  * 调优效率（步数、DDIM、batching、混合精度）；
  * 整理结果、绘图、撰写论文。

---

这份方案已经把问题设定、核心创新、模型结构、训练/推理流程、实验与消融、以及实际平台约束下的影响都串联在一起。你可以直接在此基础上补充相关工作部分和正式的引言，就可以往 KDD 的论文结构上靠。
