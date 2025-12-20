### 1. 模型输入与输出（I/O Specification）

#### 1.1 任务定义

* **主任务：早期谣言检测（cascade-level 二分类）**
  给定一条消息在传播早期的若干交互前缀 $\mathcal{E}_{\text{prefix}}^{(k)}$ 及其局部传播图 $G_{\text{prefix}}^{(k)}$，预测该消息是否为谣言：

  $$
  f_\theta: (\mathcal{E}_{\text{prefix}}^{(k)}, G_{\text{prefix}}^{(k)}) \rightarrow \hat{p} \in [0,1]
  $$

  其中 $\hat{p}=P(y=1 \mid \text{prefix})$，$y\in\{0,1\}$ 为真值标签（1=谣言）。

* **辅助任务：未来传播统计预测（回归）**
  在训练时，利用完整传播 $\mathcal{E}_{\text{full}}$ 得到 latent $\mathbf{x}_0$，预测未来传播总体统计：

  $$
  \hat{\text{stats}} = g_{\text{forecast}}(\mathbf{x}_0) \approx \text{stats}
  $$

  例如 total\_size, max\_depth 等。

* **生成任务（扩散模型预训练目标）**
  在训练时，对 full latent $\mathbf{x}_0$ 施加噪声 $\mathbf{x}_t$，在 prefix 条件 $\mathbf{c}$ 下预测噪声 $\hat{\epsilon}_\theta$，优化：

  $$
  \mathcal{L}_{\text{noise}} = \mathbb{E}\|\epsilon - \hat{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})\|_2^2
  $$

---

#### 1.2 输入（Input）

> 下述为**单个 batch** 的张量定义，PyTorch 风格。符号：
>
> * $B$：batch size
> * $M$：每个 cascade 的最大事件数（转发/评论数，padding）
> * $L$：单条文本最大 token 长度
> * $D_{\text{text}}$：文本 encoder 隐层维度（默认 768, BERT-base）
> * $D_{\text{user}}$：用户原始特征维度（如 16–64）
> * $D_{\text{time}}$：时间嵌入维度（默认 16）
> * $D_{\text{stats}}$：统计特征维度（如 2–4）

##### 1.2.1 必要输入（训练 & 推理共有）

1. **事件文本 token**

   * `event_input_ids`: `[B, M, L]`, `int64`，取值范围 `[0, V-1]`（词表大小 V），padding 为 0。
   * `event_attention_mask`: `[B, M, L]`, `bool` / `int64` in `{0,1}`。1 表示有效 token，0 表示 padding。

2. **用户特征**

   * `user_features_raw`: `[B, M, D_user_raw]`, `float32`，原始统计特征（粉丝数、关注数、注册时长等）。
   * **预处理/归一化（数据层完成）**：

     * 对长尾分布（粉丝数等）做 `log(1+x)`；
     * 每个维度按训练集统计做 `z-score` 标准化：

       $$
       x' = \frac{x - \mu}{\sigma}
       $$
   * 模型实际输入：`user_features`: `[B, M, D_user]`, `float32`，经过一层 `Linear + ReLU` 映射到统一维度。

3. **时间戳**

   * `timestamps`: `[B, M]`, `float64`/`float32`，单位为秒/毫秒（数据集原始格式）。
   * **对齐/归一化策略**：

     * 统一转为秒；
     * 对每个 cascade 内部做相对时间：

       $$
       t_i' = t_i - t_{\text{root}}
       $$
     * 再做 `log(1+t_i')` 或 min-max 归一到 `[0,1]`。
   * 模型内部通过 `TimeEnc` 转为时间嵌入 `time_emb`: `[B, M, D_time]`, `float32`。

4. **传播结构（转发/回复图）**

   * 为兼容 graph 库，推荐使用稀疏表示（PyG 风格）：

     * `edge_index`: `[2, E_total]`, `int64`，第一维表示源/目标节点索引（相对当前 batch 的展平节点），范围 `[0, B*M-1]`；
     * `batch_index`: `[B*M]`, `int64`，每个节点所属 cascade id，范围 `[0,B-1]`。
   * 每个 cascade 内事件索引与 `[B,M]` 对齐；无边时可使用空张量。

5. **前缀掩码（区分早期前缀与完整事件集）**

   * `prefix_mask`: `[B, M]`, `bool` / `int64` in `{0,1}`。

     * 1：该事件已在当前前缀中观测到；
     * 0：未观测到（仅训练时用于 full encoder）。
   * `event_mask`: `[B, M]`, `bool`，表示 padding 之外的有效事件。

6. **图可观测度（结构完整度）**

   * `graph_observed_ratio`: `[B, 1]`, `float32`，范围 `[0,1]`。

     * 1：图结构近似完整；
     * 0–1：图存在不同程度缺失（训练时可设为采样的保留率）。

##### 1.2.2 训练专用输入

1. **完整标签**

   * `labels`: `[B]`, `int64` or `float32`。

     * 二分类：`0` (非谣言), `1` (谣言)。

2. **未来传播统计**

   * `stats`: `[B, D_stats]`, `float32`。
     示例（可按数据集定义 2–4 维）：

     * `stats[...,0] = log(1 + total_size)`
     * `stats[...,1] = max_depth`
     * `stats[...,2] = log(1 + total_retweets)` 等。

3. **训练时随机采样的 diffusion 时间步**

   * 时间步 t 本身不作为外部输入，由模型内部采样：

     * `t`: `[B]`, `int64`，范围 `[1, T]`，均匀采样。
   * 对应的时间嵌入 `t_emb`: `[B, D_t]`, `float32`，内部生成。

##### 1.2.3 推理阶段的输入差异

* **推理时不需要：**

  * `labels`, `stats`（无监督信息）；
  * 完整后缀事件（真实在线场景下未知）。
* **推理时输入仅包含：**

  * 当前已观测 prefix 的事件及其图结构；
  * `prefix_mask` 将未到达的事件全部置 0；
  * `graph_observed_ratio` 可由平台估计（例如 “目前只看到了转发树，不含 follow 图”，设为 0.5）。

---

#### 1.3 输出（Output）

##### 1.3.1 训练阶段输出（forward 结果）

1. **谣言概率（主输出）**

   * `logits`: `[B, 1]`, `float32`。

     * 未经过 sigmoid 的 logit；用于 `BCEWithLogitsLoss`。
   * `prob`: `[B, 1]`, `float32`，`sigmoid(logits)` 结果，仅用于监控与评估。

   与损失/指标关系：

   * 对应损失：$\mathcal{L}_{\text{cls}}$，使用 BCE/CE：

     $$
     \mathcal{L}_{\text{cls}} = -\frac{1}{B}\sum_i \big( y_i\log \hat{p}_i + (1-y_i)\log (1-\hat{p}_i) \big)
     $$
   * 对应指标：Accuracy, Precision, Recall, F1, ROC-AUC。

2. **噪声预测（扩散任务输出）**

   * `eps_pred`: `[B, D_x]`, `float32`，预测噪声 $\hat{\epsilon}_\theta$。
   * `eps_true`: `[B, D_x]`, `float32`（内部构建），真实高斯噪声 $\epsilon$。

   损失：

   $$
   \mathcal{L}_{\text{noise}} = \frac{1}{B}\sum_i \|\epsilon_i - \hat{\epsilon}_{\theta,i}\|_2^2
   $$

3. **传播统计预测**

   * `stats_pred`: `[B, D_stats]`, `float32`。

   损失：

   $$
   \mathcal{L}_{\text{forecast}} = \frac{1}{B}\sum_i \|\hat{\text{stats}}_i - \text{stats}_i\|_2^2
   $$

4. **可选：latent 表征（用于调试/分析）**

   * `x0`: `[B, D_x]`, `float32`，full encoder + projector 得到的 latent。
   * 不直接参与损失（除非用于 forecast/head / cls）。

##### 1.3.2 推理阶段输出

1. **单次采样模式（简单评估）**

   * `prob`: `[B, 1]`, `float32`；
   * `pred_label`: `[B]`, `int64` (`0/1`)，阈值默认为 0.5 或在验证集上调参得到 $\tau^\*$：

     * `pred_label = (prob >= tau).long()`。

2. **多次采样 + 不确定性（完整 RumorDiffusion）**

   * 采样次数 `S`：超参，默认 8–16。

   * `prob_samples`: `[S, B, 1]`, `float32`；

   * `prob_mean`: `[B, 1]`, `float32`，各样本均值：

     $$
     \bar{p}_k = \frac{1}{S}\sum_s p_k^{(s)}
     $$

   * `prob_var`: `[B, 1]`, `float32`，方差：

     $$
     u_k = \frac{1}{S}\sum_s (p_k^{(s)} - \bar{p}_k)^2
     $$

   * `decision`: `[B]`, `int64` in `{0,1,2}`（可选编码）：

     * `0`：预测为非谣言（$\bar{p}_k<\tau_0, u_k<\gamma$）；
     * `1`：预测为谣言（$\bar{p}_k>\tau_1, u_k<\gamma$）；
     * `2`：不确定，继续等待更多交互。

   与早停策略、延迟指标的对应：

   * 在不同 k 下记录首次 `decision!=2` 对应的交互数，计算平均 detection latency；
   * 在该决策点上计算 Accuracy/F1 等指标。

---

### 2. 模型框架与组件（Architecture & Components）

#### 2.1 整体结构与数据流（从输入到输出）

**训练阶段单次 forward 的数据流：**

1. **原始 batch → 事件级特征**

   * 文本：`event_input_ids` & `attention_mask` → TextEncoder → `text_emb`: `[B,M,D_text]`
   * 用户：`user_features_raw` → 归一化 → MLP → `user_emb`: `[B,M,D_user]`
   * 时间：`timestamps` → TimeEnc → `time_emb`: `[B,M,D_time]`
   * 合并：`event_feat = concat(text_emb, user_emb, time_emb)`

     * 形状：`[B, M, D_event]`, `D_event = D_text + D_user + D_time`

2. **构建图表示**

   * 使用 `edge_index` & `batch_index` 将 `event_feat` 视为图节点特征：

     * 展平为：`x = event_feat.view[B*M, D_event]`；
     * 送入 GraphEncoder(GNN)。

3. **GraphEncoder 输出**

   * 得到图上下文特征 `graph_emb`: `[B*M, D_gnn]`；
   * reshape 回 `[B,M,D_gnn]`。

4. **Prefix/Full encoder 分支**

   * 使用 `prefix_mask` & `event_mask`：

     * **Prefix 分支**：

       * 只对 `prefix_mask==1` 的节点做 masked pooling/attention：
         → `c = f_prefix( event_feat, graph_emb, prefix_mask )`
         → `c`: `[B, D_c]`。
     * **Full 分支**（仅训练）：

       * 对 `event_mask==1` 的全部节点做图 pooling：
         → `h0 = f_full( event_feat, graph_emb, event_mask )`
         → `h0`: `[B, D_h]`；
       * `x0 = W * h0` → `[B, D_x]`。

5. **Diffusion 前向过程（训练内部）**

   * 采样 `t` & `epsilon`；
   * 计算 `x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t)*epsilon` → `[B, D_x]`；
   * 使用 `c` & `graph_observed_ratio` 组成条件向量 `cond`，拼入 Diffusion Network。

6. **去噪网络（条件扩散核心）**

   * 输入：`x_t`: `[B, D_x]`，时间嵌入 `t_emb`: `[B, D_t]`，条件 `cond`: `[B, D_cond]`；
   * 合并后经多层 MLP/Transformer → 输出 `eps_pred`: `[B, D_x]`。

7. **任务头**

   * Classification head：`x0` → `logits`: `[B,1]`；
   * Forecast head：`x0` → `stats_pred`: `[B,D_stats]`。

8. **损失计算 & 反向传播**

   * `L_noise`, `L_cls`, `L_forecast` 按权重加权 → `L_total`。

**推理阶段：**

* Full 分支停用，仅执行步骤 1–4 的 prefix encoder；
* 不再有 `x0` from full encoder，需要通过 **逆向扩散采样**从 `x_T` 迭代到 `x_0^{(s)}`，再走 classification head。

---

#### 2.2 核心组件清单

##### 2.2.1 数据与特征层

1. **文本预处理**

   * 分词器：与选用预训练模型一致（BERT tokenizer等）；
   * 小写/清洗 URL/emoji 可选；
   * 限制 `L`（如 64/128），超长截断，短文本 padding；
   * 特殊 token：`[CLS]` / `[SEP]` 由 tokenizer 处理。

2. **用户特征工程**

   * 数值特征：粉丝数、关注数、微博数、注册时间等；
   * 类别特征：认证类型、性别等用 one-hot/embedding 映射；
   * 统一为 `user_features_raw: float32`；
   * 数据层做标准化：保存 `mean/std` 到配置。

3. **时间特征**

   * 将原始时间戳（UNIX 时间）转换为相对时间（相对 root 消息）；
   * 做 `log(1 + ∆t)` 缩放；
   * TimeEnc 实现：

     * 选用 Time2Vec 或正弦位置编码：

       $$
       \text{TimeEnc}(t) = [\sin(\omega_1 t), \cos(\omega_1 t), \dots]
       $$

4. **图构建**

   * 使用转发/评论关系构建有向树/图：

     * 节点：事件（转发/评论）；
     * 边：`parent → child`；
   * 若有 follow 图，可加入用户层边，构为异质图（复现最低限度时可先只用转发树）；
   * 统一构建 `edge_index` & `batch_index`。

---

##### 2.2.2 主体网络

**(A) TextEncoder**

* 选型：`bert-base-uncased` 或对应中文模型 `bert-base-chinese`；
* 输出：`[B,M,L,D_text]` → 取 `[CLS]` 或平均池化 → `[B,M,D_text]`；
* 默认策略：

  * 冻结前 N 层，只微调最后 K 层 + pooler（如前 8 冻结，后 4 可训练）；
  * dropout `p=0.1`。

**(B) UserEncoder**

* 2 层 MLP：

  * 输入：`[B,M,D_user_raw]`；
  * 隐层：`Linear(D_user_raw → 2*D_user)` + ReLU；
  * 输出层：`Linear(2*D_user → D_user)` + LayerNorm；
* 默认 `D_user = 64`。

**(C) TimeEncoder**

* Time2Vec 或 Sinusoidal：

  * `D_time = 16`；
  * 输出 `[B,M,D_time]`。

**(D) GraphEncoder**

* 选型：

  * 轻量复现可选 2–3 层 GAT/GCN；
  * 若使用 Graph Transformer 则更重，但更强。

* 输入：`node_feat = concat(text_emb, user_emb, time_emb)` → `[B*M,D_event]`

* 超参默认：

  * `num_layers = 2`；
  * `hidden_dim = D_gnn = 256`；
  * `num_heads = 4`（对 GAT/Transformer）；
  * 激活：ReLU，dropout=0.1。

* 输出：`graph_emb`: `[B*M,D_gnn]`。

**(E) Prefix/Full Encoder**

* 基本思路：在 `event_feat` 和 `graph_emb` 上做 masked pooling：

  * 先 fuse：

    $$
      z_i = \text{MLP}([ \text{text}_i \Vert \text{user}_i \Vert \text{time}_i \Vert \text{graph}_i ])
    $$

    输出 `z`: `[B,M,D_z]`（默认 `D_z=256`）。

* Prefix encoder：

  * 对 `prefix_mask` 有效节点做 attention pooling：

    * 学习 query 向量 $\mathbf{q}\in\mathbb{R}^{D_z}$；
    * 注意力权重：

      $$
      \alpha_i = \text{softmax}_i(\mathbf{q}^\top z_i + m_i)
      $$

      其中 `m_i` 对非 prefix 节点设为 $-\infty$（mask）；
    * 聚合：

      $$
      c = \sum_i \alpha_i z_i
      $$
  * 输出 `c`: `[B,D_c]`（默认 `D_c=256`）。

* Full encoder：

  * 类似 attention pooling，但 mask 使用 `event_mask`；
  * 输出 `h0`: `[B,D_h]`，默认 `D_h=256`。

**(F) Latent Projector**

* 简单 MLP：

  * `Linear(D_h → D_x)` + LayerNorm；
  * 默认 `D_x = 256`；
* 输出 `x0`: `[B, D_x]`。

**(G) Diffusion 去噪网络**

* 时间步 T & 噪声调度：

  * 训练步数 `T_train = 1000`（线性 schedule 可复现性高）；
  * 线性 β：

    $$
    \beta_t \in [1e-4, 0.02],\ t=1,\dots,T
    $$
  * 预计算 $\alpha_t, \bar{\alpha}_t$。

* 时间嵌入：

  * `t`: `[B]` → 经过正弦位置编码 + MLP → `t_emb`: `[B,D_t]`，`D_t=256`。

* 条件向量：

  * 由 `c` 与 `graph_observed_ratio` 拼接：

    * `cond = concat(c, r)` → `[B, D_c+1]`；
    * 再过 `Linear(D_c+1 → D_cond)`，默认 `D_cond=256`。

* 去噪网络结构（1D MLP/小 Transformer）：

  * 输入拼接：

    * `x_input = concat(x_t, t_emb, cond)` → `[B, D_x + D_t + D_cond]`；

  * 3–4 层 MLP：

    * 结构：`Linear → GELU → LayerNorm`，每层残差连接；
    * 中间宽度：`hidden_dim = 512`。

  * 输出：`eps_pred`: `[B,D_x]`。

  > 若想更贴近 UNet 风格，可把 `x` 视为长度为 1 的 “序列”，但复现最低限度时 MLP 足够。

---

##### 2.2.3 任务头（Heads）

1. **Classification Head**

   * 输入：`x0`: `[B,D_x]`；
   * 结构：

     * `Linear(D_x → D_x)` + GELU；
     * Dropout `p=0.1`；
     * `Linear(D_x → 1)`；
   * 输出：`logits`: `[B,1]`。

2. **Forecast Head**

   * 输入：`x0` 或 `h0`;
   * 结构：

     * `Linear(D_x → D_x)` + ReLU；
     * `Linear(D_x → D_stats)`；
   * 输出：`stats_pred`: `[B,D_stats]`。

---

##### 2.2.4 损失函数与权重

> 为便于复现，统一（重新）编号公式，便于文档引用及代码注释。

* **(Eq.1) 噪声预测损失**：

  $$
  \mathcal{L}_{\text{noise}} =
  \mathbb{E}_{t,\mathbf{x}_0,\epsilon,k}\big[
    \|\epsilon - \hat{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}^{(k)})\|_2^2
  \big]
  $$

* **(Eq.2) 谣言分类损失**（二分类 BCE）：

  $$
  \mathcal{L}_{\text{cls}} =
  -\frac{1}{B}\sum_{i=1}^B
   \big(
     y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)
   \big)
  $$

  其中 $\hat{p}_i = \sigma(\text{logits}_i)$。

* **(Eq.3) 未来传播统计损失**：

  $$
  \mathcal{L}_{\text{forecast}} =
  \frac{1}{B}\sum_{i=1}^B
    \|\hat{\text{stats}}_i - \text{stats}_i\|_2^2
  $$

* **(Eq.4) 总损失**：

  $$
  \mathcal{L}_{\text{total}} =
   \lambda_1 \mathcal{L}_{\text{noise}} +
   \lambda_2 \mathcal{L}_{\text{cls}} +
   \lambda_3 \mathcal{L}_{\text{forecast}}
  $$

* **默认权重（可调）**：

  * $\lambda_1 = 1.0$
  * $\lambda_2 = 1.0$
  * $\lambda_3 = 0.1$

---

##### 2.2.5 优化与训练策略

1. **优化器**

   * AdamW；
   * `lr_text = 1e-5`（文本 encoder 的较小 LR）；
   * `lr_other = 2e-4`（其余模块）；
   * `weight_decay = 0.01`。

2. **学习率调度**

   * 线性 warmup + cosine decay：

     * warmup 比例：前 10% 的 total steps；
     * 剩余 90% 做 cosine 下降至 1e-6。

3. **Batch / Accumulation**

   * 由于 TextEncoder 比较重，建议：

     * `batch_size = 16–32`（视显存）；
     * 若过小，采用 gradient accumulation 以保证 `effective_batch_size >= 64`。

4. **梯度与正则化**

   * 梯度裁剪：`max_norm = 1.0`；
   * Dropout：

     * TextEncoder 内置 dropout；
     * MLP & GNN 内 dropout=0.1；
   * 可选 Label Smoothing：0.0–0.1（非必须）。

5. **混合精度与分布式**

   * 建议开启 AMP（PyTorch autocast）；
   * 多 GPU 使用 DDP，确保：

     * 所有随机数（Python/NumPy/Torch/CUDA）固定 seed；
     * 数据划分与 shuffle 受 `DistributedSampler` 控制。

---

##### 2.2.6 推理与后处理

1. **逆向扩散采样（推理）**

   * 设置推理步数 `T_infer` 小于训练步数（如 50）；
   * 使用 DDIM 子采样加速：

     * 从 `x_T^s ~ N(0,I)` 开始；
     * 对 t 从 T 到 1 以 stride 更新（例如使用 50 个关键时间步）；
     * 每一步调用 `eps_pred = Model(x_t, t, c)`，按 DDIM 公式更新 `x_{t-1}`。

2. **早期检测评估模式**

   * 对固定的 k（如 5/10/20），构造 batch 的 `prefix_mask`；
   * 对每个前缀：

     * 采样 `S` 条 trajectory（默认 8）；
     * 得到 `prob_samples` → `prob_mean` & `prob_var`；
     * 使用单一阈值 `tau` 或双阈值 `tau_0, tau_1` 做分类；
   * 评估 Accuracy/F1/AUC，并记录 detection latency（对于 early stopping 模式）。

3. **早停策略（在线推理）**

   * 给定阈值 $\tau_0 < \tau_1$ 和不确定性阈值 $\gamma$：

     * 若 `prob_mean > tau_1 and prob_var < gamma` → 预测谣言（报警）；
     * 若 `prob_mean < tau_0 and prob_var < gamma` → 预测非谣言；
     * 否则 → 返回“不确定”，需等待下一条交互；
   * 阈值在验证集上通过搜索确定，目标是平衡 F1 与平均检测交互数。

---

##### 2.2.7 数据集与划分

1. **数据来源与许可**

   * Twitter15 / Twitter16：原论文提供 splits，需遵守 Twitter API 和数据使用条款；
   * PHEME：公开数据，遵守许可证；
   * Weibo：需注意中文谣言数据集的使用协议。

2. **标准划分**

   * 若官方或主流复现代码提供 train/val/test 划分，优先使用；
   * 若未提供，建议：

     * 按事件级（cascade 级）划分（避免数据泄漏）；
     * 典型：70/10/20 或 80/10/10，固定 random seed；
     * 保存划分索引到磁盘，供复现实验使用。

3. **样本统计**

   * 每个数据集需统计：

     * 样本总数（cascade 数量）；
     * 平均 / 中位数 cascade 长度（事件数 M）；
     * 谣言/非谣言的类比例；
     * 不同前缀长度 k 下可用样本数量（有些 cascade 太短，无法提供大 k）。

4. **标注转换**

   * 将原始标签映射到 `{0,1}`；
   * 若存在多类（如 “non-rumor, rumor, unverified” 等），根据论文设定：

     * 二分类版本：将 “rumor”=1，其余=0；
     * 或多分类版本（扩展版本，不在最小复现范围内）。

---

### 3. 复现 To‑do List（可操作清单）

> 每一步包含：**目标 / 要点 / 产出 / 验收标准**，用于严格跟踪复现进度。

---

1. [ ] **获取资料与建立文献包**

   * 目标：收集 RumorDiffusion 论文 PDF、附录、所有可用的补充材料与相关 baseline 论文。
   * 要点：

     * 下载主论文、附录（若有）、作者提供的任何 slides/tech report；
     * 收集早期谣言检测相关工作（RvNN, UPFD, BiGCN, dEFEND 等）；
     * 记录每篇文献的关键信息：任务定义、数据集版本、评测指标。
   * 产出：

     * `docs/` 目录：PDF 与笔记；
     * 一份简短 Markdown/Notion 文档，总结任务设定与评估协议。
   * 验收：

     * 所有引用的 baseline 论文均可追溯；
     * 能在 5 分钟内查到任一公式或实验设置的位置。

---

2. [ ] **环境与依赖初始化**

   * 目标：搭建可复现的开发/实验环境。
   * 要点：

     * 记录 OS、CUDA、Python、PyTorch、transformers、PyG/DGL 等版本；
     * 固定依赖在 `requirements.txt` 或 `environment.yml`；
     * 预留多 GPU/单 GPU 两套配置。
   * 产出：

     * `env/requirements.txt` 或 `env/environment.yml`；
     * 环境检查脚本 `scripts/check_env.py`。
   * 验收：

     * 换一台机器按文档执行即可完成环境搭建；
     * `check_env.py` 通过，打印出版本信息与 GPU 可用性。

---

3. [ ] **项目结构初始化**

   * 目标：定义清晰、可扩展的项目目录与模块边界。
   * 建议结构（可微调，但职责相近）：

     ```text
     RumorDiffusion/
       configs/
         dataset/
         model/
         train/
       data/
         raw/
         processed/
       src/
         datasets/
         features/
         models/
           text/
           graph/
           diffusion/
           heads/
         train/
         eval/
         utils/
       scripts/
         preprocess_*.py
         train_*.sh
         eval_*.sh
       logs/
       checkpoints/
     ```

   * 要点：

     * `configs/` 中所有超参、路径可追踪；
     * `src/models/` 内模块按功能拆分，避免 God-class。
   * 产出：

     * 初始化的 Git 仓库；
     * 模板配置文件（如 `configs/model/rumordiffusion.yaml`）。
   * 验收：

     * `python -m src.train.dummy_run` 能跑通一个 dummy 模型（随机数据）；
     * 代码通过基本 Lint/Format（如 `black/flake8`）。

---

4. [ ] **数据下载与原始校验**

   * 目标：拉取所有目标数据集并确认完整性。
   * 要点：

     * 按各数据集官方说明获取数据（注意隐私与协议）；
     * 校验文件数量、哈希；
     * 将下载脚本化（便于他人复现）。
   * 产出：

     * `scripts/download_*_dataset.sh`；
     * `data/raw/` 目录下各数据集的原始文件。
   * 验收：

     * 重复运行下载脚本不会破坏现有数据；
     * 在 README 中记录每个数据集的来源和下载方式。

---

5. [ ] **数据预处理与早期前缀构造**

   * 目标：从 raw 数据构建统一的中间格式（事件列表 + 图结构 + 标签）。
   * 要点：

     * 解析用户、文本、时间、parent id 等字段；
     * 按 cascade 聚合事件，排序（按时间）；
     * 为每个 cascade 生成：

       * 事件列表（包含 text、user features、time）；
       * 转发/评论图 `edge_index`；
       * 标签 `y` 和传播统计 `stats`；
     * 根据设定的 k（如 5/10/20）预先生成或在 DataLoader 中动态生成 `prefix_mask`。
   * 产出：

     * `data/processed/{dataset_name}.jsonl` 或类似结构；
     * `src/datasets/{dataset_name}.py` 中 Dataset 类草实现（仅返回原始字段）。
   * 验收：

     * 随机 inspect 10 条样本，检查事件顺序、parent 关系、时间戳是否合理；
     * 统计信息与论文/公开实现中的统计基本一致（级别：均值/中位数数量级相当）。

---

6. [ ] **特征工程与规范化**

   * 目标：对文本、用户和时间特征进行统一预处理。
   * 要点：

     * 编写 tokenizer／文本清洗（支持中英）；
     * 对用户数值特征做 log/z-score，并保存 mean/std；
     * 设计 TimeEnc 接口并实现三种模式（raw/log/sin）方便消融；
     * 将这些步骤封装在 `src/features/` 里，对 Dataset 透明。
   * 产出：

     * `src/features/text.py`, `src/features/user.py`, `src/features/time.py`；
     * 更新 Dataset 使其直接输出 ready-to-feed 的 numpy/torch 张量。
   * 验收：

     * 单独运行 `scripts/inspect_features.py`，输出若干样本的特征统计与分布图；
     * 特征分布无明显异常（例如极端长尾未处理）。

---

7. [ ] **Dataset & DataLoader 实现**

   * 目标：实现完整的 PyTorch Dataset/DataLoader 流水线。
   * 要点：

     * Dataset 返回字段：

       * `event_input_ids, attention_mask, user_features_raw, timestamps, edge_index, batch_index, prefix_mask, event_mask, labels, stats`；
     * collate\_fn 负责：

       * 按 batch 对 M/L 做 padding；
       * 合并图结构到统一 `edge_index` & `batch_index`；
       * 生成 `graph_observed_ratio`（训练时随机，推理时根据配置）。
   * 产出：

     * `src/datasets/base.py`, `src/datasets/{dataset}.py`；
     * `src/datasets/collate.py`。
   * 验收：

     * 用 DataLoader 迭代几个 batch，检查 shapes 是否与 I/O 规范一致；
     * 打印 batch 中某个样本的前缀/完整事件，肉眼验证正确。

---

8. [ ] **TextEncoder / UserEncoder / TimeEncoder 实现与单测**

   * 目标：实现三类 encoder，并确保可独立测试。
   * 要点：

     * TextEncoder：封装 transformers 模型，支持“冻结前 N 层”配置；
     * UserEncoder：MLP + LayerNorm；
     * TimeEncoder：Time2Vec 或 sinusoidal。
   * 产出：

     * `src/models/text/bert_encoder.py`；
     * `src/models/features/user_encoder.py`；
     * `src/models/features/time_encoder.py`；
     * 简单单元测试脚本，随机输入下检查输出维度。
   * 验收：

     * 所有 encoder 的 `forward` 能在 CPU/GPU 上跑通；
     * 输出 shapes 与配置完全一致。

---

9. [ ] **GraphEncoder 与 Prefix/Full Encoder 实现**

   * 目标：实现 GNN 和 prefix/full 表征模块。
   * 要点：

     * GraphEncoder 基于 PyG / DGL：

       * 支持 GCN/GAT 切换（方便消融）；
     * PrefixEncoder：

       * 实现 masked attention pooling；
     * FullEncoder：

       * 实现全局图 pooling（可选 mean / max / attention）。
   * 产出：

     * `src/models/graph/graph_encoder.py`；
     * `src/models/graph/prefix_full_encoder.py`。
   * 验收：

     * 随机构建图（不同大小）时 encoder 正常工作；
     * prefix\_mask 生效：当只选部分节点时输出表征明显不同。

---

10. [ ] **Diffusion 模块实现**

* 目标：实现前向/逆向扩散过程和去噪网络。
* 要点：

  * 预计算 β, α, α\_bar 并在配置中记录；
  * 实现 `q_sample(x0, t, eps)`（前向采样）；
  * 实现去噪网络（MLP/Transformer），包含时间和条件嵌入；
  * 注意数值稳定性（LayerNorm/GELU）。
* 产出：

  * `src/models/diffusion/schedules.py`；
  * `src/models/diffusion/denoiser.py`；
  * `src/models/diffusion/utils.py`。
* 验收：

  * 在 toy latent（2D 高斯）上测试，验证训练后能复原简单分布；
  * MSE 损失逐步下降。

---

11. [ ] **任务头与整体 RumorDiffusion 模型封装**

* 目标：将 encoder + diffusion + heads 组装成一个统一的 `RumorDiffusionModel`。
* 要点：

  * `forward_train(batch)`：返回 `loss_dict`（noise/cls/forecast/total）；
  * `forward_infer_prefix(batch, S, T_infer)`：返回 `prob_samples, prob_mean, prob_var`；
  * 对外统一接口，隐藏内部细节。
* 产出：

  * `src/models/rumordiffusion.py`。
* 验收：

  * 随机数据下前向、反向传播无错误；
  * `loss_total` 为标量且能 backward。

---

12. [ ] **训练脚本与日志系统**

* 目标：实现可配置、可恢复的训练管线。
* 要点：

  * 支持从 YAML 读取模型/训练/数据参数；
  * 使用 logging 或 TensorBoard 记录 loss、指标、学习率；
  * 定期保存 checkpoint（含 optimizer、scheduler 状态）；
  * 记录随机种子与关键配置到 log。
* 产出：

  * `src/train/train_rumordiffusion.py`；
  * `scripts/train_rumordiffusion_{dataset}.sh`。
* 验收：

  * 能在单数据集上完整跑完 1–2 个 epoch；
  * 重启训练不会破坏已有权重，且 log 可连续。

---

13. [ ] **评估脚本与指标实现**

* 目标：实现与论文一致的评估指标与协议。
* 要点：

  * Accuracy, Precision, Recall, F1 (macro/micro), ROC-AUC；
  * 对不同前缀 k 计算指标；
  * 对 early stopping 策略计算：

    * 平均 detection latency；
    * 在不同 (tau0,tau1,gamma) 下的性能曲线。
* 产出：

  * `src/eval/metrics.py`；
  * `src/eval/evaluate_prefix.py`；
  * `scripts/eval_rumordiffusion_{dataset}.sh`。
* 验收：

  * 在 toy 示例上手算的指标与程序输出一致；
  * 能输出与论文表格结构匹配的结果文件（csv/json）。

---

14. [ ] **早停策略与不确定性模块实现**

* 目标：实现基于多样本采样的 uncertainty & early stopping。
* 要点：

  * 写成独立模块 `UncertaintyEarlyStopper`，方便调参与替换；
  * 支持网格搜索 (tau0,tau1,gamma)；
  * 记录：（决策时刻 k, 决策类别, 是否正确）。
* 产出：

  * `src/eval/early_stopping.py`。
* 验收：

  * 在构造的简单概率分布上，early stopping 行为符合预期；
  * 输出的 latency 与 F1 随阈值变化方向合理。

---

15. [ ] **基线模型对齐与复现**

* 目标：实现或对接关键 baseline（最少一个文本+一个图模型）以便相对比较。
* 要点：

  * Text-only BERT；
  * 图模型（如 BiGCN/UPFD）；
  * 保证数据预处理与划分一致；
* 产出：

  * `src/models/baselines/bert_baseline.py`；
  * `src/models/baselines/bigcn_baseline.py` 或对接第三方实现；
* 验收：

  * 在小规模训练下，基线性能接近论文报告范围；
  * 能在同一评估脚本下比较 RumorDiffusion 与基线。

---

16. [ ] **完整实验与结果复现**

* 目标：跑完论文主实验与关键消融。
* 要点：

  * 按论文配置运行各数据集上的 main results；
  * 运行消融：无扩散 / 无 forecast / 无结构扰动 / 不同 k；
  * 记录所有实验的 random seed 与配置。
* 产出：

  * `experiments/` 目录：用于每个实验的配置和结果；
  * 结果表格、曲线（早期前缀 vs F1；缺失率 vs 性能）。
* 验收：

  * 与论文表格中的数值差异在可接受范围内（例如 ±1–2 个百分点）；
  * 所有结果可以通过 `config + seed` 完整复现。

---

17. [ ] **差异与风险记录 & 验证计划**

* 目标：系统记录与原论文可能不一致的实现细节及其影响。
* 要点：

  * 记录所有“合理默认”的选择，例如：

    * T\_train/T\_infer 的具体数值；
    * β 调度是否线性还是 cosine；
    * TextEncoder 微调策略（冻结层数）；
    * GNN 类型（GCN vs GAT vs Transformer）；
  * 对每一处差异做：

    * 影响分析（可能使性能略高/略低）；
    * 验证计划（通过小实验估计影响量级）。
* 产出：

  * `docs/diff_and_risks.md`；
* 验收：

  * 任意复现者阅读该文档后，能快速理解本实现与论文的细微差异；
  * 所有差异都有明确说明与（至少粗略的）实验验证。

---

18. [ ] **最终整理与开源准备（如需）**

* 目标：将代码和文档整理为可对外发布的版本。
* 要点：

  * 完成 README（含 quick start / 复现实验指引）；
  * 补充注释与类型标注；
  * 清理无用脚本与临时代码。
* 产出：

  * 整洁的代码仓库；
  * README、LICENSE、CONTRIBUTING 指南（视需求）。
* 验收：

  * 不熟悉项目的同学按 README 能从零跑通一个完整实验；
  * 代码结构与文档达到“可长期维护”的标准。
