from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp import autocast


class ArbiterMLP(nn.Module):
    """Two-branch arbiter MLP operating on concatenated potentials.

    Input shape: [B, 2*H] where H is hidden_channels of CTDM backbone.
    Output: [B, C] logits.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PotentialDiscriminator(nn.Module):
    """MLP discriminator operating on single-class potential vectors.

    Input shape: [B, H] where H is hidden_channels of CTDM backbone.
    Output: [B, C] logits.
    """

    def __init__(self, hidden_channels: int, num_classes: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RumorDiffusion(nn.Module):
    """RumorDiffusion with Transformer-based arbiter over class potentials.

    The module consumes per-sample cached potentials (h_cat or h0/h1) when
    provided, or will query CTDM to obtain all-class potentials Hc.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_classes: int,
        arbiter_hidden: int = 256,
        dropout: float = 0.3,
        ctdm: Optional[nn.Module] = None,
        freeze_ctdm: bool = True,
        arbiter_type: str = "transformer",
        arbiter_readout: str = "cls",
        token_norm: str = "layernorm",
        token_dropout: float = 0.3,
        transformer_nhead: int = 4,
        transformer_nlayers: int = 2,
        transformer_dim_feedforward: int = 512,
        transformer_dropout: float = 0.3,
        # DD-GAN configuration
        dd_gan_enabled: bool = False,
        lambda_gan: float = 0.0,
        lr_d: float = 1.0e-4,
        lambda_r1: float = 10.0,
        # Pair/token and auxiliary head configuration
        use_pair_token: bool = False,
        pair_ids: Tuple[int, int] = (1, 2),
        pair_token_mode: str = "diff",
        # True/False margin loss
        pair_margin_enabled: bool = False,
        pair_margin_m: float = 0.5,
        lambda_pair_margin: float = 0.0,
        enable_pair_margin_loss: bool = True,
        # Auxiliary TF binary head
        tf_head_enabled: bool = False,
        lambda_tf_bce: float = 0.0,
        tf_infer_adjust_alpha: float = 0.0,
        enable_tf_head_loss: bool = True,
        # Class embedding configuration
        use_class_embedding: bool = False,
        class_embed_mode: str = "concat_proj",
        # DD-GAN gating
        enable_dd_gan: bool = True,
        # Inference/efficiency
        cache_potentials: bool = False,
        infer_amp: bool = False,
        uncertainty_force_ctdm: bool = False,
        # Diagnostics
        loss_logging: bool = False,
    ) -> None:
        super().__init__()
        self.ctdm = ctdm
        if freeze_ctdm and self.ctdm is not None:
            for p in self.ctdm.parameters():
                p.requires_grad = False
        # Token-level normalization for potential vectors
        self.hidden_channels = int(hidden_channels)
        self.num_classes = int(num_classes)
        self.arbiter_type = str(arbiter_type).lower()
        self.arbiter_readout = str(arbiter_readout).lower()
        assert self.arbiter_readout in ["cls", "token"], f"arbiter_readout must be 'cls' or 'token', got {self.arbiter_readout}"
        self.cache_potentials = bool(cache_potentials)
        self.infer_amp = bool(infer_amp)
        self.uncertainty_force_ctdm = bool(uncertainty_force_ctdm)
        self.loss_logging = bool(loss_logging)
        norm_choices = {
            "layernorm": nn.LayerNorm(self.hidden_channels),
            "none": nn.Identity(),
        }
        self.token_norm = norm_choices.get(str(token_norm).lower(), norm_choices["layernorm"])
        self.token_drop = nn.Dropout(p=float(token_dropout))
        # Pair/token and aux-head flags
        self.use_pair_token = bool(use_pair_token)
        self.pair_ids = (int(pair_ids[0]), int(pair_ids[1]))
        self.pair_token_mode = str(pair_token_mode).lower()
        self.pair_margin_enabled = bool(pair_margin_enabled)
        self.pair_margin_m = float(pair_margin_m)
        self.lambda_pair_margin = float(lambda_pair_margin)
        self.enable_pair_margin_loss = bool(enable_pair_margin_loss)
        self.tf_head_enabled = bool(tf_head_enabled)
        self.lambda_tf_bce = float(lambda_tf_bce)
        self.tf_infer_adjust_alpha = float(tf_infer_adjust_alpha)
        self.enable_tf_head_loss = bool(enable_tf_head_loss)
        # Multi-pair margin loss configuration (for handling multiple confusable class pairs)
        self.multi_pair_margin_list = []
        self.multi_pair_margin_values = []
        self.lambda_multi_pair_margin = 0.0
        self.enable_multi_pair_margin = True
        # [CLS] token for Transformer arbiter
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_channels))
        # Learnable class embeddings to enhance class-awareness in potential vectors (optional)
        self.use_class_embedding = bool(use_class_embedding)
        self.class_embed_mode = str(class_embed_mode).lower()
        if self.use_class_embedding:
            self.class_embedding = nn.Embedding(num_classes, self.hidden_channels)
            # Projection layer to fuse class embedding with potential vector
            self.class_embed_fusion = nn.Linear(self.hidden_channels * 2, self.hidden_channels)
            if self.class_embed_mode == "gated":
                self.class_embed_gate = nn.Linear(self.hidden_channels * 2, self.hidden_channels)
        if self.pair_token_mode == "learnable":
            self.learnable_pair_token = nn.Parameter(torch.zeros(1, 1, self.hidden_channels))
        if self.arbiter_type == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_channels,
                nhead=int(transformer_nhead),
                dim_feedforward=int(transformer_dim_feedforward),
                dropout=float(transformer_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(transformer_nlayers))
            if self.arbiter_readout == "token":
                self.token_arbiter = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.hidden_channels, arbiter_hidden),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=dropout),
                        nn.Linear(arbiter_hidden, 1),
                    ) for _ in range(num_classes)
                ])
            else:
                self.arbiter = ArbiterMLP(
                    in_dim=self.hidden_channels,
                    num_classes=num_classes,
                    hidden=arbiter_hidden,
                    dropout=dropout,
                )
            if self.tf_head_enabled:
                self.tf_head = nn.Linear(self.hidden_channels, 1)
        else:
            # Fallback MLP over concatenated first two potentials if explicitly requested
            self.input_dim = self.hidden_channels * 2
            self.input_norm = nn.LayerNorm(self.input_dim)
            self.input_drop = nn.Dropout(p=dropout)
            self.arbiter = ArbiterMLP(in_dim=self.input_dim, num_classes=num_classes, hidden=arbiter_hidden, dropout=dropout)
        # Loss hyperparameters (can be overridden externally)
        self.la_tau = 1.0
        self.la_gamma = 1.5
        self.la_smooth = 0.05
        self.lambda_brier = 0.01
        self.lambda_consistency = 0.05
        self.lambda_diffusion = 0.0
        # Class priors for logit-adjusted losses
        self.register_buffer("class_priors", torch.ones((num_classes,), dtype=torch.float32) / float(num_classes), persistent=True)
        # Diagnostics buffer
        self._last_loss_breakdown = {}

        # DD-GAN components
        self.dd_gan_enabled = bool(dd_gan_enabled) and bool(enable_dd_gan)
        if self.dd_gan_enabled:
            self.discriminator = PotentialDiscriminator(
                hidden_channels=self.hidden_channels,
                num_classes=self.num_classes,
                hidden=int(arbiter_hidden),
                dropout=float(dropout),
            )
            self.optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(), lr=float(lr_d), weight_decay=1.0e-5
            )
            self.lambda_gan = float(lambda_gan)
            self.lambda_r1 = float(lambda_r1)

    @staticmethod
    def _effective_pair_ids(pair_ids: Tuple[int, int], num_classes: int) -> Tuple[int, int] | Tuple[()]:
        valid = tuple(i for i in pair_ids if 0 <= int(i) < int(num_classes))
        if len(valid) >= 2:
            return (int(valid[0]), int(valid[1]))
        if num_classes >= 2:
            return (0, 1)
        return tuple()

    def _encode_from_tokens(self, tokens: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Encodes a sequence of tokens via Transformer and returns (logits, cls_out).

        If pair token injection is enabled and compatible with token layout,
        appends a differential token between configured class indices.
        Optionally fuses learnable class embeddings with each potential vector to enhance class-awareness.
        """
        B = int(tokens.size(0))
        H = int(tokens.size(-1))
        # Optional fusion of learnable class embeddings with each potential vector
        if self.use_class_embedding:
            num_classes = int(self.num_classes)
            class_embs = self.class_embedding(torch.arange(num_classes, device=tokens.device)).unsqueeze(0).expand(B, -1, -1)
            if int(tokens.size(1)) >= num_classes:
                tokens_with_class = tokens[:, :num_classes, :]
                class_embs = class_embs[:, :int(tokens_with_class.size(1)), :]
            else:
                class_embs = class_embs[:, :int(tokens.size(1)), :]
            tokens_slice = tokens[:, :int(class_embs.size(1)), :]
            fused_tokens = torch.cat([tokens_slice, class_embs], dim=-1)
            if self.class_embed_mode == "gated":
                proj = self.class_embed_fusion(fused_tokens)
                gate = torch.sigmoid(self.class_embed_gate(fused_tokens))
                tokens = gate * proj + (1.0 - gate) * tokens_slice
            else:
                tokens = self.class_embed_fusion(fused_tokens)
        # Optional pair token: requires tokens to include class-aligned positions
        pair_ids_eff = self._effective_pair_ids(self.pair_ids, int(self.num_classes))
        if self.use_pair_token and len(pair_ids_eff) == 2:
            p0, p1 = pair_ids_eff
            pair_token = None
            if self.pair_token_mode == "learnable":
                pair_token = self.learnable_pair_token.expand(B, 1, H)
            else:
                if int(tokens.size(1)) > max(p0, p1):
                    d_tf = tokens[:, p0, :] - tokens[:, p1, :]
                    pair_token = d_tf.unsqueeze(1)
            if pair_token is not None:
                tokens = torch.cat([tokens, pair_token], dim=1)
        cls = self.cls_token.expand(B, 1, H)
        seq = torch.cat([cls, self.token_drop(self.token_norm(tokens))], dim=1)
        out = self.transformer_encoder(seq)
        
        if self.arbiter_readout == "token":
            cls_out = out[:, 0, :]
            token_out = out[:, 1:, :]
            num_token_classes = min(int(token_out.size(1)), int(self.num_classes))
            logits_list = []
            for c in range(int(self.num_classes)):
                if c < num_token_classes:
                    logit_c = self.token_arbiter[c](token_out[:, c, :]).view(B, 1)
                else:
                    logit_c = torch.zeros(B, 1, device=token_out.device, dtype=token_out.dtype)
                logits_list.append(logit_c)
            logits = torch.cat(logits_list, dim=1)
        else:
            cls_out = out[:, 0, :]
            logits = self.arbiter(cls_out)
        
        # Optional inference-time adjustment for class pair using TF head
        if self.tf_head_enabled and self.tf_infer_adjust_alpha > 0.0 and len(pair_ids_eff) == 2:
            tf_logit = self.tf_head(cls_out).view(-1)
            p = torch.sigmoid(tf_logit).view(-1, 1)
            p0, p1 = pair_ids_eff
            logits[:, p0] = logits[:, p0] + float(self.tf_infer_adjust_alpha) * p.view(-1)
            logits[:, p1] = logits[:, p1] + float(self.tf_infer_adjust_alpha) * (1.0 - p.view(-1))
        return logits, cls_out

    def _build_tokens_from_batch(self, batch) -> Tensor:
        """Builds per-sample token sequence [B, S, H] from batch or CTDM."""
        if hasattr(batch, "h_cat"):
            x = batch.h_cat
            B = int(x.size(0))
            H = self.hidden_channels
            K = int(x.size(1)) // H
            return x.view(B, K, H)
        if hasattr(batch, "h0") and hasattr(batch, "h1"):
            return torch.stack([batch.h0, batch.h1], dim=1)
        assert self.ctdm is not None, "CTDM is required when potentials are not present in the batch"
        Hc = self._compute_ctdm_potentials(batch)
        self._last_potentials = Hc
        return Hc

    def _logits_from_tokens(self, tokens: Tensor) -> Tensor:
        """Maps potential tokens to class logits for both arbiter variants."""
        if self.arbiter_type == "transformer":
            logits, _cls = self._encode_from_tokens(tokens)
            return logits
        x = torch.cat([tokens[:, 0, :], tokens[:, 1, :]], dim=-1)
        x = self.input_drop(self.input_norm(x))
        return self.arbiter(x)

    def _forward_from_ctdm(self, batch) -> Tensor:
        """Forces on-the-fly CTDM potential inference regardless of cached fields."""
        assert self.ctdm is not None, "CTDM is required for sampled uncertainty inference"
        Hc = self._compute_ctdm_potentials(batch)
        self._last_potentials = Hc
        return self._logits_from_tokens(Hc)

    def _compute_ctdm_potentials(self, batch) -> Tensor:
        assert self.ctdm is not None, "CTDM is required when potentials are not present in the batch"
        ctdm_trainable = self.training and any(p.requires_grad for p in self.ctdm.parameters())
        if ctdm_trainable:
            return self.ctdm.potentials_all_grad(batch)
        return self.ctdm.infer_potentials_all(batch)

    def forward(self, batch) -> Tensor:
        """Build token sequence and arbitrate via Transformer [CLS] output by default."""
        def _compute():
            if self.arbiter_type == "transformer":
                tokens = self._build_tokens_from_batch(batch)
                return self._logits_from_tokens(tokens)
            # Fallback MLP over two potentials if explicitly requested
            if hasattr(batch, "h_cat"):
                x = batch.h_cat  # [B, K*H]
                B = int(x.size(0))
                H = self.hidden_channels
                x = x[:, : 2 * H]
                x = self.input_drop(self.input_norm(x))
                return self.arbiter(x)
            if hasattr(batch, "h0") and hasattr(batch, "h1"):
                x = torch.cat([batch.h0, batch.h1], dim=-1)
                x = self.input_drop(self.input_norm(x))
                return self.arbiter(x)
            assert self.ctdm is not None, "CTDM is required when potentials are not present in the batch"
            return self._forward_from_ctdm(batch)

        if (not self.training) and self.infer_amp:
            with autocast():
                return _compute()
        return _compute()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        batch,
        num_samples: int = 4,
        rumor_classes: Optional[List[int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Runs multi-sample inference and returns (mean_prob, var_prob).

        If rumor_classes provided, returns aggregated rumor probability/variance
        of shape [B, 1]; otherwise returns per-class tensors [B, C].
        """
        device = next(self.parameters()).device
        batch = batch.to(device)
        samples = max(1, int(num_samples))
        probs = []
        for _ in range(samples):
            if self.uncertainty_force_ctdm and self.ctdm is not None:
                if (not self.training) and self.infer_amp:
                    with autocast():
                        logits = self._forward_from_ctdm(batch)
                else:
                    logits = self._forward_from_ctdm(batch)
            else:
                if (not self.training) and self.infer_amp:
                    with autocast():
                        logits = self(batch)
                else:
                    logits = self(batch)
            probs.append(F.softmax(logits, dim=1))
        prob_stack = torch.stack(probs, dim=0)  # [S, B, C]
        mean_prob = prob_stack.mean(dim=0)
        var_prob = prob_stack.var(dim=0, unbiased=False)
        if rumor_classes is None:
            return mean_prob, var_prob
        mask = torch.zeros((mean_prob.size(1),), device=mean_prob.device, dtype=mean_prob.dtype)
        for cid in rumor_classes:
            if 0 <= int(cid) < mask.numel():
                mask[int(cid)] = 1.0
        rumor_prob = (mean_prob * mask.view(1, -1)).sum(dim=1, keepdim=True)
        rumor_var = (var_prob * mask.view(1, -1)).sum(dim=1, keepdim=True)
        return rumor_prob, rumor_var

    @staticmethod
    def _tf_margin_loss_static(logits: Tensor, targets: Tensor, pair_ids: Tuple[int, int], margin: float) -> Tensor:
        """Computes margin loss between logits of a configured class pair.

        Applies only to samples whose targets belong to the specified pair.
        """
        p0, p1 = int(pair_ids[0]), int(pair_ids[1])
        y = targets.view(-1)
        mask = (y == p0) | (y == p1)
        if not mask.any():
            return logits.new_zeros(())
        z = logits[mask]
        y_masked = y[mask]
        # margin for y==p0: m - (z_p0 - z_p1), for y==p1: m - (z_p1 - z_p0)
        z0 = z[:, p0]
        z1 = z[:, p1]
        diff = torch.where(y_masked == p0, z0 - z1, z1 - z0)
        loss = F.relu(float(margin) - diff)
        return loss.mean()

    @staticmethod
    def _multi_pair_margin_loss(logits: Tensor, targets: Tensor, pair_list: list, margin_list: list) -> Tensor:
        """Computes margin loss for multiple class pairs.

        Generalizes pair margin loss to handle arbitrary class pairs for better discrimination
        between confusable categories. Each pair can have its own margin threshold.

        Args:
            logits: Classification logits of shape [B, C].
            targets: Target class indices of shape [B].
            pair_list: List of tuples, each containing (class_id_0, class_id_1) pairs.
            margin_list: List of margin values corresponding to each pair.

        Returns:
            Aggregated margin loss across all pairs.
        """
        if not pair_list or len(pair_list) == 0:
            return logits.new_zeros(())
        
        total_loss = 0.0
        num_pairs = len(pair_list)
        y = targets.view(-1)
        
        for pair_ids, margin in zip(pair_list, margin_list):
            p0, p1 = int(pair_ids[0]), int(pair_ids[1])
            mask = (y == p0) | (y == p1)
            if not mask.any():
                continue
            z = logits[mask]
            y_masked = y[mask]
            z0 = z[:, p0]
            z1 = z[:, p1]
            diff = torch.where(y_masked == p0, z0 - z1, z1 - z0)
            loss = F.relu(float(margin) - diff)
            total_loss = total_loss + loss.mean()
        
        return total_loss / max(1, num_pairs)

    @staticmethod
    def _label_smoothing_targets(targets: Tensor, num_classes: int, epsilon: float) -> Tensor:
        with torch.no_grad():
            y = torch.zeros((targets.size(0), num_classes), device=targets.device, dtype=torch.float32)
            y.scatter_(1, targets.view(-1, 1), 1.0)
            y = (1.0 - epsilon) * y + epsilon / float(num_classes)
            return y

    def _la_focal_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        assert logits.dim() == 2 and targets.dim() == 1
        B, C = logits.shape
        priors = self.class_priors.to(logits.device).clamp_min(1e-6)
        z_tilde = logits + self.la_tau * priors.log().view(1, -1)
        log_p = F.log_softmax(z_tilde, dim=1)
        p = log_p.exp()
        y_s = self._label_smoothing_targets(targets, C, self.la_smooth)
        ce = -(y_s * log_p).sum(dim=1)
        p_y = p[torch.arange(B, device=logits.device), targets]
        mod = (1.0 - p_y).clamp_min(0.0).pow(self.la_gamma)
        return (mod * ce).mean()

    @staticmethod
    def _brier_loss(logits: Tensor, targets: Tensor) -> Tensor:
        B, C = logits.shape
        p = F.softmax(logits, dim=1)
        y = torch.zeros((B, C), device=logits.device, dtype=torch.float32)
        y.scatter_(1, targets.view(-1, 1), 1.0)
        return torch.mean((p - y) ** 2)

    @staticmethod
    def _symmetric_kl(p: Tensor, q: Tensor) -> Tensor:
        eps = 1e-8
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        kl_pq = (p * (p / q).log()).sum(dim=1)
        kl_qp = (q * (q / p).log()).sum(dim=1)
        return (kl_pq + kl_qp).mean()

    def training_step(self, batch) -> Tuple[Tensor, float]:
        """Performs one training step and returns (loss, aux_acc).

        Computes the primary classification objective and auxiliary
        regularization terms. When adversarial regularization is enabled,
        updates the discriminator and augments the main loss accordingly.
        """
        targets = batch.y.view(-1).long()
        loss_dict = {}
        diffusion_loss: Optional[Tensor] = None
        if self.ctdm is not None and float(self.lambda_diffusion) > 0.0:
            has_trainable = any(p.requires_grad for p in self.ctdm.parameters())
            if has_trainable:
                diffusion_loss, _ = self.ctdm.training_step(batch)

        if self.dd_gan_enabled:
            assert self.ctdm is not None, "CTDM is required when dd_gan_enabled=True"
            Hc = self._compute_ctdm_potentials(batch)
            self._last_potentials = Hc

            B, C, H = int(Hc.size(0)), int(Hc.size(1)), int(Hc.size(2))
            Hc_detached = Hc.detach().view(B * C, H)
            Hc_detached.requires_grad_(True)
            y_disc = torch.arange(C, device=Hc.device).repeat(B)

            # Step A: discriminator update
            self.discriminator.train()
            for p in self.discriminator.parameters():
                p.requires_grad = True
            with autocast(enabled=False):
                self.optimizer_d.zero_grad(set_to_none=True)
                logits_d = self.discriminator(Hc_detached.float())
                loss_d_ce = F.cross_entropy(logits_d, y_disc)
                grad_real = torch.autograd.grad(
                    outputs=logits_d.sum(),
                    inputs=Hc_detached,
                    create_graph=True,
                    only_inputs=True,
                )[0]
                r1_penalty = grad_real.pow(2).reshape(B * C, -1).sum(dim=1).mean()
                loss_d = loss_d_ce + (float(self.lambda_r1) / 2.0) * r1_penalty
                loss_d.backward()
                self.optimizer_d.step()

            # Step B: generator + classifier
            self.discriminator.eval()
            for p in self.discriminator.parameters():
                p.requires_grad = False

            if self.arbiter_type == "transformer":
                logits_main, cls_out = self._encode_from_tokens(Hc)
            else:
                x = torch.cat([Hc[:, 0, :], Hc[:, 1, :]], dim=-1)
                x = self.input_drop(self.input_norm(x))
                logits_main = self.arbiter(x)
                cls_out = None

            loss_focal = self._la_focal_loss(logits_main, targets)
            loss = loss_focal
            loss_dict["la_focal"] = float(loss_focal.detach().item())

            pair_ids_eff = self._effective_pair_ids(self.pair_ids, int(self.num_classes))
            if self.enable_pair_margin_loss and self.pair_margin_enabled and float(self.lambda_pair_margin) > 0.0:
                if len(pair_ids_eff) == 2:
                    loss_tf_margin = self._tf_margin_loss_static(logits_main, targets, pair_ids_eff, self.pair_margin_m)
                    loss = loss + float(self.lambda_pair_margin) * loss_tf_margin
                    loss_dict["pair_margin"] = float(loss_tf_margin.detach().item())

            if self.enable_multi_pair_margin and len(self.multi_pair_margin_list) > 0 and float(self.lambda_multi_pair_margin) > 0.0:
                pair_filtered = []
                margin_filtered = []
                for pair_ids, margin in zip(self.multi_pair_margin_list, self.multi_pair_margin_values):
                    if len(pair_ids) == 2 and max(pair_ids) < int(self.num_classes):
                        pair_filtered.append(pair_ids)
                        margin_filtered.append(margin)
                if len(pair_filtered) > 0:
                    loss_multi_margin = self._multi_pair_margin_loss(logits_main, targets, pair_filtered, margin_filtered)
                    loss = loss + float(self.lambda_multi_pair_margin) * loss_multi_margin
                    loss_dict["multi_pair_margin"] = float(loss_multi_margin.detach().item())

            if self.enable_tf_head_loss and self.tf_head_enabled and float(self.lambda_tf_bce) > 0.0 and cls_out is not None and len(pair_ids_eff) == 2:
                tf_logit = self.tf_head(cls_out).view(-1)
                p0, p1 = pair_ids_eff
                mask = (targets == p0) | (targets == p1)
                if mask.any():
                    tf_targets = (targets[mask] == p0).float()
                    tf_loss = F.binary_cross_entropy_with_logits(tf_logit[mask], tf_targets)
                    loss = loss + float(self.lambda_tf_bce) * tf_loss
                    loss_dict["tf_bce"] = float(tf_loss.detach().item())

            logits_g = self.discriminator(Hc.view(B * C, H))
            loss_g = F.cross_entropy(logits_g, y_disc)
            loss = loss + float(self.lambda_gan) * loss_g
            loss_dict["gan"] = float(loss_g.detach().item())

            if float(self.lambda_brier) > 0.0:
                loss_brier = self._brier_loss(logits_main, targets)
                loss = loss + float(self.lambda_brier) * loss_brier
                loss_dict["brier"] = float(loss_brier.detach().item())

            if self.ctdm is not None and float(self.lambda_consistency) > 0.0:
                obs_backup = int(self.ctdm.obs_minutes)
                min_c = int(getattr(self.ctdm, "cutoff_min", 10))
                max_c = int(getattr(self.ctdm, "cutoff_max", 30))
                t1 = int(torch.randint(low=min_c, high=max_c + 1, size=(1,)).item())
                t2 = int(torch.randint(low=min_c, high=max_c + 1, size=(1,)).item())
                self.ctdm.obs_minutes = t1
                logits_v1 = self(batch)
                self.ctdm.obs_minutes = t2
                logits_v2 = self(batch)
                self.ctdm.obs_minutes = obs_backup
                p1 = F.softmax(logits_v1, dim=1)
                p2 = F.softmax(logits_v2, dim=1)
                loss_cons = self._symmetric_kl(p1, p2)
                loss = loss + float(self.lambda_consistency) * loss_cons
                loss_dict["consistency"] = float(loss_cons.detach().item())

            if self.ctdm is not None:
                if hasattr(self, "_last_potentials"):
                    Hc_cached = self._last_potentials  # [B, C, H]
                    idx = torch.arange(Hc_cached.size(0), device=Hc_cached.device)
                    h_y = Hc_cached[idx, targets, :]
                    if float(getattr(self.ctdm, "lambda_proto", 0.0)) > 0.0:
                        proto_loss = self.ctdm._compute_prototype_contrastive_loss(h_y, targets)
                        loss = loss + float(self.ctdm.lambda_proto) * proto_loss
                        self.ctdm.update_prototypes(h_y.detach(), targets.detach())
                        loss_dict["proto"] = float(proto_loss.detach().item())
                scores = getattr(self.ctdm, "_last_scores", None)
                A_obs = getattr(self.ctdm, "_last_Aobs_batch", None)
                X_obs_last = getattr(self.ctdm, "_last_xobs", None)
                if scores is not None and A_obs is not None and X_obs_last is not None:
                    topo_bce, topo_kl = self.ctdm._topology_regularizers(scores, A_obs, X_obs_last)
                    if float(getattr(self.ctdm, "lambda_topo_bce", 0.0)) > 0.0:
                        loss = loss + float(self.ctdm.lambda_topo_bce) * topo_bce
                        loss_dict["topo_bce"] = float(topo_bce.detach().item())
                    if float(getattr(self.ctdm, "lambda_deg_kl", 0.0)) > 0.0:
                        loss = loss + float(self.ctdm.lambda_deg_kl) * topo_kl
                        loss_dict["topo_kl"] = float(topo_kl.detach().item())

            if diffusion_loss is not None:
                loss = loss + float(self.lambda_diffusion) * diffusion_loss
                loss_dict["diffusion"] = float(diffusion_loss.detach().item())

            for p in self.discriminator.parameters():
                p.requires_grad = True
            self.discriminator.train()
            loss_dict["total"] = float(loss.detach().item())
            if self.loss_logging:
                self._last_loss_breakdown = loss_dict
            return loss, 0.0

        logits = self(batch)
        loss_focal = self._la_focal_loss(logits, targets)
        loss = loss_focal
        loss_dict["la_focal"] = float(loss_focal.detach().item())

        pair_ids_eff = self._effective_pair_ids(self.pair_ids, int(self.num_classes))
        if self.enable_pair_margin_loss and self.pair_margin_enabled and float(self.lambda_pair_margin) > 0.0:
            if len(pair_ids_eff) == 2:
                loss_tf_margin = self._tf_margin_loss_static(logits, targets, pair_ids_eff, self.pair_margin_m)
                loss = loss + float(self.lambda_pair_margin) * loss_tf_margin
                loss_dict["pair_margin"] = float(loss_tf_margin.detach().item())

        if self.enable_multi_pair_margin and len(self.multi_pair_margin_list) > 0 and float(self.lambda_multi_pair_margin) > 0.0:
            pair_filtered = []
            margin_filtered = []
            for pair_ids, margin in zip(self.multi_pair_margin_list, self.multi_pair_margin_values):
                if len(pair_ids) == 2 and max(pair_ids) < int(self.num_classes):
                    pair_filtered.append(pair_ids)
                    margin_filtered.append(margin)
            if len(pair_filtered) > 0:
                loss_multi_margin = self._multi_pair_margin_loss(logits, targets, pair_filtered, margin_filtered)
                loss = loss + float(self.lambda_multi_pair_margin) * loss_multi_margin
                loss_dict["multi_pair_margin"] = float(loss_multi_margin.detach().item())

        if self.enable_tf_head_loss and self.tf_head_enabled and float(self.lambda_tf_bce) > 0.0 and self.arbiter_type == "transformer":
            tokens = self._build_tokens_from_batch(batch)
            _logits_tmp, cls_out = self._encode_from_tokens(tokens)
            p0, p1 = self.pair_ids
            mask = (targets == p0) | (targets == p1)
            if mask.any() and len(self._effective_pair_ids(self.pair_ids, int(self.num_classes))) == 2:
                tf_logit = self.tf_head(cls_out).view(-1)
                tf_targets = (targets[mask] == p0).float()
                tf_loss = F.binary_cross_entropy_with_logits(tf_logit[mask], tf_targets)
                loss = loss + float(self.lambda_tf_bce) * tf_loss
                loss_dict["tf_bce"] = float(tf_loss.detach().item())

        if float(self.lambda_brier) > 0.0:
            loss_brier = self._brier_loss(logits, targets)
            loss = loss + float(self.lambda_brier) * loss_brier
            loss_dict["brier"] = float(loss_brier.detach().item())

        if self.ctdm is not None and float(self.lambda_consistency) > 0.0:
            obs_backup = int(self.ctdm.obs_minutes)
            min_c = int(getattr(self.ctdm, "cutoff_min", 10))
            max_c = int(getattr(self.ctdm, "cutoff_max", 30))
            t1 = int(torch.randint(low=min_c, high=max_c + 1, size=(1,)).item())
            t2 = int(torch.randint(low=min_c, high=max_c + 1, size=(1,)).item())
            self.ctdm.obs_minutes = t1
            logits_v1 = self(batch)
            self.ctdm.obs_minutes = t2
            logits_v2 = self(batch)
            self.ctdm.obs_minutes = obs_backup
            p1 = F.softmax(logits_v1, dim=1)
            p2 = F.softmax(logits_v2, dim=1)
            loss_cons = self._symmetric_kl(p1, p2)
            loss = loss + float(self.lambda_consistency) * loss_cons
            loss_dict["consistency"] = float(loss_cons.detach().item())

        if self.ctdm is not None:
            if hasattr(self, "_last_potentials"):
                Hc = self._last_potentials  # [B, C, H]
                idx = torch.arange(Hc.size(0), device=Hc.device)
                h_y = Hc[idx, targets, :]
                if float(getattr(self.ctdm, "lambda_proto", 0.0)) > 0.0:
                    proto_loss = self.ctdm._compute_prototype_contrastive_loss(h_y, targets)
                    loss = loss + float(self.ctdm.lambda_proto) * proto_loss
                    self.ctdm.update_prototypes(h_y.detach(), targets.detach())
                    loss_dict["proto"] = float(proto_loss.detach().item())
            scores = getattr(self.ctdm, "_last_scores", None)
            A_obs = getattr(self.ctdm, "_last_Aobs_batch", None)
            X_obs_last = getattr(self.ctdm, "_last_xobs", None)
            if scores is not None and A_obs is not None and X_obs_last is not None:
                topo_bce, topo_kl = self.ctdm._topology_regularizers(scores, A_obs, X_obs_last)
                if float(getattr(self.ctdm, "lambda_topo_bce", 0.0)) > 0.0:
                    loss = loss + float(self.ctdm.lambda_topo_bce) * topo_bce
                    loss_dict["topo_bce"] = float(topo_bce.detach().item())
                if float(getattr(self.ctdm, "lambda_deg_kl", 0.0)) > 0.0:
                    loss = loss + float(self.ctdm.lambda_deg_kl) * topo_kl
                    loss_dict["topo_kl"] = float(topo_kl.detach().item())

        if diffusion_loss is not None:
            loss = loss + float(self.lambda_diffusion) * diffusion_loss
            loss_dict["diffusion"] = float(diffusion_loss.detach().item())
        loss_dict["total"] = float(loss.detach().item())
        if self.loss_logging:
            self._last_loss_breakdown = loss_dict
        return loss, 0.0
