"""
DecoupledConditionEncoder — v1 base + DecAlign-style OT alignment (hete_loss only).
Keeps original cosine decouple + common_align + unique_margin losses.
Adds GMM prototypes + Multi-Marginal OT on unique features (no MMD/homo).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledConditionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        cfg_condition = cfg.MODEL.get('CONDITION_MODEL', {})
        self.token_dim = int(cfg_condition.get('TOKEN_DIM', 512))
        hidden_dim = int(cfg_condition.get('HIDDEN_DIM', 256))
        dropout = float(cfg_condition.get('DROPOUT', 0.1))
        self.use_prompt = bool(cfg_condition.get('USE_PROMPT_TOKEN', True))

        radar_input_dim = int(cfg.MODEL.PRE_PROCESSOR.INPUT_DIM)
        lidar_input_dim = int(cfg.MODEL.get('PRE_PROCESSOR2', cfg.MODEL.PRE_PROCESSOR).INPUT_DIM)
        self.radar_token_proj = nn.Linear(radar_input_dim, self.token_dim)
        self.lidar_token_proj = nn.Linear(lidar_input_dim, self.token_dim)

        self.common_encoder = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.ReLU(),
        )
        self.unique_img_encoder = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.ReLU(),
        )
        self.unique_lidar_encoder = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.ReLU(),
        )
        self.unique_radar_encoder = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.ReLU(),
        )

        self.common_fusion = nn.Sequential(
            nn.Linear(self.token_dim * 4 if self.use_prompt else self.token_dim * 3, self.token_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.token_dim, self.token_dim),
            nn.LayerNorm(self.token_dim),
        )
        self.unique_refine = nn.Sequential(
            nn.Linear(self.token_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.token_dim),
            nn.LayerNorm(self.token_dim),
        )
        self.image_token_fusion = nn.Sequential(
            nn.Linear(self.token_dim * 2, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.ReLU(),
        )

        # ── OT alignment (DecAlign-style, only hete, no MMD) ──
        self.ot_enabled = bool(cfg_condition.get('OT_ENABLED', True))
        if self.ot_enabled:
            self.align_dim = int(cfg_condition.get('ALIGN_DIM', 64))
            self.align_proj_unique = nn.Linear(self.token_dim, self.align_dim)
            self.num_prototypes = int(cfg_condition.get('NUM_PROTOTYPES', 8))
            self.proto_img   = nn.Parameter(torch.randn(self.num_prototypes, self.align_dim))
            self.proto_lidar = nn.Parameter(torch.randn(self.num_prototypes, self.align_dim))
            self.proto_radar = nn.Parameter(torch.randn(self.num_prototypes, self.align_dim))
            self.logvar_img   = nn.Parameter(torch.zeros(self.num_prototypes, self.align_dim))
            self.logvar_lidar = nn.Parameter(torch.zeros(self.num_prototypes, self.align_dim))
            self.logvar_radar = nn.Parameter(torch.zeros(self.num_prototypes, self.align_dim))
            self.ot_reg       = float(cfg_condition.get('OT_REG', 0.1))
            self.ot_num_iters = int(cfg_condition.get('OT_NUM_ITERS', 50))

    def _pool_sparse_tokens(self, features, indices, batch_size):
        pooled = []
        for batch_idx in range(batch_size):
            mask = indices[:, 0] == batch_idx
            if mask.any():
                pooled.append(features[mask].mean(dim=0))
            else:
                pooled.append(torch.zeros(features.shape[1], device=features.device, dtype=features.dtype))
        return torch.stack(pooled, dim=0)

    # ──────── OT helpers ────────
    def _compute_prototypes(self, features, proto, logvar):
        N, d = features.shape
        diff = features.unsqueeze(1) - proto.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2) / d
        return F.softmax(-dist_sq, dim=1)

    def _pairwise_cost(self, mu1, logvar1, mu2, logvar2, eps=1e-9):
        K = mu1.shape[0]
        d = mu1.shape[1]
        diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=2) / d
        sigma1, sigma2 = torch.exp(logvar1), torch.exp(logvar2)
        cov_term = torch.sum(
            sigma1.unsqueeze(1) + sigma2.unsqueeze(0)
            - 2 * torch.sqrt(sigma1.unsqueeze(1) * sigma2.unsqueeze(0) + eps), dim=2) / d
        return dist_sq + cov_term

    def _multi_marginal_sinkhorn(self, C, nu_i, nu_l, nu_r, reg, num_iters=50, eps=1e-9):
        K_tensor = torch.exp(-C / reg)
        u, v, w = torch.ones_like(nu_i), torch.ones_like(nu_l), torch.ones_like(nu_r)
        for _ in range(num_iters):
            u = nu_i / (torch.sum(K_tensor * v.view(1,-1,1) * w.view(1,1,-1), dim=(1,2)) + eps)
            v = nu_l / (torch.sum(K_tensor * u.view(-1,1,1) * w.view(1,1,-1), dim=(0,2)) + eps)
            w = nu_r / (torch.sum(K_tensor * u.view(-1,1,1) * v.view(1,-1,1), dim=(0,1)) + eps)
        T = (u.view(-1,1,1) * v.view(1,-1,1) * w.view(1,1,-1)) * K_tensor
        ot_loss = torch.sum(T * C) + 0.001 * reg * (-torch.sum(T * torch.log(T + eps)))
        return T, ot_loss

    def compute_hetero_loss(self, s_img, s_lidar, s_radar):
        s_i = self.align_proj_unique(s_img)
        s_l = self.align_proj_unique(s_lidar)
        s_r = self.align_proj_unique(s_radar)
        K, d, eps = self.num_prototypes, self.align_dim, 1e-9

        w_i = self._compute_prototypes(s_i, self.proto_img, self.logvar_img)
        w_l = self._compute_prototypes(s_l, self.proto_lidar, self.logvar_lidar)
        w_r = self._compute_prototypes(s_r, self.proto_radar, self.logvar_radar)
        nu_i = w_i.mean(dim=0); nu_i = nu_i / (nu_i.sum() + eps)
        nu_l = w_l.mean(dim=0); nu_l = nu_l / (nu_l.sum() + eps)
        nu_r = w_r.mean(dim=0); nu_r = nu_r / (nu_r.sum() + eps)

        cost_il = self._pairwise_cost(self.proto_img, self.logvar_img, self.proto_lidar, self.logvar_lidar)
        cost_ir = self._pairwise_cost(self.proto_img, self.logvar_img, self.proto_radar, self.logvar_radar)
        cost_lr = self._pairwise_cost(self.proto_lidar, self.logvar_lidar, self.proto_radar, self.logvar_radar)
        C = cost_il.unsqueeze(2) + cost_ir.unsqueeze(1) + cost_lr.unsqueeze(0)
        _, ot_loss = self._multi_marginal_sinkhorn(C, nu_i, nu_l, nu_r, reg=self.ot_reg, num_iters=self.ot_num_iters)

        loss_il = torch.mean(w_i * torch.sum((s_i.unsqueeze(1) - self.proto_lidar.unsqueeze(0))**2, dim=2)) / d
        loss_ir = torch.mean(w_i * torch.sum((s_i.unsqueeze(1) - self.proto_radar.unsqueeze(0))**2, dim=2)) / d
        loss_li = torch.mean(w_l * torch.sum((s_l.unsqueeze(1) - self.proto_img.unsqueeze(0))**2, dim=2)) / d
        loss_lr = torch.mean(w_l * torch.sum((s_l.unsqueeze(1) - self.proto_radar.unsqueeze(0))**2, dim=2)) / d
        loss_ri = torch.mean(w_r * torch.sum((s_r.unsqueeze(1) - self.proto_img.unsqueeze(0))**2, dim=2)) / d
        loss_rl = torch.mean(w_r * torch.sum((s_r.unsqueeze(1) - self.proto_lidar.unsqueeze(0))**2, dim=2)) / d
        return ot_loss + loss_il + loss_ir + loss_li + loss_lr + loss_ri + loss_rl

    # ──────── Forward (v1 original + OT) ────────
    def forward(self, dict_item):
        batch_size = int(dict_item['batch_size'])

        img_token = dict_item.get('img_embedding', None)
        camera_global_token = dict_item.get('camera_global_token', None)
        if img_token is not None and camera_global_token is not None:
            img_token = self.image_token_fusion(torch.cat((img_token, camera_global_token), dim=-1))
        elif img_token is None and camera_global_token is not None:
            img_token = camera_global_token
        elif img_token is None:
            device = dict_item['sp_features'].device
            img_token = torch.zeros((batch_size, self.token_dim), device=device)

        prompt_token = dict_item.get('prompt_weather_token', None)
        if prompt_token is None:
            prompt_token = torch.zeros_like(img_token)

        radar_token = self._pool_sparse_tokens(dict_item['sp_features'], dict_item['sp_indices'], batch_size)
        lidar_token = self._pool_sparse_tokens(dict_item['sp_features_l'], dict_item['sp_indices_l'], batch_size)
        radar_token = self.radar_token_proj(radar_token)
        lidar_token = self.lidar_token_proj(lidar_token)

        common_img = self.common_encoder(img_token)
        common_radar = self.common_encoder(radar_token)
        common_lidar = self.common_encoder(lidar_token)
        common_prompt = self.common_encoder(prompt_token) if self.use_prompt else None

        common_tokens = [common_img, common_radar, common_lidar]
        if self.use_prompt:
            common_tokens.append(common_prompt)
        common_input = torch.cat(common_tokens, dim=-1)
        condition_common = self.common_fusion(common_input)

        unique_img_raw = self.unique_img_encoder(img_token)
        unique_radar_raw = self.unique_radar_encoder(radar_token)
        unique_lidar_raw = self.unique_lidar_encoder(lidar_token)

        unique_img = self.unique_refine(torch.cat((unique_img_raw, condition_common), dim=-1))
        unique_radar = self.unique_refine(torch.cat((unique_radar_raw, condition_common), dim=-1))
        unique_lidar = self.unique_refine(torch.cat((unique_lidar_raw, condition_common), dim=-1))

        dict_item['condition_common_token'] = condition_common
        dict_item['condition_unique_img'] = unique_img
        dict_item['condition_unique_radar'] = unique_radar
        dict_item['condition_unique_lidar'] = unique_lidar
        dict_item['condition_common_img'] = common_img
        dict_item['condition_common_radar'] = common_radar
        dict_item['condition_common_lidar'] = common_lidar
        dict_item['condition_token'] = condition_common

        # ── OT hete_loss (new) ──
        if self.ot_enabled:
            dict_item['condition_unique_img_raw'] = unique_img_raw
            dict_item['condition_unique_radar_raw'] = unique_radar_raw
            dict_item['condition_unique_lidar_raw'] = unique_lidar_raw
            dict_item['condition_hete_loss'] = self.compute_hetero_loss(
                unique_img_raw, unique_lidar_raw, unique_radar_raw)

        return dict_item
