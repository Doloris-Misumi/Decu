# DeCU-AvailabilityAware Fusion (DeCU + ASF)
# Combines DeCU's common/unique token decomposition with ASF's
# UCP (patch projection) + CASAP (cross-attn along patches) + SCL (sensor combination loss)

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class DecuASFFusion(nn.Module):
    """Patch-level cross-attention fusion with DeCU token conditioning.

    Replaces global scalar branch weights with per-patch cross-attention
    across sensors, conditioned on DeCU common/unique tokens.
    """

    def __init__(self, branch_feat_dim, head_dim, token_dim=512,
                 patch_size=2, dim_unified=256, n_heads=16,
                 n_repeat_ch=8, bev_h=32, bev_w=180):
        super().__init__()
        self.branch_feat_dim = branch_feat_dim
        self.patch_size = patch_size
        self.dim_unified = dim_unified
        self.n_repeat_ch = n_repeat_ch
        self.bev_h = bev_h
        self.bev_w = bev_w

        # ── UCP: project each sensor's BEV patches to unified space ──
        patch_in_dim = patch_size * patch_size * branch_feat_dim  # 2*2*768=3072

        def make_ucp():
            return nn.Sequential(
                nn.LayerNorm(patch_in_dim),
                nn.Linear(patch_in_dim, dim_unified, bias=False),
                nn.LayerNorm(dim_unified),
            )

        self.ucp_lidar = make_ucp()
        self.ucp_radar = make_ucp()
        self.ucp_camera = make_ucp()

        # ── Query: per-patch query repeated across spatial positions ──
        n_query = patch_size * patch_size * n_repeat_ch  # e.g. 2*2*8=32
        self.n_query = n_query
        self.base_query = nn.Parameter(torch.randn(1, n_query, dim_unified) * 0.02)

        # ── Patch rearrange helpers ──
        self.to_patches = Rearrange(
            'b c (y py) (x px) -> (b y x) (py px c)',
            py=patch_size, px=patch_size)
        self.from_patches = Rearrange(
            '(b y x) (py px ch) c -> b (c ch) (y py) (x px)',
            y=bev_h // patch_size, x=bev_w // patch_size,
            py=patch_size, px=patch_size, ch=n_repeat_ch)

        # ── DeCU query modulation: common+unique tokens → query bias ──
        n_patches = (bev_h // patch_size) * (bev_w // patch_size)  # total spatial patches

        # DeCU tokens → per-sample query modulation
        self.query_modulation = nn.Sequential(
            nn.Linear(token_dim * 5, 512),  # common + 3×unique + prompt → 2560
            nn.ReLU(),
            nn.Linear(512, dim_unified),
            nn.Tanh(),  # bounded modulation
        )

        # ── CASAP: single-layer cross-attention across sensors along patches ──
        self.casap = nn.MultiheadAttention(
            dim_unified, n_heads, dropout=0.0, batch_first=True)

        # ── PFT: post-feature transform ──
        self.pft = nn.Sequential(
            nn.LayerNorm(dim_unified),
            nn.Linear(dim_unified, dim_unified, bias=False),
            nn.LayerNorm(dim_unified),
        )

        # ── Final projection to head dimension ──
        out_ch = dim_unified * n_repeat_ch  # 256*8=2048
        self.head_proj = nn.Sequential(
            nn.Conv2d(out_ch, head_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.ReLU(),
        )

    def forward(self, bev_lidar, bev_radar, bev_camera,
                condition_common, unique_lidar, unique_radar,
                unique_img, prompt_token):
        """Fuse three BEV feature maps with DeCU-conditioned CASAP.

        Args:
            bev_l/r/c: [B, 768, H, W] calibrated BEV feature maps
            condition_common: [B, 512]
            unique_l/r/img: [B, 512]
            prompt_token: [B, 512]
        Returns:
            fused_bev: [B, head_dim, H, W]
        """
        B = bev_lidar.shape[0]

        # ── 1. UCP: patchify + project each sensor ──
        patches_l = self.to_patches(bev_lidar)      # [B*Np, py*px*C]
        patches_r = self.to_patches(bev_radar)
        patches_c = self.to_patches(bev_camera)

        proj_l = self.ucp_lidar(patches_l).unsqueeze(1)    # [B*Np, 1, Cu]
        proj_r = self.ucp_radar(patches_r).unsqueeze(1)
        proj_c = self.ucp_camera(patches_c).unsqueeze(1)

        kv_feats = torch.cat([proj_l, proj_r, proj_c], dim=1)  # [B*Np, 3, Cu]
        b_patch = kv_feats.shape[0]

        # ── 2. DeCU-conditioned query ──
        decu_all = torch.cat([condition_common, unique_lidar, unique_radar,
                              unique_img, prompt_token], dim=-1)  # [B, 2560]
        modulation = self.query_modulation(decu_all)  # [B, Cu]
        # Expand modulation to per-patch: [B, Cu] → [B*Np, Nq, Cu]
        n_patches_per_sample = b_patch // B
        modulation = modulation.unsqueeze(1).unsqueeze(1) \
            .expand(-1, n_patches_per_sample, self.n_query, -1) \
            .reshape(b_patch, self.n_query, self.dim_unified)

        q_feat = self.base_query.expand(b_patch, -1, -1) + modulation

        # ── 3. CASAP: cross-attention ──
        fused_patches, _ = self.casap(q_feat, kv_feats, kv_feats)  # [B*Np, Nq, Cu]

        # ── 4. PFT ──
        fused_patches = self.pft(fused_patches)

        # ── 5. Reshape back to BEV ──
        fused_bev = self.from_patches(fused_patches)  # [B, Cu*ch, H, W]

        # ── 6. Project to head dimension ──
        fused_bev = self.head_proj(fused_bev)  # [B, head_dim, H, W]

        return fused_bev

    def get_individual_fusions(self, bev_lidar, bev_radar, bev_camera,
                               condition_common, unique_lidar, unique_radar,
                               unique_img, prompt_token):
        """For SCL: generate fused BEV for all 7 sensor combinations."""
        B = bev_lidar.shape[0]
        all_bevs = {'lidar': bev_lidar, 'radar': bev_radar, 'camera': bev_camera}
        zero_bev = torch.zeros_like(bev_lidar)

        # Build unified patches for each sensor
        patches = {}
        for name, bev in all_bevs.items():
            p = self.to_patches(bev)
            if name == 'lidar':
                patches[name] = self.ucp_lidar(p).unsqueeze(1)
            elif name == 'radar':
                patches[name] = self.ucp_radar(p).unsqueeze(1)
            else:
                patches[name] = self.ucp_camera(p).unsqueeze(1)

        b_patch = patches['lidar'].shape[0]
        decu_all = torch.cat([condition_common, unique_lidar, unique_radar,
                              unique_img, prompt_token], dim=-1)
        modulation = self.query_modulation(decu_all)
        n_patches_per_sample = b_patch // B
        modulation = modulation.unsqueeze(1).unsqueeze(1) \
            .expand(-1, n_patches_per_sample, self.n_query, -1) \
            .reshape(b_patch, self.n_query, self.dim_unified)
        q_feat = self.base_query.expand(b_patch, -1, -1) + modulation

        # 7 combinations: C, L, R, L+R, C+R, C+L, C+L+R
        combos = [
            ['camera'], ['lidar'], ['radar'],
            ['lidar', 'radar'], ['camera', 'radar'],
            ['camera', 'lidar'], ['camera', 'lidar', 'radar'],
        ]

        results = []
        for combo in combos:
            kv_list = [patches[name] for name in combo]
            kv = torch.cat(kv_list, dim=1)
            fused = self.casap(q_feat, kv, kv)[0]
            fused = self.pft(fused)
            fused = self.from_patches(fused)
            fused = self.head_proj(fused)
            results.append(fused)

        return results
