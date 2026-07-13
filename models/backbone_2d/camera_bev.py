import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraBEVBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        cfg_camera = cfg.MODEL.get('CAMERA_BRANCH', {})
        self.enabled = bool(cfg_camera.get('ENABLED', True))
        stem_channels = list(cfg_camera.get('STEM_CHANNELS', [32, 64, 128, 256]))
        out_channels = int(cfg_camera.get('OUT_CHANNELS', 768))
        bev_h = int(cfg_camera.get('BEV_HEIGHT', 32))
        bev_w = int(cfg_camera.get('BEV_WIDTH', 180))

        c1, c2, c3, c4 = stem_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(),
        )
        self.to_bev = nn.Sequential(
            nn.AdaptiveAvgPool2d((bev_h, bev_w)),
            nn.Conv2d(c4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c4, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )

    def forward(self, dict_item):
        if (not self.enabled) or ('cam_front_img' not in dict_item):
            return dict_item

        img = dict_item['cam_front_img'].to(next(self.parameters()).device)
        feat_2d = self.encoder(img)
        dict_item['camera_feat_2d'] = feat_2d
        dict_item['camera_bev_feat'] = self.to_bev(feat_2d)
        dict_item['camera_global_token'] = self.global_proj(feat_2d)
        return dict_item
