import torch
import torch.nn as nn

import spconv.pytorch as spconv
from einops.layers.torch import Rearrange
from sklearn.neighbors import NearestNeighbors
import numpy as np


class RL3DFBackbone_knngate(nn.Module):
    def __init__(self, cfg):
        super(RL3DFBackbone_knngate, self).__init__()
        self.cfg = cfg
        self.roi = cfg.DATASET.RDR_SP_CUBE.ROI
        grid_size = cfg.DATASET.RDR_SP_CUBE.GRID_SIZE

        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        z_min, z_max = self.roi['z']

        z_shape = int((z_max-z_min) / grid_size)
        y_shape = int((y_max-y_min) / grid_size)
        x_shape = int((x_max-x_min) / grid_size)

        self.spatial_shape = [z_shape, y_shape, x_shape]

        cfg_model = self.cfg.MODEL
        input_dim = cfg_model.PRE_PROCESSOR.INPUT_DIM

        list_enc_channel = cfg_model.BACKBONE.ENCODING.CHANNEL
        list_enc_padding = cfg_model.BACKBONE.ENCODING.PADDING
        list_enc_stride  = cfg_model.BACKBONE.ENCODING.STRIDE
        
        # 1x1 conv / 4->ENCODING.CHANNEL[0]
        self.input_convR = spconv.SparseConv3d(
            in_channels=input_dim, out_channels=list_enc_channel[0],
            kernel_size=1, stride=1, padding=0, dilation=1, indice_key = 'sp0') 
        self.input_convL = spconv.SparseConv3d(
            in_channels=input_dim, out_channels=list_enc_channel[0],
            kernel_size=1, stride=1, padding=0, dilation=1, indice_key = 'sp0') 
        
        # encoder
        self.num_layer = len(list_enc_channel)
        for idx_enc in range(self.num_layer):
            if idx_enc == 0:
                temp_in_ch = list_enc_channel[0] 
            else:
                temp_in_ch = list_enc_channel[idx_enc-1] 
            temp_ch = list_enc_channel[idx_enc]
            temp_pd = list_enc_padding[idx_enc]
            setattr(self, f'spconv{idx_enc}R', \
                spconv.SparseConv3d(in_channels=temp_in_ch, out_channels=temp_ch, kernel_size=3, \
                    stride=list_enc_stride[idx_enc], padding=temp_pd, dilation=1, indice_key=f'sp{idx_enc}'))
            setattr(self, f'bn{idx_enc}R', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}aR', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}aR', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}bR', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}bR', nn.BatchNorm1d(temp_ch))
            
            setattr(self, f'spconv{idx_enc}L', \
                spconv.SparseConv3d(in_channels=temp_in_ch, out_channels=temp_ch, kernel_size=3, \
                    stride=list_enc_stride[idx_enc], padding=temp_pd, dilation=1, indice_key=f'sp{idx_enc}'))
            setattr(self, f'bn{idx_enc}L', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}aL', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}aL', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}bL', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}bL', nn.BatchNorm1d(temp_ch))

            # Condition Token (512) -> Gate Feature
            setattr(self, f'img_layer{idx_enc}', nn.Linear(512, temp_ch))
            setattr(self, f'value_layer{idx_enc}', nn.Linear(temp_ch, temp_ch))
            setattr(self, f'gate_layer{idx_enc}', nn.Linear(2*temp_ch, temp_ch))
            setattr(self, f'gap_layer{idx_enc}', nn.AdaptiveAvgPool1d(1))
        
        # to BEV
        list_bev_channel = cfg_model.BACKBONE.TO_BEV.CHANNEL
        list_bev_kernel = cfg_model.BACKBONE.TO_BEV.KERNEL_SIZE
        list_bev_stride = cfg_model.BACKBONE.TO_BEV.STRIDE
        list_bev_padding = cfg_model.BACKBONE.TO_BEV.PADDING
        if cfg_model.BACKBONE.TO_BEV.IS_Z_EMBED:
            self.is_z_embed = True
            for idx_bev in range(self.num_layer):
                setattr(self, f'chzcat{idx_bev}R', Rearrange('b c z y x -> b (c z) y x'))
                temp_in_channel = int(list_enc_channel[idx_bev]*z_shape/(2**idx_bev))
                temp_out_channel = list_bev_channel[idx_bev]
                setattr(self, f'convtrans2d{idx_bev}R', \
                    nn.ConvTranspose2d(in_channels=temp_in_channel, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev], padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}R', nn.BatchNorm2d(temp_out_channel))
                
                setattr(self, f'chzcat{idx_bev}L', Rearrange('b c z y x -> b (c z) y x'))
                temp_in_channel = int(list_enc_channel[idx_bev]*z_shape/(2**idx_bev))
                temp_out_channel = list_bev_channel[idx_bev]
                setattr(self, f'convtrans2d{idx_bev}L', \
                    nn.ConvTranspose2d(in_channels=temp_in_channel, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev], padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}L', nn.BatchNorm2d(temp_out_channel))
        else:
            self.is_z_embed = False
            for idx_bev in range(self.num_layer):
                temp_enc_ch = list_enc_channel[idx_bev] 
                temp_out_channel = list_bev_channel[idx_bev]
                z_kernel_size = int(z_shape/(2**idx_bev))

                setattr(self, f'toBEV{idx_bev}R', \
                    spconv.SparseConv3d(in_channels=temp_enc_ch, \
                        out_channels=temp_enc_ch, kernel_size=(z_kernel_size, 1, 1)))
                setattr(self, f'bnBEV{idx_bev}R', \
                    nn.BatchNorm1d(temp_enc_ch))
                setattr(self, f'convtrans2d{idx_bev}R', \
                    nn.ConvTranspose2d(in_channels=temp_enc_ch, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev],  padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}R', nn.BatchNorm2d(temp_out_channel))
                
                setattr(self, f'toBEV{idx_bev}L', \
                    spconv.SparseConv3d(in_channels=temp_enc_ch, \
                        out_channels=temp_enc_ch, kernel_size=(z_kernel_size, 1, 1)))
                setattr(self, f'bnBEV{idx_bev}L', \
                    nn.BatchNorm1d(temp_enc_ch))
                setattr(self, f'convtrans2d{idx_bev}L', \
                    nn.ConvTranspose2d(in_channels=temp_enc_ch, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev],  padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}L', nn.BatchNorm2d(temp_out_channel))
        # activation
        self.relu = nn.ReLU()

    def forward(self, dict_item):
        sparse_featuresR, sparse_indicesR = dict_item['sp_features'], dict_item['sp_indices']
        sparse_featuresL, sparse_indicesL = dict_item['sp_features_l'], dict_item['sp_indices_l']
        
        # Use Condition Token (img_embedding) for environment-aware gating
        if 'img_embedding' in dict_item:
            condition_token = dict_item['img_embedding'] # (B, 512)
        else:
            # Fallback if not available (should not happen if img_cls is updated)
            # print("Warning: img_embedding not found, using dummy")
            condition_token = torch.zeros((dict_item['batch_size'], 512), device=sparse_featuresR.device)

        # img_cls_output, img_cls_gap, img_cls_feat = dict_item['img_cls_output'], dict_item['img_cls_gap'], dict_item['img_cls_feat']
     
        input_sp_tensorR = spconv.SparseConvTensor(
            features=sparse_featuresR,
            indices=sparse_indicesR.int(),
            spatial_shape=self.spatial_shape,
            batch_size=dict_item['batch_size']
        )
        xR = self.input_convR(input_sp_tensorR) 
        
        input_sp_tensorL = spconv.SparseConvTensor(
            features=sparse_featuresL,
            indices=sparse_indicesL.int(),
            spatial_shape=self.spatial_shape,
            batch_size=dict_item['batch_size']
        )

        xL = self.input_convL(input_sp_tensorL) 

        list_bev_featuresR = []
        list_bev_featuresL = []
        
        for idx_layer in range(self.num_layer):
            xR = getattr(self, f'spconv{idx_layer}R')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}R')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            xR = getattr(self, f'subm{idx_layer}aR')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}aR')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            xR = getattr(self, f'subm{idx_layer}bR')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}bR')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            
            xL = getattr(self, f'spconv{idx_layer}L')(xL)
            xL = xL.replace_feature(getattr(self, f'bn{idx_layer}L')(xL.features))
            xL = xL.replace_feature(self.relu(xL.features))
            xL = getattr(self, f'subm{idx_layer}aL')(xL)
            xL = xL.replace_feature(getattr(self, f'bn{idx_layer}aL')(xL.features))
            xL = xL.replace_feature(self.relu(xL.features))
            xL = getattr(self, f'subm{idx_layer}bL')(xL)
            xL = xL.replace_feature(getattr(self, f'bn{idx_layer}bL')(xL.features))
            xL = xL.replace_feature(self.relu(xL.features))
  
            xL2 = xL
            # Use Condition Token for gating
            img_layer_feat = getattr(self, f'img_layer{idx_layer}')(condition_token)
            if len(img_layer_feat.shape) == 1:
                img_layer_feat = img_layer_feat.unsqueeze(0)
      
            batch = dict_item['batch_size']
            xR_feat, xR_indices = xR.features, xR.indices
            xL_feat, xL_indices = xL.features, xL.indices
            for batch_idx in range(batch):
                radar_indices = np.array(xR_indices[xR_indices[:, 0] == batch_idx][:, 1:].cpu())
                lidar_indices = np.array(xL_indices[xL_indices[:, 0] == batch_idx][:, 1:].cpu())
                nbrs = NearestNeighbors(n_neighbors=int(64 / (2**(idx_layer))), radius=int(8 / (2**(idx_layer)))).fit(radar_indices)
                lidar_indices_temp = lidar_indices 
                distances, knn_indices = nbrs.kneighbors(lidar_indices_temp) 
                knn_indices = torch.from_numpy(knn_indices).to(device=sparse_indicesR.device) 

                lidar_feat = xL_feat[xL_indices[:, 0] == batch_idx] 
                radar_feat = xR_feat[xR_indices[:, 0] == batch_idx] 
                query = lidar_feat
                key = radar_feat[knn_indices] 

                value = getattr(self, f'value_layer{idx_layer}')(key) 
                attn = torch.bmm(query.unsqueeze(1), key.permute(0,2,1)) 
                attn = torch.softmax(attn, dim=-1) 
                attn_value = torch.bmm(attn, value)

                img_layer_feat_batch = img_layer_feat[batch_idx].unsqueeze(0).unsqueeze(0).repeat(key.shape[0], key.shape[1], 1) 
                gate_feat = getattr(self, f'gate_layer{idx_layer}')(torch.cat((key, img_layer_feat_batch), -1)) 
                gate_gap_feat = getattr(self, f'gap_layer{idx_layer}')(gate_feat.permute(0, 2, 1)) 
                attn_value_gate = torch.einsum('abc, abc -> abc', attn_value, torch.sigmoid(gate_gap_feat.permute(0, 2, 1))) 

                if batch_idx == 0:
                    new_xL_feat = attn_value_gate.squeeze() + query
                    Fl_feat = attn_value.squeeze() 
                    Gl_feat = gate_gap_feat.squeeze()
                    FlGl_feat = attn_value_gate.squeeze() 
                else:
                    new_xL_feat = torch.cat((new_xL_feat, attn_value_gate.squeeze() + query), 0)  
                    Fl_feat = torch.cat((Fl_feat, attn_value.squeeze()), 0)
                    Gl_feat = torch.cat((Gl_feat, gate_gap_feat.squeeze()), 0)
                    FlGl_feat = torch.cat((FlGl_feat, attn_value_gate.squeeze()), 0)
               
            new_spatial_shape = [int(self.spatial_shape[0] / (2**(idx_layer))), int(self.spatial_shape[1] / (2**(idx_layer))), int(self.spatial_shape[2] / (2**(idx_layer)))]
            xL = spconv.SparseConvTensor(
                features=new_xL_feat,
                indices=xL_indices,
                spatial_shape=new_spatial_shape,
                batch_size=dict_item['batch_size']
            )

            if self.is_z_embed:
                bev_denseR = getattr(self, f'chzcat{idx_layer}R')(xL2.dense())
                bev_denseR = getattr(self, f'convtrans2d{idx_layer}R')(bev_denseR)
                bev_denseL = getattr(self, f'chzcat{idx_layer}L')(xL.dense())
                bev_denseL = getattr(self, f'convtrans2d{idx_layer}L')(bev_denseL)
            else:
                bev_spR = getattr(self, f'toBEV{idx_layer}R')(xL2) # Lidar original feature
                bev_spR = bev_spR.replace_feature(getattr(self, f'bnBEV{idx_layer}R')(bev_spR.features))
                bev_spR = bev_spR.replace_feature(self.relu(bev_spR.features))
                
                bev_spL = getattr(self, f'toBEV{idx_layer}L')(xL)
                bev_spL = bev_spL.replace_feature(getattr(self, f'bnBEV{idx_layer}L')(bev_spL.features))
                bev_spL = bev_spL.replace_feature(self.relu(bev_spL.features))

                bev_denseR = getattr(self, f'convtrans2d{idx_layer}R')(bev_spR.dense().squeeze(2))
                bev_denseL = getattr(self, f'convtrans2d{idx_layer}L')(bev_spL.dense().squeeze(2))
            
            bev_denseR = getattr(self, f'bnt{idx_layer}R')(bev_denseR)
            bev_denseR = self.relu(bev_denseR)
            bev_denseL = getattr(self, f'bnt{idx_layer}L')(bev_denseL)
            bev_denseL = self.relu(bev_denseL)

            list_bev_featuresR.append(bev_denseR)
            list_bev_featuresL.append(bev_denseL)

        bev_featuresR = torch.cat(list_bev_featuresR, dim = 1)
        bev_featuresL = torch.cat(list_bev_featuresL, dim = 1)

        dict_item['bev_feat'] = torch.cat((bev_featuresR, bev_featuresL), 1)        

        return dict_item


class TokenGuidedBEVFusion(nn.Module):
    def __init__(self, branch_feat_dim, out_dim, token_dim, hidden_dim, branch_weight_floor=0.1):
        super().__init__()
        self.branch_feat_dim = int(branch_feat_dim)
        self.branch_weight_floor = float(branch_weight_floor)
        self.shared_mix_head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.common_map_proj = nn.Sequential(
            nn.Linear(token_dim, branch_feat_dim),
            nn.LayerNorm(branch_feat_dim),
            nn.ReLU(),
        )
        self.unique_residual_gate_lidar = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, branch_feat_dim),
            nn.Sigmoid(),
        )
        self.unique_residual_gate_radar = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, branch_feat_dim),
            nn.Sigmoid(),
        )
        self.unique_residual_gate_camera = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, branch_feat_dim),
            nn.Sigmoid(),
        )
        self.shared_reduce = nn.Sequential(
            nn.Conv2d(branch_feat_dim * 4, branch_feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_feat_dim),
            nn.ReLU(),
            nn.Conv2d(branch_feat_dim, branch_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_feat_dim),
            nn.ReLU(),
        )
        self.final_reduce = nn.Sequential(
            nn.Conv2d(branch_feat_dim * 4, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )

    def forward(self, bev_feat_l, bev_feat_r, bev_feat_c, condition_common, unique_lidar, unique_radar, unique_img):
        mix_logits = self.shared_mix_head(condition_common)
        mix_weights = torch.softmax(mix_logits, dim=-1)

        shared_lidar = mix_weights[:, 0].view(-1, 1, 1, 1) * bev_feat_l
        shared_radar = mix_weights[:, 1].view(-1, 1, 1, 1) * bev_feat_r
        shared_camera = mix_weights[:, 2].view(-1, 1, 1, 1) * bev_feat_c

        common_map = self.common_map_proj(condition_common).view(-1, self.branch_feat_dim, 1, 1)
        common_map = common_map.expand(-1, -1, bev_feat_l.shape[-2], bev_feat_l.shape[-1])

        shared_input = torch.cat((shared_lidar, shared_radar, shared_camera, common_map), dim=1)
        shared_bev = self.shared_reduce(shared_input)

        residual_gate_lidar = self.unique_residual_gate_lidar(unique_lidar).view(-1, self.branch_feat_dim, 1, 1)
        residual_gate_radar = self.unique_residual_gate_radar(unique_radar).view(-1, self.branch_feat_dim, 1, 1)
        residual_gate_camera = self.unique_residual_gate_camera(unique_img).view(-1, self.branch_feat_dim, 1, 1)

        residual_lidar = residual_gate_lidar * bev_feat_l
        residual_radar = residual_gate_radar * bev_feat_r
        residual_camera = residual_gate_camera * bev_feat_c

        fused_bev = self.final_reduce(
            torch.cat((shared_bev, residual_lidar, residual_radar, residual_camera), dim=1)
        )

        weighted_branches = {
            'lidar': shared_lidar + residual_lidar,
            'radar': shared_radar + residual_radar,
            'camera': shared_camera + residual_camera,
        }
        shared_branches = {
            'lidar': shared_lidar,
            'radar': shared_radar,
            'camera': shared_camera,
        }
        residual_branches = {
            'lidar': residual_lidar,
            'radar': residual_radar,
            'camera': residual_camera,
        }
        return fused_bev, mix_weights, weighted_branches, common_map, shared_bev, shared_branches, residual_branches


class RL3DFBackbone_Branching(RL3DFBackbone_knngate):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.force_branch = str(getattr(cfg.MODEL.BACKBONE, 'FORCE_BRANCH', 'none')).lower()
        if self.force_branch not in ['none', 'lidar', 'radar', 'camera']:
            self.force_branch = 'none'

        # ── Route 2: Deeper weather auxiliary head ──
        weather_aux_cfg = self.cfg.MODEL.get('WEATHER_AUX', {})
        self.enable_weather_aux = bool(weather_aux_cfg.get('ENABLED', False))
        weather_aux_hidden = int(weather_aux_cfg.get('HIDDEN_DIM', 256))
        weather_aux_dropout = float(weather_aux_cfg.get('DROPOUT', 0.3))
        weather_num_classes = int(weather_aux_cfg.get('NUM_CLASSES', 7))
        if self.enable_weather_aux:
            self.weather_aux_head = nn.Sequential(
                nn.Linear(512, weather_aux_hidden),
                nn.ReLU(),
                nn.Linear(weather_aux_hidden, weather_num_classes),
            )

        self.branch_weight_floor = float(getattr(cfg.MODEL.BACKBONE, 'BRANCH_WEIGHT_FLOOR', 0.1))
        self.branch_weight_floor = min(max(self.branch_weight_floor, 0.0), (1.0 / 3.0) - 1e-6)
        self.initial_branch_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        self.gate_residual_floor = float(getattr(cfg.MODEL.BACKBONE, 'GATE_RESIDUAL_FLOOR', 0.15))
        self.gate_residual_floor = min(max(self.gate_residual_floor, 0.0), 0.5)

        # ── WCBR router: token_dim from CONDITION_MODEL or default 512 ──
        cond_cfg = self.cfg.MODEL.get('CONDITION_MODEL', {}) or {}
        token_dim = int(cond_cfg.get('TOKEN_DIM', 512))
        pool_dim = int(cfg.MODEL.BACKBONE.ENCODING.CHANNEL[0])
        branch_feat_dim = int(sum(cfg.MODEL.BACKBONE.TO_BEV.CHANNEL))
        head_dim = int(cfg.MODEL.HEAD.DIM)
        self.sensor_token_proj = nn.Linear(pool_dim, token_dim)  # pool dim→512
        self.condition_mlp = nn.Sequential(
            nn.Linear(token_dim * 3, 1024),  # condition + radar + lidar
            nn.ReLU(),
            nn.Linear(1024, token_dim),
        )
        self.condition_mlp_norm = nn.LayerNorm(token_dim)
        # Router: single linear layer — DeCU common + 3 unique tokens + CLIP prompt = 5×512=2560
        # unique tokens carry weather-specific sensor characteristics → router can learn weather-adaptive routing
        router_input_dim = token_dim * 5
        self.branch_router = nn.Linear(router_input_dim, 3)
        nn.init.normal_(self.branch_router.weight, std=0.02)
        nn.init.zeros_(self.branch_router.bias)
        # Project final BEV to HEAD.DIM
        self.bev_proj = nn.Sequential(
            nn.Conv2d(branch_feat_dim, head_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.ReLU(),
        )
        # ── end WCBR router ──

        self.branch_feat_dim = branch_feat_dim

        camera_cfg = cfg.MODEL.get('CAMERA_BRANCH', {})
        camera_feat_dim = int(camera_cfg.get('OUT_CHANNELS', self.branch_feat_dim))
        self.camera_align = None
        if camera_feat_dim != self.branch_feat_dim:
            self.camera_align = nn.Sequential(
                nn.Conv2d(camera_feat_dim, self.branch_feat_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.branch_feat_dim),
                nn.ReLU(),
            )

        # ── Route 1: Camera-LiDAR cross-modal alignment ──
        self.camera_lidar_align = nn.Sequential(
            nn.Conv2d(self.branch_feat_dim * 2, self.branch_feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.branch_feat_dim),
            nn.ReLU(),
            nn.Conv2d(self.branch_feat_dim, self.branch_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.branch_feat_dim),
            nn.ReLU(),
        )

        def make_calibrator():
            return nn.Sequential(
                nn.Conv2d(self.branch_feat_dim, self.branch_feat_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.branch_feat_dim),
                nn.ReLU(),
            )

        self.lidar_feat_calibrator = make_calibrator()
        self.radar_feat_calibrator = make_calibrator()
        self.camera_feat_calibrator = make_calibrator()

        cond_cfg_gate = self.cfg.MODEL.get('CONDITION_MODEL', {}) or {}
        token_dim = int(cond_cfg_gate.get('TOKEN_DIM', 512))
        gate_hidden = int(cond_cfg_gate.get('HIDDEN_DIM', 256))
        gate_input_dim = token_dim * 2

        def make_gate():
            return nn.Sequential(
                nn.Linear(gate_input_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, self.branch_feat_dim),
                nn.Sigmoid(),
            )

        self.lidar_branch_gate = make_gate()
        self.radar_branch_gate = make_gate()
        self.camera_branch_gate = make_gate()

        self.token_guided_fusion = TokenGuidedBEVFusion(
            branch_feat_dim=self.branch_feat_dim,
            out_dim=cfg.MODEL.HEAD.DIM,
            token_dim=token_dim,
            hidden_dim=gate_hidden,
            branch_weight_floor=self.branch_weight_floor,
        )

        branch_pref_cfg = cfg.MODEL.get('BRANCH_PREF', {})
        self.enable_branch_pref = bool(branch_pref_cfg.get('ENABLED', False))
        if self.enable_branch_pref:
            pref_hidden = int(branch_pref_cfg.get('HIDDEN_DIM', 256))
            self.branch_preference_head = nn.Sequential(
                nn.Linear(token_dim * 4, pref_hidden),
                nn.ReLU(),
                nn.Linear(pref_hidden, 3),
            )
        
    def forward(self, dict_item):
        sparse_featuresR, sparse_indicesR = dict_item['sp_features'], dict_item['sp_indices']
        sparse_featuresL, sparse_indicesL = dict_item['sp_features_l'], dict_item['sp_indices_l']

        batch_size = dict_item['batch_size']
        cond_cfg = self.cfg.MODEL.get('CONDITION_MODEL', {}) or {}
        token_dim = int(cond_cfg.get('TOKEN_DIM', 512))
        device = sparse_featuresR.device
        condition_common = dict_item.get('condition_common_token', None)
        if condition_common is None:
            condition_common = torch.zeros((batch_size, token_dim), device=device)
        unique_lidar = dict_item.get('condition_unique_lidar', condition_common)
        unique_radar = dict_item.get('condition_unique_radar', condition_common)
        unique_img = dict_item.get('condition_unique_img', condition_common)

        if self.enable_weather_aux:
            dict_item['weather_logits_aux'] = self.weather_aux_head(condition_common)

        if self.enable_branch_pref:
            pref_input = torch.cat((condition_common, unique_lidar, unique_radar, unique_img), dim=-1)
            pref_logits = self.branch_preference_head(pref_input)
            if self.force_branch != 'none':
                forced = torch.full_like(pref_logits, fill_value=-9.0)
                branch_to_idx = {'lidar': 0, 'radar': 1, 'camera': 2}
                forced[:, branch_to_idx[self.force_branch]] = 9.0
                pref_logits = forced
            dict_item['branch_preference_logits'] = pref_logits
            dict_item['branch_preference'] = torch.softmax(pref_logits, dim=-1)

        input_sp_tensorR = spconv.SparseConvTensor(
            features=sparse_featuresR,
            indices=sparse_indicesR.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )
        xR = self.input_convR(input_sp_tensorR) 
        
        input_sp_tensorL = spconv.SparseConvTensor(
            features=sparse_featuresL,
            indices=sparse_indicesL.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )
        xL = self.input_convL(input_sp_tensorL) 

        # ── WCBR-style MLP branch routing (early, after first conv) ──
        radar_token = self._pool_sparse_tokens(xR.features, xR.indices, batch_size)
        lidar_token = self._pool_sparse_tokens(xL.features, xL.indices, batch_size)
        radar_token = self.sensor_token_proj(radar_token)
        lidar_token = self.sensor_token_proj(lidar_token)
        
        # DeCU common token + CLIP weather prompt → direct weather-aware routing
        decu_common = dict_item.get('condition_common_token', None)
        prompt_token = dict_item.get('prompt_weather_token', None)
        if decu_common is None:
            decu_common = torch.zeros((batch_size, token_dim), device=sparse_featuresR.device)
        if prompt_token is None:
            prompt_token = torch.zeros((batch_size, token_dim), device=sparse_featuresR.device)
        
        # Sensor update (radar/lidar geometry) → condition_update
        token_concat = torch.cat((decu_common, radar_token, lidar_token), dim=-1)
        condition_update = self.condition_mlp(token_concat)
        decu_common = self.condition_mlp_norm(decu_common + condition_update)
        
        # Router input: DeCU common + unique(Lidar/Radar/Camera) + CLIP weather prompt
        # unique tokens contain weather-specific sensor patterns → key for weather-adaptive routing
        router_input = torch.cat((decu_common, unique_lidar, unique_radar, unique_img, prompt_token), dim=-1)
        branch_logits = self.branch_router(router_input)
        branch_real_weights = torch.softmax(branch_logits, dim=-1)
        if self.branch_weight_floor > 0.0:
            branch_real_weights = (1.0 - 3.0 * self.branch_weight_floor) * branch_real_weights + self.branch_weight_floor
        dict_item['branch_real_weights'] = branch_real_weights
        # ── end WCBR router ──

        xL_pure = xL

        list_bev_L = []
        list_bev_R = []
        
        for idx_layer in range(self.num_layer):
            xR = getattr(self, f'spconv{idx_layer}R')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}R')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            xR = getattr(self, f'subm{idx_layer}aR')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}aR')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            xR = getattr(self, f'subm{idx_layer}bR')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}bR')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            
            xL_pure = getattr(self, f'spconv{idx_layer}L')(xL_pure)
            xL_pure = xL_pure.replace_feature(getattr(self, f'bn{idx_layer}L')(xL_pure.features))
            xL_pure = xL_pure.replace_feature(self.relu(xL_pure.features))
            xL_pure = getattr(self, f'subm{idx_layer}aL')(xL_pure)
            xL_pure = xL_pure.replace_feature(getattr(self, f'bn{idx_layer}aL')(xL_pure.features))
            xL_pure = xL_pure.replace_feature(self.relu(xL_pure.features))
            xL_pure = getattr(self, f'subm{idx_layer}bL')(xL_pure)
            xL_pure = xL_pure.replace_feature(getattr(self, f'bn{idx_layer}bL')(xL_pure.features))
            xL_pure = xL_pure.replace_feature(self.relu(xL_pure.features))

            def to_bev(tensor, idx_layer, suffix):
                if self.is_z_embed:
                    bev_dense = getattr(self, f'chzcat{idx_layer}{suffix}')(tensor.dense())
                    bev_dense = getattr(self, f'convtrans2d{idx_layer}{suffix}')(bev_dense)
                else:
                    bev_sp = getattr(self, f'toBEV{idx_layer}{suffix}')(tensor)
                    bev_sp = bev_sp.replace_feature(getattr(self, f'bnBEV{idx_layer}{suffix}')(bev_sp.features))
                    bev_sp = bev_sp.replace_feature(self.relu(bev_sp.features))
                    bev_dense = getattr(self, f'convtrans2d{idx_layer}{suffix}')(bev_sp.dense().squeeze(2))
                
                bev_dense = getattr(self, f'bnt{idx_layer}{suffix}')(bev_dense)
                bev_dense = self.relu(bev_dense)
                return bev_dense

            list_bev_L.append(to_bev(xL_pure, idx_layer, 'L'))
            list_bev_R.append(to_bev(xR, idx_layer, 'R'))

        bev_feat_L = torch.cat(list_bev_L, dim=1)
        bev_feat_R = torch.cat(list_bev_R, dim=1)

        bev_feat_C = dict_item.get('camera_bev_feat', None)
        if bev_feat_C is None:
            bev_feat_C = torch.zeros_like(bev_feat_L)
        else:
            bev_feat_C = bev_feat_C.to(bev_feat_L.device)
            if self.camera_align is not None:
                bev_feat_C = self.camera_align(bev_feat_C)
            if bev_feat_C.shape[-2:] != bev_feat_L.shape[-2:]:
                bev_feat_C = torch.nn.functional.interpolate(
                    bev_feat_C,
                    size=bev_feat_L.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )

        bev_feat_L = self.lidar_feat_calibrator(bev_feat_L)
        bev_feat_R = self.radar_feat_calibrator(bev_feat_R)
        bev_feat_C = self.camera_feat_calibrator(bev_feat_C)

        # ── Route 1: Camera-LiDAR cross-modal alignment ──
        # Let camera BEV "see" where LiDAR has strong geometric features
        bev_feat_C = self.camera_lidar_align(
            torch.cat((bev_feat_C, bev_feat_L.detach()), dim=1)
        )

        gate_lidar = self.lidar_branch_gate(torch.cat((condition_common, unique_lidar), dim=-1)).view(-1, self.branch_feat_dim, 1, 1)
        gate_radar = self.radar_branch_gate(torch.cat((condition_common, unique_radar), dim=-1)).view(-1, self.branch_feat_dim, 1, 1)
        gate_camera = self.camera_branch_gate(torch.cat((condition_common, unique_img), dim=-1)).view(-1, self.branch_feat_dim, 1, 1)

        gate_floor = self.gate_residual_floor
        gated_lidar = (gate_floor + (1.0 - gate_floor) * gate_lidar) * bev_feat_L
        gated_radar = (gate_floor + (1.0 - gate_floor) * gate_radar) * bev_feat_R
        gated_camera = (gate_floor + (1.0 - gate_floor) * gate_camera) * bev_feat_C

        # Use pre-computed branch weights for BEV fusion
        w_L = branch_real_weights[:, 0].view(-1, 1, 1, 1)
        w_R = branch_real_weights[:, 1].view(-1, 1, 1, 1)
        w_C = branch_real_weights[:, 2].view(-1, 1, 1, 1)
        final_bev = w_L * gated_lidar + w_R * gated_radar + w_C * gated_camera
        final_bev = self.bev_proj(final_bev)  # project to HEAD.DIM
        dict_item['final_bev'] = final_bev

        dict_item['gated_bev_lidar'] = gated_lidar
        dict_item['gated_bev_radar'] = gated_radar
        dict_item['gated_bev_camera'] = gated_camera
        dict_item['calibrated_bev_lidar'] = bev_feat_L
        dict_item['calibrated_bev_radar'] = bev_feat_R
        dict_item['calibrated_bev_camera'] = bev_feat_C
        dict_item['branch_real_weights'] = branch_real_weights
        dict_item['bev_feat'] = final_bev
        return dict_item

    def _pool_sparse_tokens(self, features, indices, batch_size):
        """Pool sparse features to per-batch mean token (WCBR-compatible)."""
        pooled = []
        for batch_idx in range(batch_size):
            mask = indices[:, 0] == batch_idx
            if mask.any():
                pooled.append(features[mask].mean(dim=0))
            else:
                pooled.append(torch.zeros(features.shape[1], device=features.device, dtype=features.dtype))
        return torch.stack(pooled, dim=0)
