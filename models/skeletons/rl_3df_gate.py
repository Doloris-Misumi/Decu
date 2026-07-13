import torch
import torch.nn as nn
import torch.nn.functional as F

from models import pre_processor, backbone_2d, backbone_3d, head, roi_head, img_cls
from models.condition import __all__ as condition_modules

class RL3DF_gate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.MODEL

        # CLIP switch
        self.use_clip = bool(self.cfg_model.get('USE_CLIP', True))

        if self.use_clip:
            from models.text_encoder.clip_encoder import TextEncoder
            self.text_encoder = TextEncoder(freeze=True)
            self.text_encoder.eval()
            self.weather_vocab = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']
            weather_prompts = [f"A {w} driving scene" for w in self.weather_vocab]
            with torch.no_grad():
                weather_features = self.text_encoder(weather_prompts)
                weather_features = F.normalize(weather_features, dim=-1)
            self.register_buffer('weather_features', weather_features)
            self.prompt_token_proj = nn.Sequential(
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
            )
        else:
            self.text_encoder = None

        self.list_module_names = [
            'pre_processor', 'pre_processor2', 'img_cls', 'condition_model', 'backbone_2d', 'backbone_3d', 'head', 'roi_head',
        ]
        self.list_modules = []
        self.build_rl_detector()

    def build_rl_detector(self):
        for name_module in self.list_module_names:
            module = getattr(self, f'build_{name_module}')()
            if module is not None:
                self.add_module(name_module, module) # override nn.Module
                self.list_modules.append(module)

    def build_img_cls(self):
        if self.cfg_model.get('IMG_CLS', None) is None:
            return None
        
        module = img_cls.__all__[self.cfg_model.IMG_CLS.NAME]()
        return module 

    def build_pre_processor(self):
        if self.cfg_model.get('PRE_PROCESSOR', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR.NAME](self.cfg)
        return module 
    
    def build_pre_processor2(self):
        if self.cfg_model.get('PRE_PROCESSOR2', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR2.NAME](self.cfg)
        return module 

    def build_backbone_3d(self):
        cfg_backbone = self.cfg_model.get('BACKBONE', None)
        return backbone_3d.__all__[cfg_backbone.NAME](self.cfg)

    def build_backbone_2d(self):
        cfg_backbone = self.cfg_model.get('CAMERA_BRANCH', None)
        if cfg_backbone is None:
            return None
        if not bool(cfg_backbone.get('ENABLED', True)):
            return None
        return backbone_2d.__all__[cfg_backbone.NAME](self.cfg)

    def build_condition_model(self):
        cfg_condition = self.cfg_model.get('CONDITION_MODEL', None)
        if cfg_condition is None:
            return None
        return condition_modules[cfg_condition.NAME](self.cfg)

    def build_head(self):
        if (self.cfg.MODEL.get('HEAD', None)) is None:
            return None
        module = head.__all__[self.cfg_model.HEAD.NAME](self.cfg)
        return module

    def build_roi_head(self):
        if (self.cfg.MODEL.get('ROI_HEAD', None)) is None:
            return None
        head_module = roi_head.__all__[self.cfg_model.ROI_HEAD.NAME](self.cfg)
        return head_module

    def forward(self, x):
        # CLIP: soft weighted weather embedding from text prompt
        if self.use_clip and 'condition_prompts' in x:
            with torch.no_grad():
                prompt_features = self.text_encoder(x['condition_prompts'])
                prompt_features = F.normalize(prompt_features, dim=-1)
            weather_logits = prompt_features @ self.weather_features.t()
            weather_probs = torch.softmax(weather_logits, dim=-1)
            prompt_weather_token = weather_probs @ self.weather_features
            x['prompt_weather_token'] = self.prompt_token_proj(prompt_weather_token)
            x['weather_probs'] = weather_probs

        for module in self.list_modules:
            x = module(x)

        return x
