import diffusion_feature
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from transformers.models.clip.modeling_clip import CLIPTextEmbeddings
from transformers.models.clip.configuration_clip import CLIPTextConfig
from net.imagePrompts import imagePrompts

# proxy
os.environ['HTTP_PROXY'] = 'http://localhost:7098'
os.environ['HTTPS_PROXY'] = 'http://localhost:7098'





class PyramidFeatureExtraction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=ks//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for ks in [1, 3, 5, 7]
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        branch_outs = [branch(x) for branch in self.branches]
        return self.fusion(torch.cat(branch_outs, dim=1))

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv2d(dim, dim//8, 1)
        self.key = nn.Conv2d(dim, dim//8, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        query = self.query(x).view(B, -1, H*W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H*W)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value(x).view(B, -1, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return x + self.gamma * out

class DiffAGIQA(nn.Module):
    def __init__(self, output_dim=1, version='1-5', img_size=512, device='cuda', dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        
        # 初始化 Diffusion Feature Extractor
        self.feature_extractor = diffusion_feature.FeatureExtractor(
            layer={
                'mid-vit-block0-ffn-inner': True,
                # 'up-level0-repeat1-res-out': True,
                "up-level1-repeat0-vit-block0-ffn-inner": True,
                "up-level1-repeat1-vit-block0-ffn-inner": True,
                "up-level1-repeat2-vit-block0-ffn-inner": True,
                "up-level2-repeat0-vit-block0-ffn-inner": True
            },
            version=version,
            img_size=img_size,
            device=device
        )
        self.image_prompts = imagePrompts()
        # 使用金字塔特征提取
        self.pyramid_extractors = nn.ModuleList([
            PyramidFeatureExtraction(dim, 256) 
            for dim in [5120, 5120, 5120, 5120, 2560]
        ])
        
        # 跨维度注意力
        self.cross_attentions = nn.ModuleList([
            CrossAttention(256) for _ in range(6)
        ])
        
        # 特征融合网络
        self.fusion_quality = nn.Sequential(
            nn.Conv2d(256*5, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1, groups=4),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.fusion_align = nn.Sequential(
            nn.Conv2d(256*5, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1, groups=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 改进的回归头
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
        
        self.align_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
    
    def extract_features(self, images, prompts='', task='quality'):
        features = []
            
        for p, i in zip(prompts, images):
            i = i.cuda()
            i = self.feature_extractor.preprocess_image(i,is_tensor=True)
            i = self.image_prompts(i)
            with torch.no_grad():
                encoded_prompts = self.feature_extractor.encode_prompt(p)
                fea = self.feature_extractor.extract(encoded_prompts, batch_size=1, image=i, image_type='tensor')
                features.append(fea)
                
        keys = list(features[0].keys())
        stacked_features = {}
        
        for key in keys:
            tensors = [d[key] for d in features]
            stacked_features[key] = torch.stack(tensors)
            
        feat1 = stacked_features[keys[0]].squeeze(1)
        feat2 = stacked_features[keys[1]].squeeze(1)
        feat3 = stacked_features[keys[2]].squeeze(1)
        feat4 = stacked_features[keys[3]].squeeze(1)
        feat5 = stacked_features[keys[4]].squeeze(1)
        # feat6 = stacked_features[keys[5]].squeeze(1)
        
        return feat1, feat2, feat3, feat4, feat5

    def forward(self, images, prompts=''):
        z1, z2, z3, z4, z5 = self.extract_features(images, prompts, 'quality')
        # for x in [z1, z2, z3, z4, z5]:
        #     print(x.shape)
        
        # 统一特征尺寸
        target_size = 32
        features = [F.adaptive_avg_pool2d(z1, (target_size,target_size)),
                   F.adaptive_avg_pool2d(z2, (target_size,target_size)),
                   F.adaptive_avg_pool2d(z3, (target_size,target_size)),
                   F.adaptive_avg_pool2d(z4, (target_size,target_size)),
                   F.adaptive_avg_pool2d(z5, (target_size,target_size))]

        # 金字塔特征提取
        pyramid_features = [
            extractor(feat) for extractor, feat in zip(self.pyramid_extractors, features)
        ]
        
        # 跨维度注意力
        attended_features = [
            attn(feat) for attn, feat in zip(self.cross_attentions, pyramid_features)
        ]
        
        # 特征融合
        fused_features = torch.cat(attended_features, dim=1)
        quality_feat = self.fusion_quality(fused_features)
        # align_feat = self.fusion_align(fused_features)
        
        # 预测
        quality = self.quality_head(quality_feat)
        # alignment = self.align_head(align_feat)
        
        return quality