# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Projected discriminator architecture from
"StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis".
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import RandomCrop, Normalize
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch_utils import misc
from networks.shared import ResidualBlock, FullyConnectedLayer
from networks.vit_utils import make_vit_backbone, forward_vit
from training.diffaug import DiffAugment


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size = kernel_size,
            padding = kernel_size//2,
            padding_mode = 'circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

        self.joint_conv = nn.Sequential(
            nn.Conv2d(2*64, 2*64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*64, 64, 3, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)
        channel = out.shape[1]
        batch = out.shape[0]

        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)
            cmap = cmap.unsqueeze(-1)
            cmap = cmap.repeat(1, 1, 14, 14)
            out = out.reshape(batch, channel, 14, 14)
            h_c_code = torch.cat((out, cmap), 1)
            out = self.joint_conv(h_c_code)
            out = out.sum(1, keepdim=True)
            out = out.reshape(batch, 1, -1)
            #out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out

# class DiscHead(nn.Module):
#     def __init__(self, channels: int, c_dim: int, cmap_dim: int = 384):
#         super().__init__()
#         self.channels = channels
#         self.c_dim = c_dim
#         self.cmap_dim = cmap_dim
#
#         if self.c_dim > 0:
#             self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
#         else:
#             None
#
#         self.joint_conv = nn.Sequential(
#             nn.Conv2d(12*64, 12*64, 3, 1, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(12*64, 6*64, 3, 1, 1, bias=False),
#         )
#
#     def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
#         channel = x.shape[1]
#         batch = x.shape[0]
#
#         if self.c_dim > 0:
#             cmap = self.cmapper(c).unsqueeze(-1)
#             cmap = cmap.unsqueeze(-1)
#             cmap = cmap.repeat(1, 1, 14, 14)
#             x = x.reshape(batch, channel, 14, 14)
#             h_c_code = torch.cat((x, cmap), 1)
#             out = self.joint_conv(h_c_code)
#             out = out.sum(1, keepdim=True)
#             out = out.reshape(batch, 1, -1)
#             #out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
#
#         return out

class DINO(torch.nn.Module):
    def __init__(self, hooks: list[int] = [2,5,8,11], hook_patch: bool = True):
        super().__init__()
        self.n_hooks = len(hooks) + int(hook_patch)

        self.model = make_vit_backbone(
            timm.create_model('vit_small_patch16_224_dino', pretrained=True),
            patch_size=[16,16], hooks=hooks, hook_patch=hook_patch,
        )
        self.model = self.model.eval().requires_grad_(False)

        self.img_resolution = self.model.model.patch_embed.img_size[0]
        self.embed_dim = self.model.model.embed_dim
        self.norm = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' input: x in [0, 1]; output: dict of activations '''
        x = F.interpolate(x, self.img_resolution, mode='area')
        x = self.norm(x)
        features = forward_vit(self.model, x)
        return features


class NetD(nn.Module):
    def __init__(self, c_dim: int, diffaug: bool = True, p_crop: float = 0.5):
        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.p_crop = p_crop

        self.dino = DINO()

        heads = []
        for i in range(self.dino.n_hooks):
            heads += [str(i), DiscHead(self.dino.embed_dim, c_dim)],
        self.heads = nn.ModuleDict(heads)

    def train(self, mode: bool = True):
        self.dino = self.dino.train(False)
        self.heads = self.heads.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Apply augmentation (x in [-1, 1]).
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')

        # Transform to [0, 1].
        x = x.add(1).div(2)

        # Take crops with probablity p_crop if the image is larger.
        if x.size(-1) > self.dino.img_resolution and np.random.random() < self.p_crop:
            x = RandomCrop(self.dino.img_resolution)(x)

        # Forward pass through DINO ViT.
        features = self.dino(x)

        # Apply discriminator heads.
        logits = []
        for k, head in self.heads.items():
            logits.append(head(features[k], c).view(x.size(0), -1))
        logits = torch.cat(logits, dim=1)

        return logits


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        nn.init.constant_(m[0].weight, val=0)
        nn.init.constant_(m[3].weight, val=0)
    else:
        nn.init.constant_(m, val=0)

class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            nn.init.kaiming_uniform_(self.conv_mask.weight, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

def word_level_correlation_focus_RF(img_features, words_emb,
               cap_lens, batch_size, class_ids, labels, word_labels, ContextBlock, fake_features):
    masks = []
    att_maps = []
    result = 0
    fake_result = 0
    cap_lens = cap_lens.data.tolist()
    similar_list = []

    for i in range(batch_size):
        if class_ids is not None:
            ###mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask = (class_ids == class_ids[i]).numpy().astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))

        words_num = cap_lens[i]
        #word = words_emb[i, :, :words_num].unsqueeze(0).contiguous() #DAMSMencoders
        word = words_emb[i, :words_num, :].unsqueeze(0).contiguous().float()  #ViT-L/14
        word = word.permute(0, 2, 1)  #ViT-L/14
        cur_word_labels = word_labels[i, :words_num]
        fake_word_labels = torch.zeros(cur_word_labels.size()).cuda()

        #context = img_features[i, :, :, :].unsqueeze(0).contiguous()#DAMSMencoders
        context = img_features[i, :, :, :]  #ViT-L/14
        context = torch.sum(context, 0).unsqueeze(0)     #keep dimension ViT-B/32
        tmp_conv = torch.nn.Conv2d(768, 512, 1).cuda()   #keep dimension ViT-B/32
        context = tmp_conv(context)                      #keep dimension ViT-B/32
        ###################################################

        batch_size, queryL = word.size(0), word.size(2)
        ih, iw = context.size(2), context.size(3)
        sourceL = ih * iw #ViT-B/32
        #sourceL = context.size(1) # ViT-L/14

        word_self = word.unsqueeze(3).contiguous()
        word_self = ContextBlock(word_self)
        word_self = word_self.squeeze(3).contiguous()
        sum_word_self = word_self.sum(dim=1, keepdim=False)
        sum_word_self = sum_word_self.repeat(sourceL, 1)

        context = context.view(batch_size, -1, sourceL)
        contextT = torch.transpose(context, 1, 2).contiguous()

        attn = torch.bmm(contextT, word)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax(dim=0)(attn)
        attn = attn.view(batch_size, sourceL, queryL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        attn = attn.view(batch_size * queryL, sourceL)
        attn = attn * 5.0 #cfg.TRAIN.SMOOTH.GAMMA1
        attn = nn.Softmax(dim=0)(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        attnT = torch.transpose(attn, 1, 2).contiguous()

        attnT = attnT.mul(sum_word_self)

        weightedContext = torch.bmm(context, attnT)
        weightedContext = weightedContext + word

        ###################################################
        cur_weiContext = weightedContext[0, :, :]
        cur_weiContext = cur_weiContext.transpose(0, 1)
        sum_weiContext = cur_weiContext.sum(dim=1, keepdim=False)
        soft_weiContext = nn.Softmax(dim=0)(sum_weiContext)
        cur_result = nn.BCELoss()(soft_weiContext, cur_word_labels.float())

        result += cur_result

        ############ fake images ############
        #fake_context = fake_features[i, :, :, :].unsqueeze(0).contiguous()#DAMSMencoders
        fake_context = img_features[i, :, :, :]  # ViT-L/14
        fake_context = torch.sum(fake_context, 0).unsqueeze(0)     #keep dimension ViT-B/32
        tmp_conv_fake = torch.nn.Conv2d(768, 512, 1).cuda()        #keep dimension ViT-B/32
        fake_context = tmp_conv_fake(fake_context)                 #keep dimension ViT-B/32
        ###################################################
        batch_size, queryL = word.size(0), word.size(2)
        fih, fiw = fake_context.size(2), fake_context.size(3)
        sourceL = fih * fiw #ViT-B/32
        #sourceL = fake_context.size(1)  # ViT-L/14

        fake_context = fake_context.view(batch_size, -1, sourceL)
        fake_contextT = torch.transpose(fake_context, 1, 2).contiguous()

        attnf = torch.bmm(fake_contextT, word)
        attnf = attnf.view(batch_size * sourceL, queryL)
        attnf = nn.Softmax(dim=0)(attnf)
        attnf = attnf.view(batch_size, sourceL, queryL)
        attnf = torch.transpose(attnf, 1, 2).contiguous()
        attnf = attnf.view(batch_size * queryL, sourceL)
        attnf = attnf * 5.0 #cfg.TRAIN.SMOOTH.GAMMA1
        attnf = nn.Softmax(dim=0)(attnf)
        attnf = attnf.view(batch_size, queryL, sourceL)
        attnTf = torch.transpose(attnf, 1, 2).contiguous()

        attnTf = attnTf.mul(sum_word_self)

        weightedContextf = torch.bmm(fake_context, attnTf)
        weightedContextf = weightedContextf + word

        ###################################################
        cur_weiContextf = weightedContextf[0, :, :]
        cur_weiContextf = cur_weiContextf.transpose(0, 1)
        sum_weiContextf = cur_weiContextf.sum(dim=1, keepdim=False)
        soft_weiContextf = nn.Softmax(dim=0)(sum_weiContextf)
        cur_resultf = nn.BCELoss()(soft_weiContextf, fake_word_labels.float())

        fake_result += cur_resultf

    return result, fake_result

#*************************************************************** 20230925
class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
class GALIPNetD(nn.Module):
    def __init__(self, ndf, imsize, ch_size, mixed_precision):
        super(GALIPNetD, self).__init__()
        self.mixed_precision = mixed_precision
        self.DBlocks = nn.ModuleList([
            D_Block(768, 768, 3, 1, 1, res=True, CLIP_feat=True), #ViT-B/32
            D_Block(768, 768, 3, 1, 1, res=True, CLIP_feat=True), #ViT-B/32
            #D_Block(1024, 1024, 3, 1, 1, res=True, CLIP_feat=True), #ViT-L/14
            #D_Block(1024, 1024, 3, 1, 1, res=True, CLIP_feat=True), #ViT-L/14
        ])
        self.main = D_Block(768, 512, 3, 1, 1, res=True, CLIP_feat=False) #ViT-B/32
        #self.main = D_Block(1024, 768, 3, 1, 1, res=True, CLIP_feat=False) #ViT-L/14

    def forward(self, h):
        with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
            out = h[:,0]
            for idx in range(len(self.DBlocks)):
                out = self.DBlocks[idx](out, h[:,idx+1])
            out = self.main(out)
        return out

class D_Block(nn.Module):
    def __init__(self, fin, fout, k, s, p, res, CLIP_feat):
        super(D_Block, self).__init__()
        self.res, self.CLIP_feat = res, CLIP_feat
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        if self.res==True:
            self.gamma = nn.Parameter(torch.zeros(1))
        if self.CLIP_feat==True:
            self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, CLIP_feat=None):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if (self.res==True)and(self.CLIP_feat==True):
            return x + self.gamma*res + self.beta*CLIP_feat
        elif (self.res==True)and(self.CLIP_feat!=True):
            return x + self.gamma*res
        elif (self.res!=True)and(self.CLIP_feat==True):
            return x + self.beta*CLIP_feat
        else:
            return x
#*************************************************************** 20230925