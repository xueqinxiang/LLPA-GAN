# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Generator architecture from
"StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis".
"""

from typing import Union, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch_utils import misc
from torch_utils.ops import upfirdn2d, conv2d_resample, bias_act, fma
from networks.shared import FullyConnectedLayer, MLP
from networks.clip import CLIP

######################################################################################
from collections import OrderedDict
import math


#***************************************************************

#****clip textencodr tes
import transformers
from transformers import CLIPTokenizer, CLIPTextModel
#****clip textencodr test****#

class CLIP_Mapper(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_Mapper, self).__init__()
        model = CLIP.visual
        # print(model)
        self.define_module(model)
        for param in model.parameters():
            param.requires_grad = False

    def define_module(self, model):
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(self, img: torch.Tensor, prompts: torch.Tensor):
        x = img.type(self.dtype)
        prompts = prompts.type(self.dtype)
        grid = x.size(-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        # Local features
        selected = [1,2,3,4,5,6,7,8]
        begin, end = 0, 12
        prompt_idx = 0
        for i in range(begin, end):
            if i in selected:
                prompt = prompts[:,prompt_idx,:].unsqueeze(0)
                prompt_idx = prompt_idx+1
                x = torch.cat((x,prompt), dim=0)
                x = self.transformer.resblocks[i](x)
                x = x[:-1,:,:]
            else:
                x = self.transformer.resblocks[i](x)
        return x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype) # ViT-B/32 model
        #return x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 1024, grid, grid).contiguous().type(img.dtype) # ViT-L/14 model


class CLIP_Adapter(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, G_ch, CLIP_ch, cond_dim, k, s, p, map_num, CLIP, lstm):
        super(CLIP_Adapter, self).__init__()
        self.lstm = lstm
        self.CLIP_ch = CLIP_ch
        self.FBlocks = nn.ModuleList([])
        self.FBlocks.append(M_Block(in_ch, mid_ch, out_ch, cond_dim, k, s, p, lstm))
        for i in range(map_num-1):
            self.FBlocks.append(M_Block(out_ch, mid_ch, out_ch, cond_dim, k, s, p, lstm))
        self.conv_fuse = nn.Conv2d(out_ch, CLIP_ch, 5, 1, 2)
        self.CLIP_ViT = CLIP_Mapper(CLIP)
        self.conv = nn.Conv2d(768, G_ch, 5, 1, 2) #ViT-B/32
        #self.conv = nn.Conv2d(1024, G_ch, 5, 1, 2) #ViT-L/14
        #
        self.fc_prompt = nn.Linear(cond_dim, CLIP_ch*8)

    def forward(self,out,c):
        prompts = self.fc_prompt(c).view(c.size(0),-1,self.CLIP_ch)
        for FBlock in self.FBlocks:
            out = FBlock(out,c)
        fuse_feat = self.conv_fuse(out)
        map_feat = self.CLIP_ViT(fuse_feat,prompts)
        return self.conv(fuse_feat+0.1*map_feat)

class M_Block(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, cond_dim, k, s, p, lstm):
        super(M_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, k, s, p)
        self.fuse1 = DFBLK(cond_dim, mid_ch, lstm)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, k, s, p)
        self.fuse2 = DFBLK(cond_dim, out_ch, lstm)
        self.learnable_sc = in_ch != out_ch
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, text):
        h = self.conv1(h)
        h = self.fuse1(h, text)
        h = self.conv2(h)
        h = self.fuse2(h, text)
        return h

    def forward(self, h, c):
        return self.shortcut(h) + self.residual(h, c)
#***************************************************************

class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
class CLIP_IMG_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_IMG_ENCODER, self).__init__()
        model = CLIP.visual
        # print(model)
        self.define_module(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, model):
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer
        self.ln_post = model.ln_post
        self.proj = model.proj

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def transf_to_CLIP_input(self,inputs):
        device = inputs.device
        if len(inputs.size()) != 4:
            raise ValueError('Expect the (B, C, X, Y) tensor.')
        else:
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            inputs = F.interpolate(inputs*0.5+0.5, size=(224, 224))
            inputs = ((inputs+1)*0.5-mean)/var
            return inputs

    def forward(self, img: torch.Tensor):
        x = self.transf_to_CLIP_input(img)
        x = x.type(self.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid =  x.size(-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        # Local features
        #selected = [1,4,7,12]
        #selected = [1,4,8]
        selected = [2, 5, 9]
        local_features = []
        for i in range(12):
            x = self.transformer.resblocks[i](x)
            if i in selected:
                local_features.append(x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype)) #ViT-B/32
                #local_features.append(x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 1024, grid, grid).contiguous().type(img.dtype)) #ViT-L/14
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return torch.stack(local_features, dim=1), x.type(img.dtype)


class CLIP_TXT_ENCODER(nn.Module):
    def __init__(self, CLIPINT):
        super(CLIP_TXT_ENCODER, self).__init__()
        self.define_module(CLIPINT)
        # print(model)
        for param in self.parameters():
            param.requires_grad = False

        # self.clip = CLIP()
        # del self.clip.model.visual  # only using the text encoder
        # self.c_dim = self.clip.txt_dim

        # self.device = torch.device("cuda", 1)
        # #self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir='.').to(self.device) #ViT-B/32 605M
        # #self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', cache_dir='.') #ViT-B/32
        # self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14', cache_dir='.').to(self.device) #ViT-L/14 1.71G
        # self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14', cache_dir='.') #ViT-L/14

    def define_module(self, CLIP):
        self.transformer = CLIP.transformer
        self.vocab_size = CLIP.vocab_size
        self.token_embedding = CLIP.token_embedding
        self.positional_embedding = CLIP.positional_embedding
        self.ln_final = CLIP.ln_final
        self.text_projection = CLIP.text_projection

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, text):
        # if self.c_dim > 0:
        #     assert text is not None
        #     c = self.clip.model.encode_text(text) #if is_list_of_strings(text) else text #(batch, 768)
        # text_inputs = self.tokenizer(
        #     text, padding="max_length", max_length=self.tokenizer.model_max_length,
        #     truncation=True, return_tensors="pt",
        # )
        # text_input_ids = text_inputs.input_ids
        # # (last_hidden_state, pooled_output, encoder_outputs[1:])
        # text_embeds = self.text_encoder(text_input_ids.to(self.device))
        # text_embeds = text_embeds[0]   #(batch, 77, 768)

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        sent_emb = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return sent_emb, x

#******************* adding trained text encoder **********************************#
class L2MultiheadAttention(nn.Module):
    """ Kim et al. "The Lipschitz Constant of Self-Attention" https://arxiv.org/abs/2006.04710 """
    def __init__(self, embed_dim, num_heads):
        super(L2MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.v_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.q_weight)
        nn.init.zeros_(self.v_weight)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        """
        Args:
            x: (T, N, D)
            attn_mask: (T, T) added to pre-softmax logits.
        """

        T, N, _ = x.shape

        q = torch.einsum("tbm,mhd->tbhd", x, self.q_weight)
        k = torch.einsum("tbm,mhd->tbhd", x, self.q_weight)
        squared_dist = (
            torch.einsum("tbhd,tbhd->tbh", q, q).unsqueeze(1)
            + torch.einsum("sbhd,sbhd->sbh", k, k).unsqueeze(0)
            - 2 * torch.einsum("tbhd,sbhd->tsbh", q, k)
        )
        attn_logits = -squared_dist / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_logits, dim=1)  # (T, S, N, H)
        A = torch.einsum("mhd,nhd->hmn", self.q_weight, self.q_weight) / math.sqrt(
            self.head_dim
        )
        XA = torch.einsum("tbm,hmn->tbhn", x, A)
        PXA = torch.einsum("tsbh,sbhm->tbhm", attn_weights, XA)
        PXAV = torch.einsum("tbhm,mhd->tbhd", PXA, self.v_weight).reshape(
            T, N, self.embed_dim
        )
        return self.out_proj(PXAV)

class TextEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8):
        super().__init__()

        self.embedding = nn.Linear(in_dim, out_dim)
        self.l2attn = L2MultiheadAttention(out_dim, num_heads)
        self.ff = nn.Sequential(
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.ln1 = nn.LayerNorm(out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, text_embeds):
        text_embeds = self.embedding(text_embeds)
        text_embeds = text_embeds.unsqueeze(1)
        out1 = self.l2attn(text_embeds)
        out1 = self.ln1(out1 + text_embeds)
        out2 = self.ff(out1)
        output = self.ln2(out2 + out1)
        return output
#******************* adding trained text encoder **********************************#

class NetC(nn.Module):
    def __init__(self, ndf, cond_dim, mixed_precision):
        super(NetC, self).__init__()
        self.cond_dim = cond_dim
        self.mixed_precision = mixed_precision
        self.joint_conv = nn.Sequential(
            #*****nn.Conv2d(512+512, 128, 4, 1, 0, bias=False),
            nn.Linear(980 + 768, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            #*****nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            )

    def forward(self, out, cond):
        with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
            #*****cond = cond.view(-1, self.cond_dim, 1, 1)
            #*****cond = cond.repeat(1, 1, 7, 7)
            h_c_code = torch.cat((out, cond), 1)
            out = self.joint_conv(h_c_code)
        return out


def is_list_of_strings(arr: Any) -> bool:
    if arr is None: return False
    is_list = isinstance(arr, list) or isinstance(arr, np.ndarray) or  isinstance(arr, tuple)
    entry_is_str = isinstance(arr[0], str)
    return is_list and entry_is_str


def normalize_2nd_moment(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

class MogLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, mog_interation):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.mog_interations = mog_interation

        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))

        #Mogrifiers
        self.Q = nn.Parameter(torch.Tensor(hidden_sz, input_sz))
        self.R = nn.Parameter(torch.Tensor(input_sz, hidden_sz))

        self.init_weights()
        self.noise2h = nn.Linear(100, 612)  # ViT-B/32
        self.noise2c = nn.Linear(100, 612)  # ViT-B/32
        #self.noise2h = nn.Linear(100,868)  #ViT-L/14
        #self.noise2c = nn.Linear(100,868)  #ViT-L/14
        #self.init_hidden()
        self.hidden_seq = []
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def init_hidden(self,noise):
        h_t = self.noise2h(noise)
        c_t = self.noise2c(noise)

        self.c_t = c_t
        self.h_t = h_t

    def mogrify(self, xt, ht):
        for i in range(1, self.mog_interations+1):
            if(i % 2 == 0):
                ht = (2*torch.sigmoid(xt @ self.R)*ht)
            else:
                xt = (2*torch.sigmoid(ht @ self.Q)*xt)
        return xt, ht

    def forward(self, x):
        """Assumes x is of shape (batch, sequence, feature)"""
        #bs, seq_sz, _ = x.size()
        #hidden_seq = []
        c_t = self.c_t
        h_t = self.h_t
        HS = self.hidden_size
#        x_t = x[:, t, :]
        x_t = x
        x_t, h_t = self.mogrify(x_t, h_t)

        # batch the computations into a single matrix multiplication
        gates = x_t @ self.W + h_t @ self.U + self.bias
        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, :HS]), # input
            torch.sigmoid(gates[:, HS:HS*2]), # forget
            torch.tanh(gates[:, HS*2:HS*3]),
            torch.sigmoid(gates[:, HS*3:]), # output
        )
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        self.c_t = c_t
        self.h_t = h_t

        return h_t, c_t

class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        clip,                         # Clip TextEncoder model
        z_dim: int,                   # Input latent (Z) dimensionality, 0 = no latent.
        conditional: bool = True,     # Text conditional?
        num_layers: int = 2,          # Number of mapping layers.
        activation: str = 'lrelu',    # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 0.01,  # Learning rate multiplier for the mapping layers.
        x_avg_beta: float = 0.995,    # Decay for tracking the moving average of W during training.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.x_avg_beta = x_avg_beta
        self.num_ws = None

        self.mlp = MLP([z_dim]*(num_layers+1), activation=activation,
                       lr_multiplier=lr_multiplier, linear_out=True)

        if conditional:
            self.clip = CLIP()
            #self.clip = clip
            del self.clip.model.visual # only using the text encoder
            #del self.clip.visual
            self.c_dim = self.clip.txt_dim
        else:
            self.c_dim = 0

        self.w_dim = self.c_dim + self.z_dim
        self.register_buffer('x_avg', torch.zeros([self.z_dim]))

    def forward(
        self,
        z: torch.Tensor,
        c: Union[None, torch.Tensor, list[str]],
        truncation_psi: float = 1.0,
    ) -> torch.Tensor:
        misc.assert_shape(z, [None, self.z_dim])

        # Forward pass.
        x = self.mlp(normalize_2nd_moment(z))

        # Update moving average.
        if self.x_avg_beta is not None and self.training:
            self.x_avg.copy_(x.detach().mean(0).lerp(self.x_avg, self.x_avg_beta))

        # Apply truncation.
        if truncation_psi != 1:
            assert self.x_avg_beta is not None
            x = self.x_avg.lerp(x, truncation_psi)

        # Build latent.
        if self.c_dim > 0:
            assert c is not None
            c = self.clip.encode_text(c) if is_list_of_strings(c) else c
            w = torch.cat([x, c], 1)
        else:
            w = x

        # Broadcast latent codes.
        if self.num_ws is not None:
            w = w.unsqueeze(1).repeat([1, self.num_ws, 1])

        return w

class MappingNetwork_new(torch.nn.Module):
    def __init__(
        self,
        in_ch, mid_ch, out_ch, G_ch, CLIP_ch, cond_dim, k, s, p, map_num, lstm,
        clip,                         # Clip TextEncoder model
        z_dim: int,                   # Input latent (Z) dimensionality, 0 = no latent.
        conditional: bool = True,     # Text conditional?
        num_layers: int = 2,          # Number of mapping layers.
        activation: str = 'lrelu',    # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 0.01,  # Learning rate multiplier for the mapping layers.
        x_avg_beta: float = 0.995,    # Decay for tracking the moving average of W during training.
    ):
        super().__init__()

        #################--new
        self.lstm = lstm
        self.CLIP_ch = CLIP_ch
        self.FBlocks = nn.ModuleList([])
        self.FBlocks.append(M_Block(in_ch, mid_ch, out_ch, cond_dim, k, s, p, lstm))
        for i in range(map_num-1):
            self.FBlocks.append(M_Block(out_ch, mid_ch, out_ch, cond_dim, k, s, p, lstm))
        self.conv_fuse = nn.Conv2d(out_ch, CLIP_ch, 5, 1, 2)
        self.conv = nn.Conv2d(768, G_ch, 5, 1, 2) #ViT-B/32
        #self.conv = nn.Conv2d(1024, G_ch, 5, 1, 2) #ViT-L/14
        #################--new

        # ####add visual fusion
        # self.fc_prompt = nn.Linear(cond_dim, CLIP_ch * 8)
        # self.CLIP_ViT = CLIP_Mapper(clip)
        # ####add visual fusion

        self.z_dim = z_dim
        self.x_avg_beta = x_avg_beta
        self.num_ws = None

        self.mlp = MLP([z_dim]*(num_layers+1), activation=activation,
                       lr_multiplier=lr_multiplier, linear_out=True)

        if conditional:
            self.clip = CLIP()
            #self.clip = clip
            del self.clip.model.visual # only using the text encoder
            #del self.clip.visual
            self.c_dim = self.clip.txt_dim
        else:
            self.c_dim = 0

        self.w_dim = self.c_dim + self.z_dim
        self.register_buffer('x_avg', torch.zeros([self.z_dim]))

    def forward(
        self,
        z_expand,
        z: torch.Tensor,
        c: Union[None, torch.Tensor, list[str]],
        truncation_psi: float = 1.0,
    ) -> torch.Tensor:
        cond = torch.cat((z, c), dim=1)  #####20230923

        misc.assert_shape(z, [None, self.z_dim])

        # Forward pass.
        x = self.mlp(normalize_2nd_moment(z))

        # Update moving average.
        if self.x_avg_beta is not None and self.training:
            self.x_avg.copy_(x.detach().mean(0).lerp(self.x_avg, self.x_avg_beta))

        # Apply truncation.
        if truncation_psi != 1:
            assert self.x_avg_beta is not None
            x = self.x_avg.lerp(x, truncation_psi)

        # Build latent.
        if self.c_dim > 0:
            assert c is not None
            c = self.clip.encode_text(c) if is_list_of_strings(c) else c
            w = torch.cat([x, c], 1)
        else:
            w = x

        # Broadcast latent codes.
        if self.num_ws is not None:
            w = w.unsqueeze(1).repeat([1, self.num_ws, 1])

        #return w

        #################--new
        out = z_expand
        for FBlock in self.FBlocks:
            #out = FBlock(out, w)
            out = FBlock(out, cond) #####20230923
        fuse_feat = self.conv_fuse(out)
        return self.conv(fuse_feat), w #####20230923
        #################--new

        # ####add visual fusion
        # prompts = self.fc_prompt(cond).view(cond.size(0), -1, self.CLIP_ch)
        # map_feat = self.CLIP_ViT(fuse_feat,prompts)
        # return self.conv(fuse_feat+0.1*map_feat), w
        # ####add visual fusion

class MappingNetwork_addTrainedTextEncoder(torch.nn.Module):
    def __init__(
        self,
        in_ch, mid_ch, out_ch, G_ch, CLIP_ch, cond_dim, k, s, p, map_num, lstm,
        clip,                         # Clip TextEncoder model
        z_dim: int,                   # Input latent (Z) dimensionality, 0 = no latent.
        conditional: bool = True,     # Text conditional?
        num_layers: int = 2,          # Number of mapping layers.
        activation: str = 'lrelu',    # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 0.01,  # Learning rate multiplier for the mapping layers.
        x_avg_beta: float = 0.995,    # Decay for tracking the moving average of W during training.
    ):
        super().__init__()

        self.lstm = lstm
        self.CLIP_ch = CLIP_ch
        self.FBlocks = nn.ModuleList([])
        self.FBlocks.append(M_Block(in_ch, mid_ch, out_ch, cond_dim, k, s, p, lstm))
        for i in range(map_num-1):
            self.FBlocks.append(M_Block(out_ch, mid_ch, out_ch, cond_dim, k, s, p, lstm))
        self.conv_fuse = nn.Conv2d(out_ch, CLIP_ch, 5, 1, 2)
        self.conv = nn.Conv2d(768, G_ch, 5, 1, 2) #ViT-B/32
        #self.conv = nn.Conv2d(1024, G_ch, 5, 1, 2) #ViT-L/14

        tin_dim = cond_dim - z_dim #768 #ViT-L/14
        tout_dim = tin_dim #256
        self.text_encoder = TextEncoder(tin_dim, tout_dim)

        # ####add visual fusion
        # self.fc_prompt = nn.Linear(cond_dim, CLIP_ch * 8)
        # self.CLIP_ViT = CLIP_Mapper(clip)
        # ####add visual fusion

    def forward(
        self,
        z_expand,
        z: torch.Tensor,
        c: Union[None, torch.Tensor, list[str]],
        truncation_psi: float = 1.0,
    ) -> torch.Tensor:
        condtest = torch.cat((z, c), dim=1)
        c = c.float() #float16 -> float32
        text_embeds = c
        seq_len = text_embeds.shape[1]
        text_embeds = self.text_encoder(text_embeds)
        # t_local, t_global = torch.split(text_embeds, [seq_len - 1, 1], dim=1)
        # cond = torch.cat((z, t_global), dim=1)
        # w = torch.cat((z, t_local), dim=1)

        text_embeds = text_embeds.squeeze(1)
        cond = torch.cat((z, text_embeds), dim=1)
        w = torch.cat((z, text_embeds), dim=1)

        out = z_expand
        for FBlock in self.FBlocks:
            out = FBlock(out, cond)
            #out = FBlock(out, condtest)
        fuse_feat = self.conv_fuse(out)
        return self.conv(fuse_feat), w

        ####add visual fusion
        # prompts = self.fc_prompt(cond).view(cond.size(0), -1, self.CLIP_ch)
        # #prompts = self.fc_prompt(condtest).view(condtest.size(0), -1, self.CLIP_ch)
        # map_feat = self.CLIP_ViT(fuse_feat,prompts)
        # return self.conv(fuse_feat+0.1*map_feat), w
        # #return self.conv(fuse_feat + 0.1 * map_feat), condtest
        ####add visual fusion


class NetG(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size, mixed_precision, CLIP, lstm = None):
        super(NetG, self).__init__()
        self.ngf = ngf
        self.mixed_precision = mixed_precision
        self.lstm = lstm
        self.c_dim = 768

        # # build CLIP Mapper1
        # self.mapping = MappingNetwork(CLIP, z_dim=nz, conditional=True)
        # self.c_dim = self.mapping.c_dim
        # self.fc = nn.Linear(cond_dim+nz, ngf * 8 * 8 * 8) #connect to GBlocks
        # # build CLIP Mapper1

        # build CLIP Mapper2
        # self.code_sz, self.code_ch, self.mid_ch = 7, 64, 32#7, 64, 32
        # self.CLIP_ch = 768 #768 #1024
        # self.fc_code = nn.Linear(nz, self.code_sz*self.code_sz*self.code_ch)
        # self.mapping = CLIP_Adapter(self.code_ch, self.mid_ch, self.code_ch, ngf*8, self.CLIP_ch, cond_dim+nz, 3, 1, 1, 4, CLIP, lstm)
        # build CLIP Mapper2

        # build CLIP Mapper3
        # self.code_sz, self.code_ch, self.mid_ch = 16, 64, 32
        # self.CLIP_ch = 1024 #ViT-L/14
        # #self.CLIP_ch = 768  #ViT-B/32
        # self.fc_code = nn.Linear(nz, self.code_sz*self.code_sz*self.code_ch)
        # self.mapping = MappingNetwork_new(self.code_ch, self.mid_ch, self.code_ch, ngf*8, self.CLIP_ch, cond_dim+nz, 3, 1, 1, 4, lstm, CLIP, z_dim=nz, conditional=True)
        # build CLIP Mapper3

        # build CLIP Mapper4
        #self.code_sz, self.code_ch, self.mid_ch = 16, 64, 32 #ViT-L/14
        self.code_sz, self.code_ch, self.mid_ch = 7, 64, 32  #ViT-B/32
        #self.CLIP_ch = 1024 #ViT-L/14
        self.CLIP_ch = 768  # ViT-B/32
        self.fc_code = nn.Linear(nz, self.code_sz*self.code_sz*self.code_ch)
        self.mapping = MappingNetwork_addTrainedTextEncoder(self.code_ch, self.mid_ch, self.code_ch, ngf*8, self.CLIP_ch, cond_dim+nz, 3, 1, 1, 4, lstm, CLIP, z_dim=nz, conditional=True)
        # build CLIP Mapper3

        # build GBlocks
        self.GBlocks = nn.ModuleList([])
        in_out_pairs = list(get_G_in_out_chs(ngf, imsize))
        imsize = 4
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            if idx<(len(in_out_pairs)-1):
                imsize = imsize*2
            else:
                imsize = 224
            self.GBlocks.append(G_Block(cond_dim+nz, in_ch, out_ch, imsize, lstm))
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(out_ch, ch_size, 3, 1, 1),
            #nn.Tanh(),
            )
        # self.to_rgb = nn.Sequential(
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(out_ch, 256, 3, 1, 1),
        #     nn.Conv2d(256, ch_size, 1),
        #     #nn.Tanh(),
        #     )

    def forward(self, noise, c, eval=False): # x=noise, c=ent_emb
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else dummy_context_mgr() as mp:
            cond = torch.cat((noise, c), dim=1)

            # # build CLIP Mapper1
            # mapout = self.mapping(noise, c, truncation_psi=1.0)
            # out = self.fc(mapout)
            # out = out.view(c.size(0), 8*self.ngf, 8, 8)
            # # build CLIP Mapper1

            # build CLIP Mapper2
            #out = self.mapping(self.fc_code(noise).view(noise.size(0), self.code_ch, self.code_sz, self.code_sz), cond)
            # build CLIP Mapper2

            # build CLIP Mapper3
            out, clip_w = self.mapping(self.fc_code(noise).view(noise.size(0), self.code_ch, self.code_sz, self.code_sz), noise, c, truncation_psi=1.0)
            # build CLIP Mapper3

            # fuse text and visual features
            for GBlock in self.GBlocks:
                out = GBlock(out, cond)
                #out = GBlock(out, clip_w) #####20230923
            # convert to RGB image
            out = self.to_rgb(out)
        return out

class NetC(nn.Module):
    def __init__(self, ndf, cond_dim, mixed_precision):
        super(NetC, self).__init__()
        self.cond_dim = cond_dim
        self.mixed_precision = mixed_precision
        self.joint_conv = nn.Sequential(
            nn.Conv2d(512 + 512, 128, 4, 1, 0, bias=False), #ViT-B/32
            #nn.Conv2d(768 + 768, 128, 4, 1, 0, bias=False), #ViT-L/14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            )

    def forward(self, out, cond):
        with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
            cond = cond.view(-1, self.cond_dim, 1, 1)
            cond = cond.repeat(1, 1, 7, 7)   #ViT-B/32
            #cond = cond.repeat(1, 1, 16, 16)  #ViT-L/14
            h_c_code = torch.cat((out, cond), 1)
            out = self.joint_conv(h_c_code)
        return out

class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, imsize,lstm):
        super(G_Block, self).__init__()
        self.imsize = imsize
        self.learnable_sc = in_ch != out_ch
        self.lstm = lstm
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch, lstm)
        self.fuse2 = DFBLK(cond_dim, out_ch, lstm)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):
        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def forward(self, h, y):
        h = F.interpolate(h, size=(self.imsize, self.imsize))
        return self.shortcut(h) + self.residual(h, y)

class DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch, lstm):
        super(DFBLK, self).__init__()
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)
        self.lstm = lstm

    def forward(self, x, y=None):
        lstm_input = y
        y,_  =  self.lstm(lstm_input)
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        y, _ = self.lstm(lstm_input)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return h


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs


def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs