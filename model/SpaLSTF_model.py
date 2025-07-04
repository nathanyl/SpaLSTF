import torch
import torch.nn as nn
import numpy as np
import math
import einops
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import sys
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class XCA(nn.Module):
    """Cross-Covariance Attention (XCA) with channels updated using a weighted sum based on the cross-covariance matrix."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # Temperature scaling for softmax

        # Define q, k, v projection layers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Reshape to (3, Batch, Heads, Tokens, Features per Head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Normalize Q and K for channel-level attention
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Compute cross-covariance attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (B, Heads, Channels, Channels)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.squeeze(0)
        return x


    
def modulate(x, shift, scale):

    res = x * (1 + scale) + shift
    return res

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    time emb to frequency_embedding_size dim, then to hidden_size
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTblock(nn.Module):
    # adaBN -> attn -> mlp
    def __init__(self,
                 feature_dim=2000,
                 mlp_ratio=4.0,
                 num_heads=10,
                 **kwargs) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        self.attn = XCA(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs)

        self.norm2 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU()

        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = Mlp(in_features=feature_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 6 * feature_dim, bias=True)
        )

    def forward(self,x,c):
        # Project condition to 6 * hiddensize and then slice it into 6 parts along the column.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # attention blk
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # mlp
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT. adaLN -> linear
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Projected into output shape
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        # shift scale
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # shift scale
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        # projection
        x = self.linear(x)
        return x      

BaseBlock = {'dit':DiTblock}


class CustomGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CustomGCNConv, self).__init__(aggr='add', **kwargs)  # Aggregation: 'add'
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))  # Parameter matrix for feature pre-multiplication
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        # 1. Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 2. Compute normalized adjacency matrix.
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 3. Perform feature transformation (pre-multiplication).
        x = torch.matmul(self.weight, x)  # Pre-multiply feature with weight matrix W

        # 4. Message passing.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j



# class GCNEncoder(torch.nn.Module):
#     def __init__(self, num_features, hidden_dim, dropout=0.2, is_training=False):
#         super(GCNEncoder, self).__init__()
#         self.conv1 = CustomGCNConv(num_features, hidden_dim)
#         self.conv2 = CustomGCNConv(hidden_dim, hidden_dim)
#         self.linear = nn.Linear(hidden_dim, hidden_dim)
#         self.is_training = is_training
#         self.dropout = nn.Dropout(dropout)
#
#     def embed(self, x, edge_index):
#         # Graph convolution layers
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.dropout(x)
#         x = self.conv2(x, edge_index)
#         x = torch.relu(x)
#         return x
#
#     def forward(self, x, edge_index):
#         # Compute embeddings for the entire graph
#         embeddings = self.embed(x, edge_index)
#         # Optional linear transformation on node embeddings
#         x = self.linear(embeddings.T)
#         return x, embeddings

class SpaLSTF(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 depth,
                 dit_type,
                 num_heads,
                 classes,
                 mlp_ratio=4.0,
                 lstm_hidden_size=128,  # 新增：LSTM隐藏层大小
                 lstm_num_layers=2,  # 新增：LSTM层数
                 **kwargs) -> None:
        """ denoising model

        Args:
            input_size (_type_): input dim
            hidden_size (_type_): scale input to hidden dim
            depth (_type_): dit block num
            dit_type (_type_): which type block to use
            num_heads (_type_): transformer head num
            classes (_type_): class num
            mlp_ratio (float, optional): _description_. Defaults to 4.0.
        """        
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.classes = classes
        self.mlp_ratio = mlp_ratio
        self.dit_type = dit_type

        self.in_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size)
        )
        self.cond_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size)
        )

        # celltype emb
        self.condi_emb = nn.Embedding(classes, hidden_size)

        # time emb
        self.time_emb = TimestepEmbedder(hidden_size=self.hidden_size)

        # DiT block
        self.blks = nn.ModuleList([
            BaseBlock[dit_type](self.hidden_size, mlp_ratio=self.mlp_ratio, num_heads=self.num_heads) for _ in
            range(self.depth)
        ])
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True  # 确保输入为 (batch_size, seq_len, feature_dim)
        )
        # self.gcn_encoder = GCNEncoder(self.input_size, self.input_size,)
        # out layer
        self.trans = nn.Sequential(
            nn.Linear(lstm_hidden_size, hidden_size)
        )
        self.out_layer = FinalLayer(self.hidden_size, self.input_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            # elif isinstance(module, CustomGCNConv):  # 图卷积模块的初始化
            #     module.reset_parameters()
            elif isinstance(module, nn.LSTM):
                # 初始化 LSTM 层
                for name, param in module.named_parameters():
                    if "weight" in name:
                        torch.nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        torch.nn.init.constant_(param, 0)

        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.condi_emb.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.dit_type == 'dit':
            for block in self.blks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.linear.weight, 0)
        nn.init.constant_(self.out_layer.linear.bias, 0)

    def forward(self, x, t, y, edge_index, **kwargs):
        x = x.float()
        t = self.time_emb(t)

        y = self.cond_layer(y)
        y = y.unsqueeze(0)  # 从 (n, g) 变成 (1, n, g)
        # LSTM 前向传播
        y, _ = self.lstm(y)  # 输出维度为 (1, n, lstm_hidden_size)

        y = y.squeeze(0)  # 移除时间步维度，回到 (n, lstm_hidden_size)
        y = self.trans(y)
        # z = self.condi_emb(z)
        c = t + y
        # c = t
        # x, _ = self.gcn_encoder(x.T, edge_index)
        x = self.in_layer(x)


        for blk in self.blks:
            x = blk(x, c)

        x = self.out_layer(x, c)
        return x
