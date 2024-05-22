# cf. MAE
import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 如果t是元组则返回t，否则返回(t,t)
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim) # TODO: 实例归一化对比

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        # 根据指定的深度创建Transformer层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    
class VisionTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'mean', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        '''视觉Transformer

        :param tuple  image_size: 图片大小(h, w)
        :param tuple patch_size: 块大小,patch一般为正方形
        :param int dim: Transformer所有层潜在向量的大小
        :param int depth: Transformer内部层数,取12
        :param int heads: 注意力机制中的头数,取8
        :param int mlp_dim: MLP隐藏层维度
        :param str pool: 池化策略, defaults to 'mean'
        :param int channels: 输入通道数, defaults to 1
        :param int dim_head: 注意力头的数据维度, defaults to 64
        :param float dropout: 位置编码后dropout, defaults to 0.
        :param float emb_dropout: 注意力dropout, defaults to 0.
        '''
        
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # 确保正常分块和汇集方式正确
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # 切分成小块并展开成向量，线性层规整为dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), # 张量重排
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # TODO:还有余弦位置编码方式
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # dim为Transformer输入输出维度
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        
        # 标准化 or nn.Identity()
        self.to_latent = nn.LayerNorm(dim)


    def autoencoder(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape # n为patch数

        # cls_token所对应的输出向量将被用于分类任务
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) # TODO: 改成适用图像复原任务
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] # TODO: 两种汇集方式作用

        x = self.to_latent(x)
        return x
    
    def decoder(self, x):
        x += self.pos_embedding
        decoder_embeeding = self.transformer(x)
        
        return decoder_embeeding


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct the normalization for each patchs
        """
        super(AdaptiveLayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size))  # 缩放
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # 平移
        self.variance_epsilon = eps
       
    def forward(self, x):
        u = x[:, :].mean(-1, keepdim=True)
        s = (x[:, :] - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)  # 归一化
        return self.gamma * x + self.beta  # 参数化重构，寻找最优表征

class RecFieldViT(nn.Module):
    def __init__(self,
                 image_size = (112, 192),
                 patch_size = 16,
                 in_channels = 1,
                 out_channels = 1,
                 encoder_dim = 1024,
                 encoder_depth = 24,
                 encoder_heads = 16,
                 decoder_dim = 512,
                 decoder_depth = 8,
                 decoder_heads = 16,
                 ) -> None:
        super().__init__()
        image_height, image_width = pair(image_size)
        self.num_patch = (image_height // patch_size, image_width // patch_size)
        base_cfg = dict(
            image_size = image_size, 
            patch_size = patch_size,
            channels = in_channels,
            mlp_dim = 4,
            dropout = 0,
            emb_dropout = 0,
            pool = 'mean'
        )
        encoder_dict = dict(
            dim = encoder_dim, 
            depth = encoder_depth, 
            heads = encoder_heads
        )
        decoder_dict = dict(
            dim = decoder_dim,
            depth = decoder_depth,
            heads = decoder_heads
        )
        
        ENCODER_CFG = {**base_cfg, **encoder_dict}
        DECODER_CFG = {**base_cfg, **decoder_dict}
        
        # vit embeeding 
        self.Encoder = VisionTransformer(**ENCODER_CFG)
        self.Decoder = VisionTransformer(**DECODER_CFG)
        
        output_dim = out_channels * patch_size * patch_size
        self.proj = nn.Linear(encoder_dim, decoder_dim) # 线性层将作用于最后一个维度上
        self.restruction = nn.Linear(decoder_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.patch_norm = AdaptiveLayerNorm(output_dim)
        # 反卷积层：(N, output_dim, H, W) -> (N, out_channels, H*patch_size, W*patch_size)
        self.unconv = nn.ConvTranspose2d(output_dim, out_channels, patch_size, patch_size)
        
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
            
    def forward(self, x):
        t_encoded = self.Encoder.autoencoder(x)
        proj_t_encoded = self.proj(t_encoded)
        t_decoded = self.Decoder.decoder(proj_t_encoded)
        
        outputs = self.restruction(t_decoded)
        # cls_token = outputs[:, 0, :] # 分类任务用
        image_token = outputs[:, 1:, :] # (b, num_patches, patches_vector)
        image_norm_token = self.patch_norm(image_token)
        n, l, dim = image_norm_token.shape
        image_norm_token = image_norm_token.view(-1, self.num_patch[0], self.num_patch[1], dim).permute(0, 3, 1, 2) # 变回NCHW
        restore_image = self.unconv(image_norm_token)
        
        return restore_image