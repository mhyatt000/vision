"""borrowed from https://github.com/lucidrains/vit-pytorch"""

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim, ldim, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, ldim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ldim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, layers, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        PreNorm(dim, MLP(dim, mlp_dim, dropout=dropout)),
                    ]
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x):
        for attn, fc in self.layers:
            x = attn(x) + x
            x = fc(x) + x
        return x


""" from theaisummer.com
1. Split an image into patches
2. Flatten the patches
3. Produce lower-dimensional linear embeddings from the flattened patches
4. Add positional embeddings
5. Feed the sequence as an input to a standard transformer encoder
6. Pretrain the model with image labels (fully supervised on a huge dataset)
7. Finetune on the downstream dataset for image classification
"""


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        i=256,
        p=32,
        nclasses=1000,
        dim=1024,
        layers=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        pool="cls",
        channels=3,
        dim_head=64,
    ):

        super(VisionTransformer, self).__init__()

        ex = f"Image dimensions must be divisible by the patch size {i}%{p}!= {i%p}"
        assert i % p == 0, ex
        num_patches = i % p
        patch_dim = channels * p ** 2

        ex = "pool type must be either cls (cls token) or mean (mean pooling)"
        assert pool in {"cls", "mean"}, ex

        # flattens image patches ... w*h  -> n*pw*ph
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, layers, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, nclasses))

    def forward(self, img):

        # split into patches
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # flatten patches
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional embedding
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        # input to nn
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class ViT(VisionTransformer):
    """ViT Base"""

    """
    Model       Layers  Hidden size D   MLP size    Heads   Params
    ViT-Base    12      768             3072        12      86M
    ViT-Large   24      1024            4096        16      307M
    ViT-Huge    32      1280            5120        16      632M
    """

    def __init__(self, size="B", p=16, **kwargs):

        kwargs2 = {
            "B": {"layers": 12, "dim": 768, "mlp_dim": 3072, "heads": 12},
            "L": {"layers": 24, "dim": 1024, "mlp_dim": 4096, "heads": 16},
            "H": {"layers": 32, "dim": 1280, "mlp_dim": 5120, "heads": 16},
        }

        super(ViT, self).__init__(p=p, **kwargs, **kwargs2[size])
