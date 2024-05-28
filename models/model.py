import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

# import matplotlib.pyplot as plt

# classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.visualize = False

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        # if self.visualize:
        #     self.save_attn(attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)

    # def save_attn(self, attn):
    #     attn = attn.detach().cpu().numpy()
    #     fig, ax = plt.subplots()
    #     data_idx = 0
    #     head_idx = 0
    #     cax = ax.matshow(attn[data_idx, head_idx], cmap="viridis")
    #     plt.colorbar(cax, ax=ax)
    #     plt.title("Attention Map")
    #     plt.savefig("attention_map.png")
    #     plt.show()
    #     plt.close(fig)


class RowColTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        nfeats,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        style="col",
        visualize=False,
    ):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        self.style = style

        self.dim = dim
        nfeats = nfeats

        for _ in range(depth):
            if self.style == "colrow":
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim,
                                Residual(
                                    Attention(
                                        dim,
                                        heads=heads,
                                        dim_head=dim_head,
                                        dropout=attn_dropout,
                                    )
                                ),
                            ),
                            PreNorm(
                                dim, Residual(FeedForward(dim, dropout=ff_dropout))
                            ),
                            PreNorm(
                                dim * nfeats,
                                Residual(
                                    Attention(
                                        dim * nfeats,
                                        heads=heads,
                                        dim_head=64,
                                        dropout=attn_dropout,
                                    )
                                ),
                            ),
                            PreNorm(
                                dim * nfeats,
                                Residual(FeedForward(dim * nfeats, dropout=ff_dropout)),
                            ),
                        ]
                    )
                )
            else:
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim * nfeats,
                                Residual(
                                    Attention(
                                        dim * nfeats,
                                        heads=heads,
                                        dim_head=64,
                                        dropout=attn_dropout,
                                    )
                                ),
                            ),
                            PreNorm(
                                dim * nfeats,
                                Residual(FeedForward(dim * nfeats, dropout=ff_dropout)),
                            ),
                        ]
                    )
                )

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape
        if self.style == "colrow":
            for attn1, ff1, attn2, ff2 in self.layers:
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, "b n d -> 1 b (n d)")
                x = attn2(x)
                x = ff2(x)
                x = rearrange(x, "1 b (n d) -> b n d", n=n)
        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, "b n d -> 1 b (n d)")
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, "1 b (n d) -> b n d", n=n)
        return x


# transformer
class Transformer(nn.Module):
    def __init__(
        self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Residual(
                                Attention(
                                    dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=attn_dropout,
                                )
                            ),
                        ),
                        PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    ]
                )
            )

    def forward(self, x, x_cont=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


# mlp
class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]), nn.ReLU(), nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
