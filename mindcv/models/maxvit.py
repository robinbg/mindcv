from mindspore import Tensor
from mindspore import nn
from mindspore import Parameter
from mindspore.nn import CellList
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import Normal
import mindspore.numpy as mnp
import mindspore

from .registry import register_model
# helpers
__all__ = [
    "MaxViT",
    "maxvit_t",
    "maxvit_s",
    "maxvit_b",
    "maxvit_l",
    "maxvit_xl"
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "first_conv": "patch_embed.proj",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "maxvit_t": _cfg(url=""),
    "maxvit_s": _cfg(url=""),
    "maxvit_b": _cfg(url=""),
    "maxvit_l": _cfg(url=""),
    "maxvit_xl":_cfg(url="")
}


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class PreNormResidual(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm((dim,))
        self.fn = fn

    def construct(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Cell):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.SequentialCell([
            nn.Dense(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Dense(inner_dim, dim),
            nn.Dropout(dropout)
        ])
    def construct(self, x):
        return self.net(x)
class SqueezeExcitation(nn.Cell):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.SequentialCell([
            nn.Dense(dim, hidden_dim, has_bias=False),
            nn.SiLU(),
            nn.Dense(hidden_dim, dim, has_bias=False),
            nn.Sigmoid()
        ])

    def construct(self, x):
        y = P.ReduceMean()(x, (2,3))
        y = self.gate(y)
        y = P.Reshape()(y, (y.shape[0], y.shape[1], 1, 1))
        return x * y


class MBConvResidual(nn.Cell):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def construct(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Cell):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def construct(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = Tensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)
        
def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.SequentialCell([
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, group = hidden_dim, pad_mode = 'pad'),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    ])

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net


class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Dense(dim, dim * 3, has_bias=False)

        self.attend = nn.SequentialCell([
            nn.Softmax(axis = -1),
            nn.Dropout(dropout)
        ])

        self.to_out = nn.SequentialCell([
            nn.Dense(dim, dim, has_bias=False),
            nn.Dropout(dropout)
        ])

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = mnp.arange(window_size)
        grid = mnp.stack(mnp.meshgrid(pos, pos, indexing = 'ij'))
        grid = grid.transpose((1,2,0))
        grid = grid.reshape((-1, grid.shape[2]))
        grid_1 = grid.reshape((grid.shape[0], 1, grid.shape[1]))
        grid_2 = grid.reshape((1, grid.shape[0], grid.shape[1]))
        rel_pos = grid_1 - grid_2
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * Tensor([2 * window_size - 1, 1]).astype(mindspore.float32)).sum(axis = -1).astype(mindspore.int64)

        self.rel_pos_indices = Parameter(rel_pos_indices, name='rel_pos_indices')

    def construct(self, p):
        batch, height, width, window_height, window_width, _ = p.shape
        h = self.heads
        b,x,y,w1,w2,d = p.shape
        p = p.reshape((b*x*y, w1*w2,d))
        q, k, v = self.to_qkv(p).split(-1, 3)
        q = q.reshape((q.shape[0], q.shape[1], h, -1)).transpose((0,2,1,3))
        k = k.reshape((k.shape[0], k.shape[1], h, -1)).transpose((0,2,1,3))
        v = v.reshape((v.shape[0], v.shape[1], h, -1)).transpose((0,2,1,3))
        q = q * self.scale
        sim = P.BatchMatMul(transpose_b=True)(q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + bias.transpose()
        attn = self.attend(sim)
        out = P.BatchMatMul()(attn, v)
        w1, w2 = window_height, window_width
        out = out.transpose(0,2,1,3)
        out = out.reshape((out.shape[0], w1, w2, -1))
        out = self.to_out(out)
        out = out.reshape((-1, height, width, w1, w2, out.shape[-1]))
        return out

class Rearrange_1(nn.Cell):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def construct(self, x):
        b,d,a,c = x.shape
        x = x.reshape((b,d, a//self.w, self.w, c//self.w, self.w))
        return x.transpose((0, 2, 4, 3, 5, 1))

class Rearrange_2(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, t):
        b,x,y,w1,w2,d = t.shape
        t = t.transpose((0 ,5, 3, 1, 4, 2))
        return t.reshape((b,d, w1*x, w2*y))
class MaxViT(nn.Cell):
    def __init__(
        self,
        *,
        num_classes,
        num_blocks,
        num_channels,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        assert isinstance(num_blocks, tuple), 'num_blocks needs to be tuple if integers indicating number of transformer blocks at that stage'

        # convolutional stem


        self.conv_stem = nn.SequentialCell([
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1, pad_mode='pad'),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1, pad_mode='pad')
        ])

        # variables


        num_stages = len(num_blocks) - 1
        num_blocks = num_blocks[1:]
        dims = tuple(num_channels)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = CellList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, num_blocks)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                block = nn.SequentialCell([
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),

                    Rearrange_1(w),  # block-like attention
                    PreNormResidual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange_2(),

                    Rearrange_1(w),  # grid-like attention
                    PreNormResidual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange_2(),
                ])

                self.layers.append(block)

        # mlp head out

        self.mlp_head = nn.SequentialCell([
            nn.LayerNorm((dims[-1],)),
            nn.Dense(dims[-1], num_classes)
        ])

    def construct(self, x):
        x = self.conv_stem(x)
        for stage in self.layers:
            x = stage(x)
        x = P.ReduceMean(keep_dims=False)(x, (2,3))
        assert len(x.shape) == 2
        return self.mlp_head(x)

@register_model
def maxvit_t(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> MaxViT:
    """Get MaxViT tiny model
    Refer to the base class "models.MaxViT" for more details.
    """
    default_cfg = default_cfgs["maxvit_t"]
    model = MaxViT( num_classes=num_classes,
                   num_blocks=(2,2,2,5,2), num_channels=(64,64,128,256,512), dim_conv_stem=64, dropout=0.2, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model

@register_model
def maxvit_s(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> MaxViT:
    """Get MaxViT tiny model
    Refer to the base class "models.MaxViT" for more details.
    """
    default_cfg = default_cfgs["maxvit_s"]
    model = MaxViT( num_classes=num_classes,
                   num_blocks=(2,2,2,5,2), num_channels=(64,96,192,384,768), dim_conv_stem=64, dropout=0.3, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model

@register_model
def maxvit_b(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> MaxViT:
    """Get MaxViT tiny model
    Refer to the base class "models.MaxViT" for more details.
    """
    default_cfg = default_cfgs["maxvit_b"]
    model = MaxViT( num_classes=num_classes,
                   num_blocks=(2,2,6,14,2), num_channels=(64,96,192,984,768), dim_conv_stem=64,  dropout=0.4,**kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model

@register_model
def maxvit_l(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> MaxViT:
    """Get MaxViT tiny model
    Refer to the base class "models.MaxViT" for more details.
    """
    default_cfg = default_cfgs["maxvit_l"]
    model = MaxViT( num_classes=num_classes,
                   num_blocks=(2,2,6,14,2), num_channels=(128,128,256,512,1024), dim_conv_stem=128,  dropout=0.6, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model

@register_model
def maxvit_xl(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> MaxViT:
    """Get MaxViT tiny model
    Refer to the base class "models.MaxViT" for more details.
    """
    default_cfg = default_cfgs["maxvit_xl"]
    model = MaxViT( num_classes=num_classes,
                   num_blocks=(2,2,6,14,2), num_channels=(192,192,384,768,1024), dim_conv_stem=192,dropout=0.6, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
