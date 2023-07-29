import math
import copy
from functools import partial
import mindspore

from mindspore import nn, Tensor, Parameter, ops
from mindspore.common.initializer import Normal, XavierUniform

from .helpers import load_pretrained
from .registry import register_model

__all__ = [
    "GatedMlp", 
    "gmlp_ti",
    "gmlp_s",
    "gmlp_b"
]

def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "features.0",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "gmlp_ti": _cfg(url=""),
    "gmlp_s": _cfg(url=""),
    "gmlp_b": _cfg(url="")
}

class Identity(nn.Cell):
    def __init__(self):
        super().__init__()
    def construct(self, x):
        return x

class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        seed = min(seed, 0) # always be 0
        self.shape = ops.Shape()
        self.ones = ops.Ones()
        print(self.keep_prob)
        self.dropout = nn.Dropout(self.keep_prob)

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x) # B N C
            mask = self.ones((x_shape[0], 1, 1), ms.float32)
            x = self.dropout(mask)*x
        return x

class GMlp(nn.Cell):
    """ GatedMLP module
    
    Impl using nn.Dense and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> gate -> fc -> dropout
    
    Attributes:
        fc1: nn.Dense
        fc2: nn.Dense
        act: GELU
        gate: gate layer
        dropout1: dropout after fc1
        dropout2: dropout after fc2
    """
    
    def __init__(self, in_features, hidden_features, gate_layer=None, dropout=0.):
        super().__init__()
        self.fc1 = nn.Dense(in_features, hidden_features, weight_init=XavierUniform(), bias_init=Normal(1e-6))
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2
        else:
            self.gate = Identity()
        self.fc2 = nn.Dense(hidden_features, in_features, weight_init=XavierUniform(), bias_init=Normal(1e-6))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(1.0 - dropout)
    
    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
        
class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding

    Args:
        image_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Cell, optional): Normalization layer. Default: None
    """

    def __init__(self, image_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,
                              pad_mode='pad', has_bias=True, weight_init="truncatedNormal")

        if norm_layer is not None:
            if isinstance(embed_dim, int):
                embed_dim = (embed_dim,)
            self.norm = norm_layer(embed_dim, epsilon=1e-5)
        else:
            self.norm = None

    def construct(self, x):
        """docstring"""
        B = x.shape[0]
        x = ops.Reshape()(self.proj(x), (B, self.embed_dim, -1))  # B Ph*Pw C
        x = ops.Transpose()(x, (0, 2, 1))

        if self.norm is not None:
            x = self.norm(x)
        return x

class SpatialGatingUnit(nn.Cell):
    def __init__(self, dim, seq_len):
        super().__init__()
        gate_dim = dim // 2
        self.norm = nn.LayerNorm((gate_dim,), epsilon=1e-6)
        self.proj = nn.Dense(seq_len,
                              seq_len,
                              weight_init='xavier_uniform',
                              bias_init=Normal(sigma=1e-6))


    def construct(self, x):
        u, v = x.chunk(2, axis=-1)
        v = self.norm(v)
        v = self.proj(v.transpose([0, 2, 1]))
        return u * v.transpose([0, 2, 1])

class SpatialGatingBlock(nn.Cell):
    def __init__(self, dim, seq_len, mlp_ratio=4, dropout=0., droppath=0.):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm = nn.LayerNorm((dim,), epsilon=1e-6)
        sgu = partial(SpatialGatingUnit, seq_len=seq_len)
        self.mlp_channels = GMlp(dim, channels_dim, gate_layer=sgu, dropout=dropout)
        self.drop_path = DropPath(droppath)

    def construct(self, x):
        h = x
        x = self.norm(x)
        x = self.mlp_channels(x)
        x = self.drop_path(x)
        x = x + h

        return x


class GatedMlp(nn.Cell):
    def __init__(self,
                 num_classes=1000,
                 image_size=224,
                 in_channels=3,
                 patch_size=16,
                 num_mixer_layers=30,
                 embed_dim=256,
                 mlp_ratio=6,
                 dropout=0.,
                 droppath=0.,
                 patch_embed_norm=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.embed_dim = embed_dim

        norm_layer=nn.LayerNorm((embed_dim,), epsilon=1e-6)
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_embed_norm else None)


        self.mixer_layers = nn.SequentialCell(*[SpatialGatingBlock(
            embed_dim,
            self.patch_embed.num_patches,
            mlp_ratio,
            dropout,
            droppath) for _ in range(num_mixer_layers)])

        self.norm = nn.LayerNorm((embed_dim,), epsilon=1e-6)
        self.head = nn.Dense(embed_dim, self.num_classes)

    def construct(self, x):
        x = self.patch_embed(x)
        x = self.mixer_layers(x)
        x = self.norm(x)
        x = x.mean(axis=1)
        return x

def build_gated_mlp(args):
    model = GatedMlp(num_classes=args.class_num,
                     image_size=args.train_image_size,
                     in_channels=args.in_channels,
                     num_mixer_layers=args.num_mixer_layers,
                     embed_dim=args.hidden_size,
                     mlp_ratio=args.mlp_ratio,
                     dropout=args.dropout,
                     droppath=args.drop_path)
    return model

@register_model
def gmlp_ti(pretrained: bool = False,
            num_classes: int = 1000,
            in_channels: int = 3,
            image_size: int = 224,
            **kwargs):
    default_cfg = default_cfgs["gmlp_ti"]
    model = GatedMlp(num_classes = num_classes, image_size=image_size, in_channels=in_channels, embed_dim=128)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model

@register_model
def gmlp_s(pretrained: bool = False,
            num_classes: int = 1000,
            in_channels: int = 3,
            image_size: int = 224,
            **kwargs):
    default_cfg = default_cfgs["gmlp_ti"]
    model = GatedMlp(num_classes = num_classes, image_size=image_size, in_channels=in_channels, embed_dim=256,droppath=0.05)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model

@register_model
def gmlp_b(pretrained: bool = False,
            num_classes: int = 1000,
            in_channels: int = 3,
            image_size: int = 224,
            **kwargs):
    default_cfg = default_cfgs["gmlp_ti"]
    model = GatedMlp(num_classes = num_classes, image_size=image_size, in_channels=in_channels, embed_dim=512, droppath=0.2)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model