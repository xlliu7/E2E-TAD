import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple, _triple

import warnings

import torch



def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

    
def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def build_conv_layer(cfg, *args, **kwargs):
    return nn.Conv3d(*args, **kwargs)


def build_activation_layer(cfg):
    return nn.ReLU()


def build_norm_layer(cfg, num_features, postfix=''):
    return 'bn'+postfix, nn.BatchNorm3d(num_features)


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(typename='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # reset padding to 0 for conv module
        conv_padding = padding
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['typename'] not in ['Tanh', 'PReLU', 'Sigmoid']:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and \
                    self.act_cfg['typename'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x



class BasicBlock3d(nn.Module):
    """BasicBlock 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        inflate (bool): Whether to inflate kernel. Default: True.
        non_local (bool): Determine whether to apply non_local module in this
            block. Default: False.
        non_local_cfg (dict): Config for non_local module. Default: ``dict()``.
        conv_cfg (dict): Config dict for convolution layer.
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type``,
            Default: ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 inflate=True,
                 non_local=False,
                 non_local_cfg=dict(),
                 conv_cfg=dict(typename='Conv3d'),
                 norm_cfg=dict(typename='BN3d'),
                 act_cfg=dict(typename='ReLU'),
                 with_cp=False,
                 **kwargs):
        super().__init__()
        assert style in ['pytorch', 'caffe']
        # make sure that only ``inflate_style`` is passed into kwargs
        assert set(kwargs).issubset(['inflate_style'])

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        self.conv1_stride_s = spatial_stride
        self.conv2_stride_s = 1
        self.conv1_stride_t = temporal_stride
        self.conv2_stride_t = 1

        if self.inflate:
            conv1_kernel_size = (3, 3, 3)
            conv1_padding = (1, dilation, dilation)
            conv2_kernel_size = (3, 3, 3)
            conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 3, 3)
            conv1_padding = (0, dilation, dilation)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=conv1_padding,
            dilation=(1, dilation, dilation),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes * self.expansion,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s),
            padding=conv2_padding,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

        if self.non_local:
            self.non_local_module = build_non_local_module(
                self.non_local_cfg,
                default_args=dict(in_channels=self.conv2.norm.num_features))

    def forward(self, x):
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        if self.non_local:
            out = self.non_local_module(out)

        return out



class Bottleneck3d(nn.Module):
    """Bottleneck 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        inflate (bool): Whether to inflate kernel. Default: True.
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        non_local (bool): Determine whether to apply non_local module in this
            block. Default: False.
        non_local_cfg (dict): Config for non_local module. Default: ``dict()``.
        conv_cfg (dict): Config dict for convolution layer.
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type``,
            Default: ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 inflate=True,
                 inflate_style='3x1x1',
                 non_local=False,
                 non_local_cfg=dict(),
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        super().__init__()
        assert style in ['pytorch', 'caffe']
        assert inflate_style in ['3x1x1', '3x3x3']

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        if self.style == 'pytorch':
            self.conv1_stride_s = 1
            self.conv2_stride_s = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride_s = spatial_stride
            self.conv2_stride_s = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1

        if self.inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=conv1_padding,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s),
            padding=conv2_padding,
            dilation=(1, dilation, dilation),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            # No activation in the third ConvModule for bottleneck
            act_cfg=None)

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

        if self.non_local:
            self.non_local_module = build_non_local_module(
                self.non_local_cfg,
                default_args=dict(in_channels=self.conv3.norm.num_features))

    def forward(self, x):
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        if self.non_local:
            out = self.non_local_module(out)

        return out


class ResNet3d(nn.Module):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Channel num of input features. Default: 3.
        base_channels (int): Channel num of stem output features. Default: 64.
        out_indices (Sequence[int]): Indices of output feature. Default: (3, ).
        num_stages (int): Resnet stages. Default: 4.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
            Default: ``(1, 2, 2, 2)``.
        temporal_strides (Sequence[int]):
            Temporal strides of residual blocks of each stage.
            Default: ``(1, 1, 1, 1)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Default: ``(5, 7, 7)``.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Default: 2.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Default: 2.
        with_pool2 (bool): Whether to use pool2. Default: True.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default: -1.
        inflate (Sequence[int]): Inflate Dims of each block.
            Default: (1, 1, 1, 1).
        inflate_style (str): ``3x1x1`` or ``1x1x1``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        conv_cfg (dict): Config for conv layers. required keys are ``type``
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type`` and
            ``requires_grad``.
            Default: ``dict(type='BN3d', requires_grad=True)``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU', inplace=True)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        non_local (Sequence[int]): Determine whether to apply non_local module
            in the corresponding block of each stages. Default: (0, 0, 0, 0).
        non_local_cfg (dict): Config for non_local module. Default: ``dict()``.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
        kwargs (dict, optional): Key arguments for "make_res_layer".
    """

    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 base_channels=64,
                 out_indices=(3, ),
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 conv1_kernel=(5, 7, 7),
                 conv1_stride_t=2,
                 pool1_stride_t=2,
                 with_pool2=True,
                 style='pytorch',
                 frozen_stages=-1,
                 inflate=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 conv_cfg=dict(typename='Conv3d'),
                 norm_cfg=dict(typename='BN3d'),
                 act_cfg=dict(typename='ReLU'),
                 norm_eval=False,
                 with_cp=False,
                 non_local=(0, 0, 0, 0),
                 non_local_cfg=dict(),
                 zero_init_residual=True,
                 pretrained=None,
                 freeze_bn_affine=False,
                 **kwargs):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(
            dilations) == num_stages
        self.conv1_kernel = conv1_kernel
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_t = pool1_stride_t
        self.with_pool2 = with_pool2
        self.style = style
        self.frozen_stages = frozen_stages
        self.stage_inflations = _ntuple(num_stages)(inflate)
        self.non_local_stages = _ntuple(num_stages)(non_local)
        self.inflate_style = inflate_style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.freeze_bn_affine = freeze_bn_affine

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = self.base_channels

        self.non_local_cfg = non_local_cfg

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                non_local=self.non_local_stages[i],
                non_local_cfg=self.non_local_cfg,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                with_cp=with_cp,
                **kwargs)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * self.base_channels * 2**(
            len(self.stage_blocks) - 1)

    @staticmethod
    def make_res_layer(block,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       style='pytorch',
                       inflate=1,
                       inflate_style='3x1x1',
                       non_local=0,
                       non_local_cfg=dict(),
                       norm_cfg=None,
                       act_cfg=None,
                       conv_cfg=None,
                       with_cp=False,
                       **kwargs):
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature
                in each block.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides in
                residual and conv layers. Default: 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Default: 1.
            dilation (int): Spacing between kernel elements. Default: 1.
            style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``,
                the stride-two layer is the 3x3 conv layer, otherwise
                the stride-two layer is the first 1x1 conv layer.
                Default: ``pytorch``.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Default: 1.
            inflate_style (str): ``3x1x1`` or ``1x1x1``. which determines
                the kernel sizes and padding strides for conv1 and conv2
                in each block. Default: '3x1x1'.
            non_local (int | Sequence[int]): Determine whether to apply non_local
                module in the corresponding block of each stages.
                Default: 0.
            non_local_cfg (dict): Config for non_local module.
                Default: ``dict()``.
            conv_cfg (dict | None): Config for norm layers. Default: None.
            norm_cfg (dict | None): Config for norm layers. Default: None.
            act_cfg (dict | None): Config for activate layers. Default: None.
            with_cp (bool | None): Use checkpoint or not. Using checkpoint
                will save some memory while slowing down the training speed.
                Default: False.
        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks
        non_local = non_local if not isinstance(non_local,
                                            int) else (non_local, ) * blocks
        assert len(inflate) == blocks and len(non_local) == blocks
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                style=style,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp,
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    style=style,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                    **kwargs))

        return nn.Sequential(*layers)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=self.conv1_kernel,
            stride=(self.conv1_stride_t, 2, 2),
            padding=tuple([(k - 1) // 2 for k in _triple(self.conv1_kernel)]),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(self.pool1_stride_t, 2, 2),
            padding=(0, 1, 1))

        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        """Initiate the parameters in backbone."""
        for m in self.modules():
            if isinstance(m, _ConvNd):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3d):
                    constant_init(m.conv3.bn, 0)
                elif isinstance(m, BasicBlock3d):
                    constant_init(m.conv2.bn, 0)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                    if self.freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def load_pretrained_weight(self):
        from torchvision.models.utils import load_state_dict_from_url
        ckpt = load_state_dict_from_url('https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth')

        state_dict = ckpt['state_dict']
        # pdb.set_trace()
        state_dict = {k[len('backbone.'):]:v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)



def measure_time(model, x, repeat=20, warmup=10):
    for _ in range(warmup):
        y = model(x)
    
    s = time.time()
    for _ in range(repeat):
        y = model(x)
    diff = time.time() - s
    return diff/repeat


if __name__ == '__main__':
    from torchvision.models.utils import load_state_dict_from_url
    # import torch.utils
    import os
    import pdb
    import time
    from flop_count import flop_count

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    cfg = dict(
        depth=50,
        norm_eval=True,
        out_indices=(3, ),
        frozen_stages=1,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False)

    with_tpool = True
    if not with_tpool:
        cfg.update(dict(conv1_stride_t=1, pool1_stride_t=1, with_pool2=False))

    model = ResNet3d(
        **cfg
    )

    device = torch.device('cuda:0')
    bs, t, s = 1, 512, 96
    if not with_tpool:
        t = t // 8
    x = torch.rand([bs, 3, t, s, s]).cuda()
    model.to(device)

    if True:
        model.eval()
        model.requires_grad_(False)


    for _ in range(10):
        y = model(x)

    