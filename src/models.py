from .models_utils import pixel_shuffle_layer

import torch
from torch import nn


class InvertibleAEBlock(nn.Module):
    """
    Class for invertible block with convolutions and non-linear activations.
    The autoencoder consist of many blocks of this type
    
    Parameters
    ----------
    input_channels [int]
        number of input channels
    output_channels [int]
        number of input channels
    first_block [bool]
        flag determining whether initializing the first block of AE
    stride [int]
        stride parameter for neural network layers
    learnable_batch_norm [bool]
        flag passed to nn.BatchNorm3d class, determining if the batch norm has learnable params
    layer_scaling_factor [int]
        number by witch the output channel will be divided
    droput_rate [float]
        float specifying the dropout rate
    """
    def __init__(self,
                 input_channels,
                 output_channels,
                 first_block=False,
                 stride=1,
                 learnable_batch_norm=True,
                 layer_scaling_factor=2,
                 dropout_rate=0.0):
        """
        Initialize invertible block
        """
        super().__init__()
        self.stride = stride
        self.pixel_shuffle_layer = pixel_shuffle_layer(stride)
        layers = []

        if not first_block:
            layers.append(nn.BatchNorm3d(input_channels//2, affine=learnable_batch_norm))
            layers.append(nn.ReLU(inplace=True))
        
        out = output_channels // layer_scaling_factor
        actual_output_channels = out if out > 0 else 1
    
        if self.stride == 2:
            layers.append(nn.Conv3d(input_channels // 2, actual_output_channels, kernel_size=3,
                                    stride=(1,2,2), padding=1, bias=False))
        else:
            layers.append(nn.Conv3d(input_channels // 2, actual_output_channels, kernel_size=3,
                                    stride=self.stride, padding=1, bias=False))

        layers.append(nn.BatchNorm3d(actual_output_channels, affine=learnable_batch_norm))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(actual_output_channels, actual_output_channels,
                      kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.BatchNorm3d(actual_output_channels, affine=learnable_batch_norm))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(actual_output_channels, output_channels, kernel_size=3,
                      padding=1, bias=False))

        self.neural_block = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of InvertibleAEBlock"""
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.neural_block(x2)

        if self.stride == 2:
            x1 = self.pixel_shuffle_layer.forward(x1)
            x2 = self.pixel_shuffle_layer.forward(x2)

        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """Inverse pass of InvertibleAEBlock"""
        x2, y1 = x[0], x[1]

        if self.stride == 2:
            x2 = self.pixel_shuffle_layer.inverse(x2)

        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1

        if self.stride == 2:
            x1 = self.pixel_shuffle_layer.inverse(x1)

        return (x1, x2)


class AutoEncoder(nn.Module):
    """
    Class implementing the autoencoder. It joins together multiple InvertibleAEBlocks
        
    Parameters
    ----------
    num_blocks [list]
        list containing numbers of blocks with specific output size,
        members of this list correspond to members of num_strides and num_channels lists,
        e.g. first element of num_blocks is 4, let first element of num_strides be x and 
        first element of num_channels be y. In the autoencoder there will be 4 invertible blocks
        with first having stride x and rest 1. All these blocks will have y output channels
    num_strides [list]
        list containing strides for first blocks from groups of block with same output channel size
    num_channels [list]
        list containing number of output channels for every group of blocks, if not given will be assigned
        with values calculated from input channel number
    init_ds [int]
        TBD
    input_shape [list]
        shape of the input data
    droput_rate [float]
        float specifying the dropout rate
    learnable_batch_norm [bool]
        flag passed to nn.BatchNorm3d class, determining if the batch norm has learnable params
    layer_scaling_factor [int]
        number by witch the output channel will be divided
    """
    def __init__(self,
                 num_blocks: list,
                 num_strides: list, 
                 input_shape: list, 
                 num_channels: list=None, 
                 init_ds: int=2,
                 dropout_rate: float=0.0, 
                 learnable_batch_norm: bool=True, 
                 layer_scaling_factor: int=2
                 ):
        """Initialize AutoEncoder class"""
        super().__init__()

        if len(input_shape) < 3:
            raise ValueError("Input data should have at least 3 dimensions")
        
        self.ds = input_shape[2] // 2**(num_strides.count(2)+init_ds//2)
        self.init_ds = init_ds
        self.input_channels = input_shape[0] * 2**self.init_ds
        # Set class parameters
        self.num_blocks = num_blocks
        self.num_strides = num_strides
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.learnable_batch_norm = learnable_batch_norm
        self.layer_scaling_factor = layer_scaling_factor

        if not self.num_channels:
            self.num_channels = [self.input_channels//2, self.input_channels//2 * 4,
                         self.input_channels//2 * 4**2, self.input_channels//2 * 4**3]

        if self.init_ds == 0:
            raise ValueError("Init_ds value can't be equal to 0")
        self.init_psi = pixel_shuffle_layer(self.init_ds)
        self.stack = self.stack_invertible_blocks(InvertibleAEBlock)

    def stack_invertible_blocks(self, block_type):
        """Stack invertible blocks to create the autoencoder topology
        
        Parameters
        ----------
        block_type [callable]
            class for blocks to be stacked

        Returns
        -------
        block_list [nn.ModuleList]
            list of modules populated with blocks of block_type
        """

        input_channels = self.input_channels
        block_list = nn.ModuleList()
        strides = []
        channels = []
        first_block = True

        # Create lists with parameters for specific blocks
        for channel, depth, stride in zip(self.num_channels, self.num_blocks, self.num_strides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)
        # Create blocks and append them to block_list
        for channel, stride in zip(channels, strides):
            block_list.append(
                block_type(
                    input_channels=input_channels, 
                    output_channels=channel, 
                    first_block=first_block,
                    stride=stride,
                    dropout_rate=self.dropout_rate,
                    learnable_batch_norm=self.learnable_batch_norm,
                    layer_scaling_factor=self.layer_scaling_factor
                )
            )
            input_channels = 2 * channel
            first_block = False
        
        return block_list

    def forward(self, input, encode=True):

        if encode:
            n = self.input_channels // 2

            if self.init_ds == 0: # this check can break the code
                print("Init_ds value == 0 but it shouldn't be")
            x = self.init_psi.forward(input)
            out = (x[:, :n, :, :, :], x[:, n:, :, :, :])

            for block in self.stack:
                out = block.forward(out)
            x = out
        else:
            out = input
            for i in range(len(self.stack)):
                out = self.stack[-1 - i].inverse(out)
            out = torch.cat((out[0], out[1]), 1)
            x = self.init_psi.inverse(out)
        return x