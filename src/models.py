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
    hidden_channel_scaling_factor [int]
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
                 hidden_channel_scaling_factor=2,
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
        
        out = output_channels // hidden_channel_scaling_factor
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
    init_downscale_factor [int]
        TBD
    input_shape [list]
        shape of the input data
    droput_rate [float]
        float specifying the dropout rate
    learnable_batch_norm [bool]
        flag passed to nn.BatchNorm3d class, determining if the batch norm has learnable params
    hidden_channel_scaling_factor [int]
        number by witch the output channel will be divided
    """
    def __init__(self,
                 num_blocks: list,
                 num_strides: list, 
                 input_shape: list, 
                 num_channels: list=None, 
                 init_downscale_factor: int=2,
                 dropout_rate: float=0.0, 
                 learnable_batch_norm: bool=True, 
                 hidden_channel_scaling_factor: int=2
                 ):
        """Initialize AutoEncoder class"""
        super().__init__()

        if len(input_shape) < 3:
            raise ValueError("Input data should have at least 3 dimensions")
        
        # Set class parameters
        self.final_downscale_factor = input_shape[2] // 2**(num_strides.count(2)+init_downscale_factor//2)
        self.init_downscale_factor = init_downscale_factor
        self.input_channels = input_shape[0] * 2**self.init_downscale_factor
        self.num_blocks = num_blocks
        self.num_strides = num_strides
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.learnable_batch_norm = bool(learnable_batch_norm)
        self.hidden_channel_scaling_factor = hidden_channel_scaling_factor

        if not self.num_channels:
            self.num_channels = [self.input_channels//2, self.input_channels//2 * 4,
                         self.input_channels//2 * 4**2, self.input_channels//2 * 4**3]

        if self.init_downscale_factor == 0:
            raise ValueError("init_downscale_factor value can't be equal to 0")
        self.init_pixel_shuffle = pixel_shuffle_layer(self.init_downscale_factor)
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
                    hidden_channel_scaling_factor=self.hidden_channel_scaling_factor
                )
            )
            input_channels = 2 * channel
            first_block = False
        
        return block_list

    def forward(self, input, encode=True):

        if encode:
            n = self.input_channels // 2

            if self.init_downscale_factor == 0: # this check can break the code
                print("init_downscale_factor value == 0 but it shouldn't be")
            x = self.init_pixel_shuffle.forward(input)
            # Split the data in two parts for the two paths of the AE
            out = (x[:, :n, :, :, :], x[:, n:, :, :, :])

            for block in self.stack:
                out = block.forward(out)
            x = out
        else:
            out = input
            for i in range(len(self.stack)):
                out = self.stack[-1 - i].inverse(out)
            out = torch.cat((out[0], out[1]), 1)
            x = self.init_pixel_shuffle.inverse(out)
        return x


class ConvLSTMCell(nn.Module):
    """This is an implementation of ConvLSTM from this paper https://arxiv.org/abs/1506.04214
    In this implementation 2d convolutions were switched for 3d convolutions to improve feature
    extraction along temporal dimension. Also this implementation uses bigger architecture with 
    doubled gates. 

    Parameters
    ----------
    input_size [int]
    hidden_size [int]
    memory_size [int]
    """

    def __init__(self,
                 input_size, 
                 hidden_size, 
                 memory_size):
        super().__init__()
        
        self.KERNEL_SIZE = 3 # universal 3x3 kernel used 
        self.PADDING = self.KERNEL_SIZE // 2 # padding selected so that output shape remains the same

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        gates_params = [input_size + hidden_size + hidden_size , hidden_size, self.KERNEL_SIZE]
        self.in_gate = nn.Conv3d(*gates_params, padding=self.PADDING)
        self.remember_gate = nn.Conv3d(*gates_params, padding=self.PADDING)
        self.cell_gate = nn.Conv3d(*gates_params, padding=self.PADDING)

        gates_params = [input_size + memory_size + hidden_size , memory_size, self.KERNEL_SIZE]
        self.in_gate1 = nn.Conv3d(*gates_params, padding=self.PADDING)
        self.remember_gate1 = nn.Conv3d(*gates_params, padding=self.PADDING)
        self.cell_gate1 = nn.Conv3d(*gates_params, padding=self.PADDING)

        self.w1 = nn.Conv3d(hidden_size + memory_size, hidden_size, 1)
        self.out_gate = nn.Conv3d(
            input_size + hidden_size + hidden_size + memory_size, hidden_size, self.KERNEL_SIZE, padding=self.PADDING
            )

    def forward(self, input, prev_state):
        """Forward pass of the ConvLSTM
        
        Parameters
        ----------
        input [tuple]
            tuple which first element is the actual input and the second element is memory output from the previous cell
        prev_state [torch.tensor]
            tensor with state output from the previous cell
        """
        input_, prev_memo = input
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.zeros(state_size, requires_grad=True).cuda(),
                torch.zeros(state_size, requires_grad=True).cuda()
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden, prev_cell), 1)

        in_gate = torch.sigmoid(self.in_gate(stacked_inputs))
        remember_gate = torch.sigmoid(self.remember_gate(stacked_inputs))
        cell_gate = torch.tanh(self.cell_gate(stacked_inputs))

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)

        stacked_inputs1 = torch.cat((input_, prev_memo, cell), 1)

        in_gate1 = torch.sigmoid(self.in_gate1(stacked_inputs1))
        remember_gate1 = torch.sigmoid(self.remember_gate1(stacked_inputs1))
        cell_gate1 = torch.tanh(self.cell_gate1(stacked_inputs1))

        memo = (remember_gate1 * prev_memo) + (in_gate1 * cell_gate1)

        out_gate = torch.sigmoid(self.out_gate(torch.cat((input_, prev_hidden, cell, memo), 1)))
        hidden = out_gate * torch.tanh(self.w1(torch.cat((cell, memo), 1)))

        return (hidden, cell), memo


class RecurrentReversiblePredictor(nn.Module):
    """
    Class implementing recurrent reversible predictor module. 
    It consists of ST-LSTM and ConvLSTM layers arranged in similar manner to the AutoEncoder module
    
    Parameters
    ----------
    input_size [int]
        input data size
    hidden_size[int]
        size of the hidden state tensor
    output_size [int]
        output data size
    num_layers [int]
        number of layers of the module
    batch_size [int]
        batch size used during training
    temporal [int]
        temporal dimension of the input data, every input contains some stacked frames
    width [int]
        width of the input data
    height [int]
        height of the input data
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 output_size: int,
                 num_layers: int,
                 batch_size: int,
                 temporal: int=3, 
                 width: int=8,
                 height: int=8):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.temporal = temporal
        self.width = width
        self.height = height

        self.convlstm = nn.ModuleList(
            [  
                ConvLSTMCell(input_size, hidden_size, hidden_size) if i == 0 
                else ConvLSTMCell(hidden_size, hidden_size, hidden_size) 
                for i in range(self.num_layers)
            ]
        )

        self.attention = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(self.hidden_size, self.hidden_size, 1, 1, 0),
                    # nn.ReLU(),
                    # nn.Conv3d(self.hidden_size, self.hidden_size, 3, 1, 1),
                    nn.Sigmoid()
                ) 
                for i in range(self.num_layers)
            ]
        )

        self.init_hidden()
        self.prev_hidden = self.hidden

    def init_hidden(self):
        hidden = []

        for i in range(self.num_layers):
            hidden.append((torch.zeros(self.batch_size, self.hidden_size, self.temporal, self.width, self.height, requires_grad=True).cuda(),
                           torch.zeros(self.batch_size, self.hidden_size, self.temporal, self.width, self.height, requires_grad=True).cuda()))
        self.hidden = hidden

    def get_empty_memory(self):
        """Return empty tensor for hidden memory"""
        return torch.zeros((self.batch_size, self.hidden_size, self.temporal, self.width, self.height), requires_grad=True).to(self.device)

    def forward(self, input):
        input_, memo = input
        x1, x2 = input_
            # self.copy(self.hidden)
        for i in range(self.num_layers):
            out = self.convlstm[i]((x1,memo), self.hidden[i])
            self.hidden[i] = out[0]
            memo = out[1]
            g = self.attention[i](self.hidden[i][0])
            # Weighted sum
            x2 = (1 - g) * x2 + g * self.hidden[i][0]
            x1, x2 = x2, x1

        return (x1,x2),memo
