import torch
from torch import nn

class pixel_shuffle_layer(nn.Module):
    """
    Class implementing invertible pixel shuffle layer.
    Based on paper https://arxiv.org/pdf/1609.05158.pdf
    """
    def __init__(self, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.scaling_factor_sq = scaling_factor*scaling_factor

    def forward(self, input):
        """
        Forward pass of this layer downsamples high resolution image to lower resolution image.
        By doing so extra channels are gained
        e.g. input image with size 1x64x64 will be downsampled 
        to size 1*scaling_factor_sq x 64/scaling_factor x 64/scaling_factor
        """

        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, s_height, s_width, s_depth) = output.size()

        desired_depth = s_depth * self.scaling_factor_sq
        desired_height = int(s_height / self.scaling_factor)
        
        t_1 = output.split(self.scaling_factor, 3)
        stack = [t_t.contiguous().view(batch_size,temp, desired_height, desired_depth) for t_t in t_1]
        output = torch.stack(stack, 2)
        output = output.permute(0, 4, 1, 3, 2)
        return output.contiguous()

    def inverse(self, input):
        """Inverse pass of this layer upsamples low resolution image to high resolution image"""
        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, desired_height, desired_width, desired_depth) = output.size()
        s_depth = int(desired_depth / 4)
        s_width = int(desired_width * 2)
        s_height = int(desired_height * 2)

        t_1 = output.contiguous().view(batch_size, temp, desired_height, desired_width, 4, s_depth)
        spl = t_1.split(2, 4)
        stack = [t_t.contiguous().view(batch_size, temp, desired_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).transpose(1, 2).permute(0, 1, 3, 2, 4, 5)
        output = output.contiguous().view(batch_size, temp, s_height, s_width, s_depth)
        output = output.permute(0, 4, 1, 2, 3)
        return output.contiguous()
