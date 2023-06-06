import torch
import torch.nn.functional as F
import numpy as np
import math

class ConvMax(torch.nn.Module):
    def __init__(self, input_channel=1, filters=4, kernel_size=3, pool_size=2, activation='relu', **kwargs):

        # Define Conv2d
        super(ConvMax, self).__init__()
        self.conv = torch.nn.Conv2d(input_channel, filters, kernel_size, padding=1)
        self.maxpool = torch.nn.MaxPool2d(pool_size, pool_size)
        self.relu = torch.nn.ReLU(inplace=False)

        # self.add_module('Conv', self.conv)
        # self.add_module('MaxPool', self.maxpool)
        # self.add_module('ReLU', self.relu)

    def call(self, input_tensor):
        x = self.model(input_tensor)
        return x
    
    def forward(self, x):
        x = F.relu(self.maxpool(self.conv(x)))
        return x

class RepeatedConvMax(torch.nn.Module):
    def __init__(self, repetitions=4, filters=4, kernel_size=3, pool_size=2, activation='relu', **kwargs):
        super(RepeatedConvMax, self).__init__(**kwargs)

        self.repetitions = repetitions
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.activation = activation

        # Define repeated ConvMax (Conv2d + MaxPool2d)
        for i in range(self.repetitions):
            # Define ConvMax
            setattr(self, f'convmax_{i}', ConvMax(self.filters, self.kernel_size, self.pool_size, self.activation))

    def call(self, input_tensor):
        # Connect first layer
        x = getattr(self, 'convMax_0')(input_tensor)
        
        # Connect remaining layers
        for i in range(1, self.repetitions):
            x = getattr(self, f'convMax_{i}')(x)

        # Return last layer
        return x
    
    def forward(self, x):
        # Connect first layer
        x = getattr(self, 'convMax_0')(x)
        
        # Connect remaining layers
        for i in range(1, self.repetitions):
            x = getattr(self, f'convMax_{i}')(x)

        # Return last layer
        return x

        

class ODCloneNetwork(torch.nn.Module):
    def __init__(self, texture_size=64, **kwargs):
        super(ODCloneNetwork, self).__init__(**kwargs)

        self.input_texture = torch.tensor(np.random.rand(texture_size, texture_size, 3), dtype=torch.float32)
        self.input_bg = torch.tensor(np.random.rand(64, 64, 3), dtype=torch.float32)
        self.input_fg = torch.tensor(np.random.rand(64, 64, 3), dtype=torch.float32)

        self.combined_input = [self.input_texture, self.input_bg, self.input_fg]

        # Define Texture Layers -- Texture Repeated ConvMax = log2(texture_size) - 2
        texture_repeated_conv_max = int(math.log(texture_size, 2)) - 2
        self.texture_layer = RepeatedConvMax(texture_repeated_conv_max)

        # Define Background Layers (image_size = 64x64, repeated ConvMax = 4)
        self.background_layer = RepeatedConvMax(4)

        # Define Foreground Layers (image_size = 64x64, repeated ConvMax = 4)
        self.foreground_layer = RepeatedConvMax(4)

        # Define last Conv2d
        self.last_conv2D_layer = torch.nn.Conv2d(12, 4, 3)

        # Define flatten
        self.flatten_layer = torch.nn.Flatten()

        # Define last output sigmoid
        self.final_output_layer = torch.nn.Sigmoid()

        # Reinitialization
        self.final_output = self.call(self.combined_input)
        super(ODCloneNetwork, self).__init__(inputs=self.combined_input, outputs=self.final_output, **kwargs)

    def call(self, input_tensors):
        input_texture, input_background, input_foreground = input_tensors

        output_texture = self.texture_layer(input_texture)
        output_background = self.background_layer(input_background)
        output_foreground = self.foreground_layer(input_foreground)

        concat_layer = torch.cat([output_texture, output_background, output_foreground])

        x = self.last_conv2D_layer(concat_layer)
        x = self.flatten_layer(x)
        x = self.final_output_layer(x)

        return x
    
od_clone_network = ODCloneNetwork()
od_clone_network.summary()