import torch

class OrdinaryUNet():
    def __init__(self,config):
        def UpConv2D(tensor, filters, name, concat_with):
            up_i = torch.nn.Upsample((2, 2),mode='bilinear')(tensor)
            up_i = torch.cat([up_i, encoder.get_layer(concat_with).output]) # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i