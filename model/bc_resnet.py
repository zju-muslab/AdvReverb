import torch.nn as nn
from collections import OrderedDict


class SubSpecNorm(nn.Module):
    
    def __init__(self, channels, sub_bands, eps=1e-5):
        super().__init__()
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels*sub_bands, eps=eps)

    def forward(self, x):
        N, C, F, T = x.size()
        x = x.view(N, C * self.sub_bands, F // self.sub_bands, T)
        x = self.bn(x)
        return x.view(N, C, F, T)

        
class BCResBlock(nn.Module):
    
    def __init__(self, in_dim, out_dim, kernel, stride, padding, dilation, apply_subspec=True):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.input_layer = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.ReLU())
        self.f2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding, groups=out_dim, bias=False),
            SubSpecNorm(out_dim, 5) if apply_subspec else nn.BatchNorm2d(out_dim)
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel[::-1], padding='same', groups=out_dim, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
            nn.Dropout2d(0.1)
        )
        self.activation = nn.ReLU()
        
    def forward(self, x):
        if self.in_dim == self.out_dim:
            n_freq = x.shape[2]
            x1 = self.f2(x)
            x2 = x1.mean(dim=2, keepdim=True)
            x2 = self.f1(x2).repeat(1, 1, n_freq, 1)
            return self.activation(x + x1 + x2)
        else:
            x = self.input_layer(x)
            x = self.f2(x)
            n_freq = x.shape[2]
            x1 = x.mean(dim=2, keepdim=True)
            x1 = self.f1(x1).repeat(1, 1, n_freq, 1)
            return self.activation(x + x1)
        
     
class BCResNet(nn.Module):
    
    def __init__(self, network_config):
        super().__init__()
        self.config = network_config
        self.__scaling()
        cfg = self.config.input_layer
        self.input_layer = nn.Conv2d(cfg.in_dim, cfg.out_dim, cfg.kernel, cfg.stride, cfg.padding)
        cfg = self.config.back_layer
        self.back_layers = nn.ModuleList([self.__make_layer(i, cfg) for i in range(len(cfg.block_nums))])
        in_dim, out_dim, kernel = self.config.back_layer.hidden_dims[-1], self.config.neck_layer.out_dim, self.config.neck_layer.kernel
        self.neck_layer = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=kernel, groups=in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    def __scaling(self):
        self.config.input_layer.out_dim = int(self.config.input_layer.out_dim * self.config.scale)
        self.config.back_layer.hidden_dims = [int(h_dim * self.config.scale) for h_dim in self.config.back_layer.hidden_dims]
        self.config.neck_layer.out_dim = int(self.config.neck_layer.out_dim * self.config.scale)
        
    def __make_layer(self, layer, cfg):
        in_dim = self.config.input_layer.out_dim if layer == 0 else cfg.hidden_dims[layer-1]
        out_dim = cfg.hidden_dims[layer]
        block = nn.Sequential(OrderedDict([
            (f'back_layer_{layer}_bc_res_block_0', BCResBlock(in_dim, out_dim, cfg.kernel, cfg.strides[layer], cfg.padding, cfg.dilations[layer]))
            ]))
        for i in (1, cfg.block_nums[layer]):
            block.add_module(f'back_layer_{layer}_bc_res_block_{i}', BCResBlock(out_dim, out_dim, cfg.kernel, 1, cfg.padding, cfg.dilations[layer]))
        return block
    
    def forward(self, x):
        fbs = x.unsqueeze(1) # [B, 1, F, T]s
        fea = self.input_layer(fbs)
        for layer in self.back_layers:
            fea = layer(fea)
        fea = self.neck_layer(fea)
        emb = self.avg_pool(fea).view(fea.shape[0],-1)
        return emb