
import torch
from torch import nn
# import torchmeta
# from torchmeta.modules import MetaModule
# from torchmeta.modules import MetaSequential
# from torchmeta.modules import MetaLinear

from torch.nn import Softplus

from utils import get_init_fun
from utils import create_sequential_metalinear_layer, create_sequential_linear_layer

class SurfaceMapModel(nn.Module):

    def __init__(self):
        super().__init__()

        input_size  = 2
        output_size = 3
        act_fun     = Softplus

        modules = []

        # first layer input -> hiddent
        modules.append(nn.Linear(input_size, 256))
        modules.append(act_fun())

        # seq of residual blocks
        for layer in [256]*10:
            block = ResBlock(layer, act_fun)
            modules.append(block)

        # output layer
        modules.append(nn.Linear(256, output_size))

        self.mlp = nn.Sequential(*modules)

        ## initialize weights
        init_fun = get_init_fun()
        self.mlp.apply(init_fun)


    def forward(self, x, params=None):
        
        if params is not None:
            self.load_state_dict(params)

        # x = self.mlp(x, params=self.get_subdict(params, 'mlp'))
        x = self.mlp(x)
        return x



class ResBlock(nn.Module):

    def __init__(self, in_features, act_fun):
        super().__init__()

        # layer = create_sequential_metalinear_layer([in_features, in_features, in_features], act_fun)
        layer = create_sequential_linear_layer([in_features, in_features, in_features], act_fun)

        self.residual = nn.Sequential(*layer[:-1])
        self.post_act = layer[-1]


    def forward(self, x, params=None):
        if params is not None:
            sub_params = self.get_subdict(params, 'residual')
            for n, p in self.mlp.named_parameters():
                p.data = sub_params[n]
            print("never been here before")
        
        out = self.residual(x)
        out = self.post_act(out + x)
        return out
