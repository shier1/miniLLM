import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 intermediate_size, 
                 mlp_bias=False, 
                 mlp_dropout=0.3,
                 **keywargs):
        
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)
        self.dropout = nn.Dropout(mlp_dropout)
        
    def forward(self, x):
        down_proj = self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
        return down_proj


def get_ffn(config):
    return eval(config.FFN.name)(**config.FFN)