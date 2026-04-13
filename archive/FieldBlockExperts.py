import torch 
import torch.nn as nn 
from torch.nn import MultiheadAttention, LayerNorm, Linear, ReLU 
from typing import Dict, List 
import torch.nn.functional as F 

class FFN(nn.Module): 
    def __init__(self, dim_input: int, dim_output: int) -> None: 
        """
        Two linear layer refers to Transformer. 
        """
        super().__init__() 
        dim_hidden = 64 
        self.net = nn.Sequential(
            nn.Linear(dim_input, dim_hidden), 
            nn.ReLU(), 
            nn.Linear(dim_hidden, dim_output) 
        ) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.net(x) 

class SubfieldsBlock(nn.Module): 
    """
    A protocol is composed by several fields, like p = [f1, f2, ..., fn]. 
    For field fi, it can be dissected into subfields having specific meanings. 
    For subfields of a same field, they are combined by a Feedforward layer and turned into a assigned dimension. 
    Then, the result is added to field and normalized. 
    """
    def __init__(self, dim_subfields, dim_field) -> None:
        super().__init__() 
        self.ffn_subfields = FFN(dim_subfields, dim_field) 
        # self.ffn_field = FFN(dim_field, dim_hidden, dim_field) 
        self.norm = nn.LayerNorm(dim_field) 

    def forward(self, subfields, field): 
        sub_scores = self.ffn_subfields(subfields) 
        field_added = field + sub_scores 
        output = self.norm(field_added) 
        return output 

class SelfAttention(nn.Module): 
    def __init__(self, dim_input, dk) -> None:
        super().__init__() 
        self.dk = dk 
        self.query = nn.Linear(dim_input, dk) 
        self.key = nn.Linear(dim_input, dk) 
        self.value = nn.Linear(dim_input, dk) 

    def forward(self, x): 
        # x: (batch_size, num, dim_input) 
        q = self.query(x) 
        k = self.key(x) 
        v = self.value(x) 
        # q, k, v: (batch_size, num, dk) 
        scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float32)) 
        # scores: (batch_size, num, num) 
        weights = F.softmax(scores, dim=-1) 
        # weights: (batch_size, num, num) 
        output = torch.bmm(weights, v) 
        # output: (batch_size, num, dk) 
        return output 
    
class AddNorm(nn.Module): 
    def __init__(self, dim_input) -> None:
        super().__init__() 
        self.norm = nn.LayerNorm(dim_input) 

    def forward(self, before, after): 
        x = before + after 
        output = self.norm(x) 
        return output  
    
class ProtocolTreeAttention(nn.Module): 
    """
    Protocol Tree Attention Module for one protocol in a Field Block. 

    """
    def __init__(self, dims_subfields: List, dim_fields: int) -> None:
        super().__init__() 
        self.subfields_block = nn.ModuleList([
            SubfieldsBlock(dim_subfields, dim_fields) for dim_subfields in dim_fields 
        ]) 
    def forward(self, subfields: List[torch.Tensor], fields: List[torch.Tensor]):  
        pass 



class ResidualBlock(nn.Module): 
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__() 
        self.ffn = FFN(input_dim, hidden_dim, output_dim) 

    def forward(self, parent: torch.Tensor, subfields: List[torch.Tensor]): 
        if not subfields: 
            return parent 
        subfields_concat = torch.cat(subfields, dim=-1)
        subfields_out = self.ffn(subfields) 
        return parent + subfields_out 

class FieldsEmbeddingModule(nn.Module): 
    pass 

class TreeAttentionModule(nn.Module): 
    """
    Field Tree (FT) module for a specific Field Block (FB). 

    In a FB, the Tree Attention module subjects to the following rules. 
        1. Only childs of root (main fields) node for each protocol use Attention to find the correlation between fields. 
        2. Sub-fields are handled by FFN and will be added to their parent node by Residual Block. 

    Example 
    ------- 
    FT = {'eth': ['eth.dst', 'eth.src'], 'ip': ['ip.dsfield', 'ip.flags'], 
        'eth.dst': ['eth.dst.lg', 'eth.dst.ig'], 'eth.src': ['eth.src.lg', 'eth.src.ig'], 
        'ip.dsfield': ['ip.dsfield.dscp', 'ip.dsfield.ecn'], 'ip.flags': ['ip.flags.rb', 'ip.flags.df', 'ip.flags.mf']} 

    output = Attention(Attention('eth.dst', 'eth.src'), Attention('ip.dsfield', 'ip.flags')) 
        = Attention(Attention(Residual('eth.dst', FFN('eth.dst.lg', 'eth.dst.ig')), 
            Residual('eth.src', FFN('eth.src.lg', 'eth.src.ig'))), 
            Attention(Residual('ip.dsfield', FFN('ip.dsfield.dscp', 'ip.dsfield.ecn')), 
            Residual('ip.flags', FFN('ip.flags.rb', 'ip.flags.df', 'ip.flags.mf')))) 
    """
    def __init__(self, d_model, n_head, dim_forward) -> None:
        super().__init__() 
        self.attention = MultiheadAttention(d_model, n_head) 
        self.norm = LayerNorm(d_model) 
        self.ffn = nn.Sequential(
            # TODO
        ) 
        self.shortcut = nn.Sequential() 

    def forward(self, parent_fields, child_fields): 
        pass 

class MainExpertModule(nn.Module): 
    pass 

class ExtraExpertModule(nn.Module): 
    pass 

class FieldBlockExperts(nn.Module): 
    pass 