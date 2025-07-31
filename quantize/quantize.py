import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
import math
from utils import get_rot, exchange_row_col, get_hadamard
from quantize.const import CLIPMAX, CLIPMIN
import random



def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x



class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        swc=None,
        lac=None,
        act_group_size=None,
        quant_method=None,
        block_size=128,
        rotate=True,
        max_rotation_step=1024,
        permutation_times=1,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        self.rotate = rotate
        self.max_rotation_step = max_rotation_step
        self.quant_method = quant_method
        self.Bsize=block_size

        init_value = 4.             # init value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
        
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size
        self.is_weight = shape != None
        self.permutation_times = permutation_times
        self.recorded_x_max = None
        self.let_s = None
        self.act_group_size = act_group_size
        self.lac = lac
        self.swc = swc

        self.init_duquant_params = torch.tensor(1)

        if block_size == -1:
            self.block_size = 4096
        else:
            self.block_size = block_size

        if self.rotate is None:
            self.H = get_hadamard(self.block_size)
        elif self.quant_method == 'ruquant':
            self.R, self.permutation_list = [], []
            self.omg=None
            if self.rotate is not False:
                self.init_duquant_params = torch.tensor(0)

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
            
        x_int = round_ste(x.float() / scale).half()    # avoid overflow
        
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax) 

        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant  
    
    def permutation_random(self, weight, other=None):
        hidden_dim = weight.shape[-1]
        _mean = {}
        _weight = weight.detach().clone().abs()
        for _ in range(hidden_dim):
            _mean[_] = torch.max(_weight[:, _]).item()
        _mean = sorted(_mean.items(), key=lambda x: x[1], reverse=True)
        top_k = weight.shape[1] // self.block_size
        top_k_channel = []
        paired_list = []


        l = list(set(range(weight.shape[1])))
        random.shuffle(l)
        top_k_channel = top_k_channel + l[len(l)//2 :]
        paired_list = paired_list + l[:len(l)//2]

        top_k_channel = torch.tensor(top_k_channel)
        paired_list = torch.tensor(paired_list)

        ans = []
        top_k_channel, paired_list = top_k_channel.tolist(), paired_list.tolist()
        for i in range(hidden_dim):
            if i in top_k_channel:
                ans.append(paired_list[top_k_channel.index(i)])
            else:
                ans.append(top_k_channel[paired_list.index(i)])
        weight = weight[:, ans]
        return weight, torch.tensor(ans)
 
    def permutation_zigzag(self, weight):
        weight = weight.detach().clone()
        hidden_dim = weight.shape[-1]
        weight_max = weight.abs().max(dim=0).values
        weight_mean = weight_max.mean().item()
        pairs = [(i, weight_max[i].item()) for i in range(hidden_dim)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        def zigzag(numbers):
            cur = 0
            up = True
            l = [[] for i in range(hidden_dim // self.block_size)]
            for i in range(len(numbers)):
                l[cur].append(numbers[i])
                if up:
                    cur += 1
                    if cur == len(l):
                        cur -= 1
                        up = False
                else:
                    cur -= 1
                    if cur == -1:
                        cur += 1
                        up = True
            return l
        pairs = zigzag(pairs)

        for i in range(len(pairs)):
            pairs[i].sort(key=lambda x: x[1], reverse=True)

        perm = torch.zeros(hidden_dim, dtype=torch.long)
        for i in range(len(pairs)):
            perm[i * self.block_size:(i+1) * self.block_size] = torch.tensor([_[0] for _ in pairs[i]])
        weight = weight[:, perm]
        return weight, perm

    def generate_permutation_matrix(self,rows):

        perm_indices = torch.randperm(rows)

        P = torch.zeros(rows, rows)
        P[torch.arange(rows), perm_indices] = 1
        return P
    def construct_givens_matrix_optimized(self,u, v):
       
        n = u.size(0)
        assert n % 2 == 0, 
        num_givens = n // 2
        

        u_chunks = u.view(-1, 2)
        v_chunks = v.view(-1, 2)
            

        norm_u_i = torch.norm(u_chunks, dim=1)
        norm_v_i = torch.norm(v_chunks, dim=1)

        dot_product = torch.sum(u_chunks * v_chunks, dim=1)
        cross_product = v_chunks[:, 0] * u_chunks[:, 1] - v_chunks[:, 1] * u_chunks[:, 0]  # 叉积确定sinθ符号
        cos_theta = dot_product / (norm_u_i * norm_v_i)
        sin_theta = cross_product / (norm_u_i * norm_v_i)

        G_i = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1)
        ], dim=-1)

        G_i = G_i.view(-1, 2, 2)

    
        G = torch.block_diag(*G_i)

        return G
    def get_R(self,X):
        R=torch.eye(self.Bsize,self.Bsize,device=X.device)
        for i in range(self.max_rotation_step):
        
            abs_X = torch.abs(X)
         
            max_val, max_col_idx = torch.max(abs_X, dim=0)
            
            abs_max_col_idx = torch.argmax(max_val)
        
            num_cols = X.size(1)  
            
            ori_x=X[:,abs_max_col_idx]
        
        
        
            random_tensor = torch.rand_like(ori_x)
            
            normalized_random_tensor = F.normalize(random_tensor, p=2, dim=0)
           
            ori_x_norm = torch.norm(ori_x)

            
            scaled_random_tensor = normalized_random_tensor * ori_x_norm
            scaled_random_tensor.to(device=ori_x.device)
            omg=ori_x-scaled_random_tensor
            H=torch.eye(omg.shape[0],device=X.device)-2/(omg.view(1,-1)@omg.view(-1,1))*(omg.view(-1,1)@omg.view(1,-1))
           
            X=H@X
            
            R=H@R

            for i in range(3):
                
                P=self.generate_permutation_matrix(X.shape[0])
                P=P.to(device=X.device)
                X=P@X
                
                
                
                R=P@R
                
                
                abs_X = torch.abs(X)
                
                max_val, max_col_idx = torch.max(abs_X, dim=0)
                
                abs_max_col_idx = torch.argmax(max_val)
                num_cols = X.size(1)  
            
                ori_x = X[:, abs_max_col_idx]
                
                random_tensor =  torch.rand_like(ori_x)
                
                
                
                ori_x_reshaped = ori_x.view(-1, 2)
                random_tensor_reshaped = random_tensor.view(-1, 2)

               
                normalized_random_tensor = F.normalize(random_tensor_reshaped, p=2, dim=1)

                
                ori_x_norms = torch.norm(ori_x_reshaped, dim=1)

              
                scaled_random_tensor = normalized_random_tensor * ori_x_norms.unsqueeze(1)

               
                scaled_random_tensor = scaled_random_tensor.view_as(ori_x)

                
                G=self.construct_givens_matrix_optimized(scaled_random_tensor, ori_x)
               
                X=G@X
            
                R=G@R
        return R
    
    def get_omg(self,X,times=1):
        omgs=[]
        for i in range(times):
            abs_X = torch.abs(X)
            
            max_val, max_col_idx = torch.max(abs_X, dim=0)
          
            abs_max_col_idx = torch.argmax(max_val)
            
            ori_x=X[:,abs_max_col_idx]
           
            
            random_tensor = torch.rand_like(ori_x)
            ori_x_norm = torch.norm(ori_x)
            random_tensor_norm = torch.norm(random_tensor)
            
            scaled_random_tensor = random_tensor * (ori_x_norm / random_tensor_norm)
            scaled_random_tensor.to(device=ori_x.device)
            omg=ori_x-scaled_random_tensor
            
            omgs.append(omg.view(-1, 1))
            
            V = F.normalize(omg, dim=0)
    
            X=X-2 * torch.matmul(V.view(-1,1), torch.matmul(V.view(1,-1), X))
        omgs_tensor = torch.cat(omgs, dim=1)  
        return omgs_tensor
    

    
    def rotation(self, weight, max_rotation_step=None, other=None, score_func=None):
        
        
        weight = weight.detach().clone()
        weight = weight.squeeze()
        shape=weight.shape
        weight = weight.reshape(-1,self.Bsize)
        
        R=self.get_R(weight.t()).t()
        
        weight = (weight@R).reshape(shape)

        
        return weight ,None,R

    
    def calculate_std(self, weight):
        weight = weight.abs().max(dim=0).values
        groups = [weight[j * self.block_size: (j+1) * self.block_size] for j in range(weight.shape[0] // self.block_size)]
        group_means = [sum(group) / len(group) for group in groups]
        mean = sum(group_means) / len(group_means)
        variance = sum((x - mean) ** 2 for x in group_means) / len(group_means)
        return math.sqrt(variance)

    def online_duquant_cali(self, weight):
        weight = weight.detach().clone()
        T = {}
        
        self.permutation_list = None
        self.R = None
        for i in range(self.permutation_times):
            weight, _, R = self.rotation(weight)
            if self.R is None:
                self.R = R.unsqueeze(0)
            else:
                self.R = torch.cat((self.R, R.unsqueeze(0)), dim=0)
            
            weight, perm = self.permutation_zigzag(weight)
            if self.permutation_list is None:
                self.permutation_list = perm.unsqueeze(0)
            else:
                self.permutation_list = torch.cat((self.permutation_list, perm.unsqueeze(0)), dim=0)

        weight, _, R = self.rotation(weight)
        if self.R is None:
            self.R = R.unsqueeze(0)
        else:
            self.R = torch.cat((self.R, R.unsqueeze(0)), dim=0)
        self.add_omg=True
        restored_X=weight.t()
        if self.add_omg:
            self.omg=self.get_omg(weight.t(),times=1)
            for i in range(self.omg.size(1)):  
                V = F.normalize(self.omg[:, i], dim=0)  
                restored_X=restored_X-2 * torch.matmul(V.view(-1,1), torch.matmul(V.view(1,-1), restored_X))
        
        return restored_X.t().unsqueeze(0)
       

    def init_duquant(self, x: torch.Tensor):
        self.dim_omg=x.shape[-1]
        if self.quant_method is None:
            return x
        if self.rotate is None:
            
            x_shape = x.shape   # (n_tokens, hidden_dim) / (out_features, in_features)
            hadamard = self.H.to(x)
            x = x.reshape(-1, self.block_size)
            x = x.matmul(hadamard).view(x_shape)
        elif self.quant_method == 'ruquant':  
            if self.rotate:
                if not self.init_duquant_params:
                    x = self.online_duquant_cali(x)
                    self.init_duquant_params = torch.tensor(1)
                else:
                    x_size = x.shape
                    x_type = x.dtype
                    if self.permutation_list is not None:
                        for i in range(len(self.permutation_list)):
                            x = x.reshape(-1, self.block_size)
                            R = self.R[i].to(x)
                            x = x.matmul(R).reshape(x_size).squeeze(0)
                            # if False:
                            if True:
                                if len(self.permutation_list.shape) == 3:
                                    perm = (self.permutation_list[i, 0].to(x.device), self.permutation_list[i, 1].to(x.device))
                                    x[:, perm[0]], x[:, perm[1]] = x[:, perm[1]], x[:, perm[0]]
                                else:
                                    perm = self.permutation_list[i].to(x.device)
                                    x = x[:, perm]
                    if len(self.R) > 0:
                        x = x.reshape(-1, self.block_size)
                        R = self.R[-1].to(x)
                        x = x.matmul(R).reshape(x_size) 
                        
                    if self.omg is not None:
                        
                        x=x.squeeze(0)   
                        V = F.normalize(self.omg, dim=0)
                        # x=x-2 * torch.matmul(V.view(-1,1), torch.matmul(V.view(1,-1), x))
                        x=x - 2 * torch.matmul(torch.matmul(x, V.view(-1,1)), V.view(1,-1))
                        # x=x.t()
                        x=x.reshape(x_size)
                    
        else:
            raise NotImplementedError
        return x
            

    def forward(self, x: torch.Tensor, return_no_quant=False):
        if hasattr(self, 'smooth_scales'):
            x /= self.smooth_scales.to(x.device)

        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
            
        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            x = self.init_duquant(x)

        if return_no_quant:
            reduce_shape = [-1]
            xmin = x.amin(reduce_shape, keepdim=True)
            xmax =  x.amax(reduce_shape, keepdim=True)
            if self.swc:
                xmax = self.swc*xmax
                xmin = self.swc*xmin
            elif self.lwc:
                xmax = self.sigmoid(self.upbound_factor)*xmax
                xmin = self.sigmoid(self.lowbound_factor)*xmin
            if self.lac:
                xmax = self.lac*xmax
                xmin = self.lac*xmin
            return x
        
        if self.recorded_x_max is None:
            self.recorded_x_max = x.abs().reshape(-1, x.shape[-1]).max(axis=0).values
        if self.let_s is not None:
            x /= self.let_s

        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        # return x
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1] #
        xmin = x.amin(reduce_shape, keepdim=True).to(x.device)#
        xmax =  x.amax(reduce_shape, keepdim=True).to(x.device)#
        if self.swc:
            xmax = self.swc*xmax
            xmin = self.swc*xmin
        elif self.lwc:
            xmax = self.sigmoid(self.upbound_factor.to(x.device))*xmax
            xmin = self.sigmoid(self.lowbound_factor.to(x.device))*xmin
        elif self.lac:#
            xmax = self.lac*xmax
            xmin = self.lac*xmin

        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
            zero_point = -(xmin) / (self.scale)
        self.round_zero_point = zero_point.clamp(min=-CLIPMAX, max=CLIPMAX).round()
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point

    def register_duquant_params(self):
        if self.rotate is not True:
            return
        
        omg,permutation_list, R = self.omg,self.permutation_list, self.R
        delattr(self, 'R')
        delattr(self, 'omg')
        delattr(self, 'permutation_list')
        delattr(self, 'init_duquant_params')
        self.register_buffer('omg', omg)
        self.register_buffer('permutation_list', permutation_list)
        self.register_buffer('R', R)
        self.register_buffer('init_duquant_params', torch.tensor(1))

    def copy_duquant_params(self, quantizer_ref):
        if quantizer_ref.rotate is True:
            assert quantizer_ref.init_duquant_params == True
            self.R = quantizer_ref.R.clone().detach()
            
            if quantizer_ref.omg is not None:
                self.omg = quantizer_ref.omg.clone().detach()
            try:
                self.permutation_list = quantizer_ref.permutation_list.clone().detach()
            except:
                self.permutation_list = quantizer_ref.permutation_list
            self.init_duquant_params = torch.tensor(1)
