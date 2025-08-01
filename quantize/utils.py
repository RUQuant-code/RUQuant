from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
import torch.nn as nn
from quantize.int_matmul import QuantMatMul
from quantize.quantizer import UniformAffineQuantizer
from models.transformation import *
import pickle
from quantize.const import CLIPMIN

def smooth_parameters(model, use_shift=True):
    params = []
    for n, m in model.named_parameters():
        if n.find('smooth') > -1:
            params.append(m)
    return iter(params)

def let_parameters(model, use_shift=True):
    params = []
    # template = "smooth" if use_shift else "smooth_scale"
    template = "post_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)  

def lh_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('opt_omg') > -1:
            params.append(m)
    return iter(params) 


def get_duquant_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params)  

def get_post_parameters(model):
    params = []
    template = "post_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1 or n.find('omg') > -1:
            params.append(m)
    return iter(params)

def set_requires_grad(it, requires_grad):
    for param in it:
        param.requires_grad = requires_grad

def duquant_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1 or name.find('trans') > -1 or name.find('post')>-1:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, param in model.named_buffers():
        if name.find('omg') > -1 or name.find('R') > -1 or name.find('sort_index') > -1 or name.find('init_duquant_params') > -1 or name.find('permutation_list') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()      
    # for name, param in model.named_buffers():
    #     if name.find('init_duquant_params') > -1 or name.find('R') > -1 or name.find('permutation_list') > -1:
    #         destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

def post_rotate_quant_temporary(model, args):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "post_scale" in name:
                    module.data = truncate_number(module)
        post_fcs_temporary([model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj], model.qkv_post_scale)
        post_fcs_temporary([model.mlp.up_proj,model.mlp.gate_proj], model.fc1_post_scale)
        post_fcs_temporary(model.mlp.down_proj, model.down_post_scale)
        post_fcs_temporary(model.self_attn.o_proj, model.out_post_scale)



    
def post_omg_temporary(fcs, omg):
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        fc.act_quantizer.omg = omg
        fc.weight_quantizer.omg = omg
        
def post_omg_temporary_down(fc, omg):
    fc.act_quantizer.omg= omg
    fc.weight_quantizer.omg= omg
        
def post_opt_omg_temporary(model,args):
    if args.lh:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "opt_omg" in name:
                    module.data = truncate_number(module)
        
        post_omg_temporary([model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj], model.qkv_opt_omg)
        post_omg_temporary([model.mlp.up_proj,model.mlp.gate_proj], model.fc1_opt_omg)
        post_omg_temporary(model.mlp.down_proj, model.down_opt_omg)
        post_omg_temporary(model.self_attn.o_proj, model.out_opt_omg)





@torch.no_grad()   
def post_quant_inplace(model, args):
    if args.let:
        for name, module in model.named_parameters():
            if "post_scale" in name:
                module.data = truncate_number(module)
            if isinstance(module, QuantLinear):
                if module.act_quantizer.let_s is not None:
                    module.act_quantizer.let_s.requires_grad = False
                if module.weight_quantizer.let_s is not None:
                    module.weight_quantizer.let_s.requires_grad = False
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

def set_registered_x_none(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.registered_x = None
            module.act_quantizer.registered_x = None


@torch.no_grad()
def set_init_duquant_params_state(model, mode):
    if isinstance(mode, bool):
        mode = torch.tensor(mode)
    for name, module in model.named_modules():
        if hasattr(module, "init_duquant_params"):
            module.init_duquant_params = mode

def smooth_and_quant_temporary(model, args, isllama):
    if args.smooth:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.mlp.up_proj,model.mlp.down_proj,
                            model.down_smooth_scale, model.down_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        else:
            smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_ln_fcs_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.fc2.temp_weight = model.fc2.weight
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight
    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                module.temp_weight = module.weight_quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True
        

@torch.no_grad()   
def smooth_and_let_inplace(model, args):
    if args.smooth:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj], model.qkv_smooth_scale,model.qkv_smooth_shift)
        smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                model.fc1_smooth_scale,model.fc1_smooth_shift)
        smooth_fc_fc_inplace(model.mlp.up_proj,model.mlp.down_proj,
                            model.down_smooth_scale, model.down_smooth_shift)
        try:
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
        except:
            smooth_fc_inplace(model.self_attn.o_proj, model.out_smooth_scale)
            print('Detected GQA in o_proj')
    
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.use_temporary_parameter=False
        
@torch.no_grad()
def quant_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight, return_no_quant=False)

@torch.no_grad()
def quant_ght_inplace(model):
  
    R = model.self_attn.k_proj.weight_quantizer.R

    
    R = torch.diag(torch.diag(R))
    
    model.self_attn.k_proj.weight_quantizer.R=None
    model.self_attn.v_proj.weight_quantizer.R=None
    model.self_attn.q_proj.weight_quantizer.R=None
    
   
            
@torch.no_grad()   
def smooth_and_let_inplace(model, args):
    if args.smooth:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj], model.qkv_smooth_scale,model.qkv_smooth_shift)
        smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                model.fc1_smooth_scale,model.fc1_smooth_shift)
        smooth_fc_fc_inplace(model.mlp.up_proj,model.mlp.down_proj,
                            model.down_smooth_scale, model.down_smooth_shift)
        try:
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
        except:
            smooth_fc_inplace(model.self_attn.o_proj, model.out_smooth_scale)
            print('Detected GQA in o_proj')
    
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.use_temporary_parameter=False
#################################################################################################

@torch.no_grad()
def quant_soft_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight, return_no_quant=True)

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
