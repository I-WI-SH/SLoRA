#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LoRALayer 
from typing import Optional, List 
from spikingjelly.activation_based import functional, encoding, neuron


# change the class name to SVDLinear when change SNN or ANN

class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
            
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            
            '''
            exp5: use LIFNode
            '''
            # self.snn_lora = neuron.LIFNode(step_mode='s',tau=2.0)
            self.snn = neuron.IFNode(step_mode='s')
            
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            
            self.lora_E = nn.Parameter(
                self.weight.new_zeros(r, 1)
            )

            self.fire_rate = nn.Parameter(
                self.weight.new_zeros(1), requires_grad= False
            )

            
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
            self.fire_rate.requires_grad = False
        
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            # and E (singular values) for zero 
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(
                    self.lora_B @ (self.lora_A*self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        
        if self.merge_weights and self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        import pdb
        # pdb.set_trace()
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:                
                T = 10
                o_1_frequency = 0
                encoder = encoding.PoissonEncoder()
                '''
                exp4: add SNN after LoRA_E;
                '''
                for t in range(T):    
                    encoded_input = encoder(self.lora_dropout(x))
                    o_1_frequency += self.snn(encoded_input @ (self.lora_A * self.lora_E).T)         
                o_1_frequency = o_1_frequency / T
                self.fire_rate.data.fill_(o_1_frequency.mean())
                result += o_1_frequency @ self.lora_B.T * self.scaling / (self.ranknum+1e-5)

                functional.reset_net(self)
            return result
        # else:
        #     return F.linear(x, T(self.weight), bias=self.bias)1


class SLoRARankAllocator(object):
    """
    The RankAllocator for AdaLoRA Model that will be called every training step. 
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model: the model that we apply AdaLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank. 
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter. 
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter. 
    """
    def __init__(
        self, model, 
        lora_r:int,
        target_rank:int, 
        init_warmup:int, 
        final_warmup:int,
        mask_interval:int,
        beta1:float, 
        beta2:float, 
        total_step:Optional[int]=None, 
        target_total_rank:Optional[int]=None,
        tb_writter=None,
        tb_writter_loginterval:int=500, 
    ):
        self.ave_target_rank = target_rank 
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r 
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup 
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {} 
        self.get_lora_param_name()

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval 

        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)


    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        assert self.total_step>self.initial_warmup+self.final_warmup


    def get_rank_pattern(self):
        # Return rank pattern 
        return self.rank_pattern


    def get_lora_param_name(self):
        # Prepare the budget scheduler 
        self.name_set = set() 
        self.total_rank = 0 
        self.shape_dict = {}
        for n,p in self.model.named_parameters():
            if "lora_A" in n: 
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0) 
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set)) 
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set) 


    def schedule_threshold(self, step:int):
        # Global budget schedule
        mask_ind = False 
        target_rank = self.target_rank 
        initial_warmup = self.initial_warmup 
        final_warmup = self.final_warmup 
        total_step = self.total_step 
        self.global_step = step
        
        if step <= initial_warmup: 
            # Initial warmup 
            curr_rank = self.total_rank 
            mask_ind = False 
        
        elif step > total_step - final_warmup: 
            # Final fine-tuning 
            curr_rank = self.target_rank 
            # Fix the rank pattern by 
            # always masking the same unimportant singluar values 
            mask_ind = True 
        
        else: 
            # Budget decreasing 
            mul_coeff = 1-(step-initial_warmup)/(total_step-final_warmup-initial_warmup)
            curr_rank = target_rank + (self.total_rank-target_rank)*(mul_coeff**3)
            curr_rank = int(curr_rank)
            mask_ind = True if step % self.mask_interval == 0 else False 
        
        return curr_rank, mask_ind 


    def update_ipt(self, model): 
        # n means a parameter matrix like: model.encoder.layers.0.self_attn.k_proj.lora_A torch.Size([12, 1024])
        for n,p in model.named_parameters():
            if "lora_" in n: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)

                with torch.no_grad():
                    # Calculate sensitivity 
                    # Use Parameter Importance Measurement Hear!
                    if p.requires_grad == True:
                        self.ipt[n] = (p * p.grad - 0.5 * (p * p.grad)**2).abs().detach()
                    else:
                        self.ipt[n] = torch.zeros_like(p)
                    # Update sensitivity 
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1-self.beta1)*self.ipt[n]
                    
                                        

    def calculate_score(self, n, p=None, metric="ipt"):
        
        if metric == "ipt":
            ipt_score = self.exp_avg_ipt[n]
        
        elif metric == "mag":
            ipt_score = p.abs().detach().clone() 
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score 


    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt


    def compute_model_ipt(self, model):
        """
        compute every type params' ipt score, cross different layers and different lora module(q,k,v,f1,f2,...)
        """
        
        ipt_E_dict = {}
        AB_dict = {}
        singular_dict = {}
        fire_rate_dict = {}

        for n,p in model.named_parameters(): 
            if "lora_A" in n: 
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("lora_A", "%s")
                if name_mat not in AB_dict: 
                    AB_dict[name_mat] = [comb_ipt]
                else:
                    AB_dict[name_mat].append(comb_ipt)
            if "lora_B" in n: 
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                if name_mat not in AB_dict: 
                    AB_dict[name_mat] = [comb_ipt]
                else:
                    AB_dict[name_mat].append(comb_ipt)
            
            # model.encoder.layers.0.self_attn.v_proj.lora_E torch.Size([12, 1])
            # use name_mat to present this matrix "model.encoder.layers.0.self_attn.v_proj"
            # because will use A & B to canculate the total importance
            if "lora_E" in n:
                ipt_score = self.calculate_score(n, p=p, metric="ipt")              
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score

            if "fire_rate" in n:
                name_mat = n.replace("fire_rate", "%s")
                fire_rate_dict[name_mat] = p.item()

        all_ipt = []
        layers_ipt = [0]*12
        layers_fire_rate = [[] for _ in range(12)]

        for name_mat in singular_dict:
            ipt_E = singular_dict[name_mat] 
            ipt_AB = torch.cat(AB_dict[name_mat], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_mat%"lora_E"
            ipt_E_dict[name_E] = sum_ipt.view(-1, 1)
            all_ipt.append(sum_ipt.view(-1))

            # layer_index = int(name_mat.split('.')[3])
            import re
            match = re.search(r'\.layer\.(\d+)\.', name_mat)
            layer_index = int(match.group(1)) if match else None

            layers_ipt[layer_index] += sum_ipt.sum().item()
            layers_fire_rate[layer_index].append(fire_rate_dict[name_mat])

        
        for i in range(len(layers_fire_rate)):
            sublist = layers_fire_rate[i]
            layers_fire_rate[i] = sum(sublist) / len(sublist) if sublist else 0.0
 
        return ipt_E_dict, all_ipt, layers_ipt, layers_fire_rate


    def mask_to_target_rank(self, model, curr_rank): 
        
        is_dict, all_is, layers_ipt, layers_fire_rate = self.compute_model_ipt(model)

        # Calculate the masking threshold 
        mask_threshold = torch.kthvalue(torch.cat(all_is), (self.total_rank-curr_rank))[0].item()

        # Mask out unimportant singular values 
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            traing_sum_param = 0
            for n,p in model.named_parameters():
                if "lora_E" in n: 
                    p.data.masked_fill_(is_dict[n]<=mask_threshold, 0.0)
                    ranknum = (is_dict[n]>mask_threshold).sum().item() 

                    if self.tb_writter is not None and self.global_step%self.log_interval==0:
                        self.tb_writter.add_scalar("Ranknum/%s"%(n,), ranknum, self.global_step) 
                        self.rank_pattern[n] = ranknum 
                        curr_sum_rank += ranknum 
                        sum_param += ranknum*self.shape_dict[n.replace("lora_E", "lora_A")][1]  
                        sum_param += ranknum*self.shape_dict[n.replace("lora_E", "lora_B")][0]  

                    

            if self.tb_writter is not None and self.global_step%self.log_interval==0:
                self.tb_writter.add_scalar("Budget/total_rank", curr_sum_rank, self.global_step)
                self.tb_writter.add_scalar("Budget/mask_threshold", mask_threshold, self.global_step)
                self.tb_writter.add_scalar("Budget/sum_param", sum_param, self.global_step)
                self.tb_writter.add_scalars("layers_fire_rate",
                                            {f'layer{i}': value for i,value in enumerate(layers_fire_rate)}, self.global_step)

                self.tb_writter.add_scalars("layers_ipt",
                                            {f'layer{i}': value for i,value in enumerate(layers_ipt)}, self.global_step)


        return mask_threshold, layers_ipt, layers_fire_rate


    def update_and_mask(self, model, global_step):
        if global_step<self.total_step-self.final_warmup:
            # Update importance scores element-wise 
            self.update_ipt(model)
            
            if self.tb_writter is not None and global_step%self.log_interval==0:
                _, _, layers_ipt, layers_fire_rate = self.compute_model_ipt(model)
                
                self.tb_writter.add_scalars("layers_fire_rate",
                                                {f'layer{i}': value for i,value in enumerate(layers_fire_rate)}, global_step)

                self.tb_writter.add_scalars("layers_ipt",
                                                {f'layer{i}': value for i,value in enumerate(layers_ipt)}, global_step)
        
        curr_rank, mask_ind = self.schedule_threshold(global_step)

        
        if mask_ind:
            # Mask to target budget 
            mask_threshold, layers_ipt_list, layers_fire_rate_list = self.mask_to_target_rank(model, curr_rank)

        else:
            mask_threshold = None 
            layers_ipt_list = None 
            layers_fire_rate_list = None
        
        self._maybe_tb_writter_log(model)
        

        return curr_rank, mask_threshold, layers_ipt_list, layers_fire_rate_list


    def _maybe_tb_writter_log(self, model):
        if self.tb_writter is not None and self.global_step%self.log_interval==0:
            with torch.no_grad():
                regu_loss = []
                for n,p in model.named_parameters():
                    if "lora_A" in n or "lora_B" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat 
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov-I, p="fro")
                        regu_loss.append(orth_regu.item())
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss/%s"%n, orth_regu.item(), self.global_step
                        )
                self.tb_writter.add_scalar(
                    "train/orth_regu_loss", sum(regu_loss)/len(regu_loss), self.global_step
                )


def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in `model`. 
    regu_loss, num_param = 0., 0
    for n,p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p 
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov-I, p="fro")
            num_param += 1
    return regu_weight*regu_loss/num_param

