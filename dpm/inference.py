import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, List, Optional
from dpm.modules import UNet, timestep_embedding


class UnetWrapper(nn.Module):
    def __init__(self, unet_model: UNet, cache_interval=0, init_fwd_step=0):
        super().__init__()
        self.unet = unet_model
        self.cache_interval = cache_interval
        self.fwd_step = init_fwd_step

    def _forward_full(self, x: Tensor, timesteps: Tensor):
        hs = []
        
        # down stage
        h = x
        t = self.unet.time_embed(timestep_embedding(timesteps, self.unet.model_channels))
        for module in self.unet.down_blocks:
            h = module(h, t)
            hs.append(h)
            
        # middle stage
        h = self.unet.middle_block(h, t)
        
        # up stage
        for module in self.unet.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, t)
        
        return self.unet.out(h)
    
    def _forward_with_cache(self, x: Tensor, timesteps: Tensor):
        raise NotImplementedError
    
    
    def forward(self, x: Tensor, timesteps: Tensor):
        
        if self.fwd_step % self.cache_interval == 0:
            res = self._forward_full(x, timesteps)
        else:
            res = self._forward_with_cache(x, timesteps)
            
        self.fwd_step += 1
        return res


class DeepCacheWrapper(UnetWrapper):
    def __init__(
        self, 
        unet_model: UNet, 
        cache_interval:int=0, 
        init_fwd_step:int=0, 
        cache_layer_list:Optional[List]=None
    ):
        
        super().__init__(unet_model, cache_interval, init_fwd_step)
        self.cache_layer_list = cache_layer_list
        self.cache_features: Dict[int, Tensor] = {}
        self._init_cache_ids()
        
    def _init_cache_ids(self):
        if self.cache_layer_list:
            # copy form init parameter
            self.cache_layer_list = self.cache_layer_list.copy()
        else:
            # all downsample layers
            self.cache_layer_list = []
            for i in range(len(self.unet.down_blocks)):
                self.cache_layer_list.append(i)

    def _forward_full(self, x: Tensor, timesteps: Tensor) -> Tensor:
        hs = []
        
        # time embedding
        t = self.unet.time_embed(timestep_embedding(timesteps, self.unet.model_channels))
        
        # down stage
        h = x
        for idx, module in enumerate(self.unet.down_blocks):
            h = module(h, t)
            hs.append(h)
    
            if idx in self.cache_layer_list:
                self.cache_features[idx] = h.clone()
        
        # middle stage
        h = self.unet.middle_block(h, t)
        
        # up stage
        for module in self.unet.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, t)
        
        return self.unet.out(h)

    def _forward_with_cache(self, x: Tensor, timesteps: Tensor):
        hs = []
        
        # down stage
        h = x
        t = self.unet.time_embed(timestep_embedding(timesteps, self.unet.model_channels))
        
        for idx, module in enumerate(self.unet.down_blocks):
            if idx in self.cache_layer_list:
                cached_h = self.cache_features[idx]    
                h = cached_h
            else:
                h = module(h, t)
            
            hs.append(h)
            
        # middle stage
        h = self.unet.middle_block(h, t)
        
        # up stage
        for module in self.unet.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, t)
        
        return self.unet.out(h)