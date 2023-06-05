'''
Code which implements the EnDi model
'''
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from numpy import sqrt as sqrt

from utils.util import nonzero_mean


class Memory:
    """ Class to store information used by the PPO class
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.predlogprobs = []
        self.targetactions = []
        self.rewards = []
        self.is_terminals = []
        self.texts = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.predlogprobs[:]
        del self.targetactions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.texts[:]


class EnDi(nn.Module):
    def __init__(self, state_h=10, state_w=10, action_dim=5, hist_len=3, n_latent_var=128,
                emb_dim=256, f_maps=64, kernel_size=2, n_hidden_layers=1):
        
        super().__init__()

        # calculate dimensions after flattening the conv layer output
        lin_dim = f_maps * (state_h - (kernel_size - 1)) * (
            state_w - (kernel_size - 1))
        self.conv_self = nn.Conv2d(hist_len*256, f_maps, kernel_size) # conv layer
        self.conv_other = nn.Conv2d(hist_len*256, f_maps, kernel_size) 

        self.state_h = state_h
        self.state_w = state_w
        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.attn_scale = sqrt(emb_dim)
    
        self.sprite_emb = nn.Embedding(25, emb_dim, padding_idx=0) # sprite embedding layer

        self.sprite_emb2 = nn.Embedding(25+2, emb_dim, padding_idx=0) # sprite embedding layer

        self.labor_weight = nn.Conv2d(hist_len*256*6, 2, kernel_size=(3, 3), padding='same')


        hidden_layers = (nn.Linear(n_latent_var, n_latent_var), nn.LeakyReLU())*n_hidden_layers
        self.action_self_layer = nn.Sequential(
                nn.Linear(lin_dim, n_latent_var),
                nn.LeakyReLU(),
                *hidden_layers,
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )

        self.action_other_layer = nn.Sequential(
                nn.Linear(lin_dim, n_latent_var),
                nn.LeakyReLU(),
                *hidden_layers,
                nn.Linear(n_latent_var, action_dim)
                )
        
        # critic 
        self.value_layer = nn.Sequential(
                nn.Linear(lin_dim, n_latent_var),
                nn.LeakyReLU(),
                *hidden_layers,
                nn.Linear(n_latent_var, 1)
                )

        # key value transforms
        self.txt_key = nn.Linear(768, emb_dim)
        self.scale_key = nn.Sequential(
            nn.Linear(768, 1),
            nn.Softmax(dim=-2)
        )
        
        self.txt_val = nn.Linear(768, emb_dim)
        self.scale_val = nn.Sequential(
            nn.Linear(768, 1),
            nn.Softmax(dim=-2)
        )

    def freeze_attention(self):
        # prevent gradient updates from changing attention weights
        for key, param in self.named_parameters():
            if 'key' in key: param.requires_grad = False
        self.sprite_emb.weight.requires_grad=False

    def attention(self, query, key, value):
        '''
        Cell by cell attention mechanism. Uses the sprite embeddings as query. Key is
        text embeddings
        '''
        kq = query @ key.t() # dot product attention
        mask = (kq != 0) # keep zeroed-out entries zero
        kq = kq / self.attn_scale # scale to prevent vanishing grads
        weights = F.softmax(kq, dim=-1) * mask
        return torch.mean(weights.unsqueeze(-1) * value, dim=-2), weights

    def batch_attention(self, query, key, value):
        """ Identical to attention() but does everything with additional batch dimension
        """
        bs = query.shape[0]
        kq = torch.bmm(
            query.view(bs, -1, self.emb_dim),
            key.permute(0, 2, 1)
        )
        kq = kq / self.attn_scale
        weights = F.softmax(kq, dim=-1)
        weights = weights * (kq != 0)
        weights = weights.view(*query.shape[:-1],-1)
        
        return torch.mean(
            weights.unsqueeze(-1) * value.view(bs,1,1,1,*value.shape[1:]), dim=-2
        )

    def forward(self):
        raise NotImplementedError

    def get_labor(self, state, temb):
        '''
        called when sampling actions from the policy
        '''
        # split the state tensor into objects and avatar
        obj_state, avt_state_self, avt_state_other, rel_pos_self_x, rel_pos_self_y, rel_pos_other_x, rel_pos_other_y = torch.split(state, [state.shape[-1]-1-1-1-1-1-1,1,1,1,1,1,1], dim=-1)
        avatar_self = nonzero_mean(self.sprite_emb(avt_state_self))
        avatar_other = nonzero_mean(self.sprite_emb(avt_state_other))
        # take non_zero mean of embedded state which acts as query
        query = nonzero_mean(self.sprite_emb(obj_state))
        # attention
        key = self.txt_key(temb) # (bs, num_sent, sent_len, emb_dim)
        key_scale = self.scale_key(temb) # (bs, num sent, sent_len, 1)
        key = key * key_scale
        key = torch.sum(key, dim=1) # (bs, num_sent, sent_len, emb_dim) -> (bs, num_sent, emb_dim)
        
        value = self.txt_val(temb)
        val_scale = self.scale_val(temb)
        value = value * val_scale
        value = torch.sum(value, dim=1)
        state_emb, weights= self.attention(query, key, value)
        state_emb = state_emb.view(self.state_h, self.state_w, -1)
        avatar_self = avatar_self.view(self.state_h, self.state_w, -1)
        avatar_other = avatar_other.view(self.state_h, self.state_w, -1)
        query_self = nonzero_mean(self.sprite_emb2(torch.cat((obj_state, rel_pos_self_x, rel_pos_self_y), dim=2)))
        labor_emb_self, weights_self = self.attention(query_self, key, value)
        labor_emb_self = labor_emb_self.view(self.state_h, self.state_w, -1)

        query_other = nonzero_mean(self.sprite_emb2(torch.cat((obj_state, rel_pos_other_x, rel_pos_other_y), dim=2)))
        labor_emb_other, weights_other = self.attention(query_other, key, value)
        labor_emb_other = labor_emb_other.view(self.state_h, self.state_w, -1)

        #obs_emb2, obs_emb3
        labor_weights = torch.cat((labor_emb_self, labor_emb_other), dim=2)
        labor_weights = self.labor_weight(labor_weights.permute(2, 0, 1))
        labor_prob = F.gumbel_softmax(labor_weights.permute(1, 2, 0), tau=1, hard=True)
        state_other = state_emb * labor_prob[:,:,1].view(self.state_h, self.state_w, -1)
        labor_self = labor_prob[:,:,0].long().cpu().numpy()
        labor_other = labor_prob[:,:,1].long().cpu().numpy()
        state_other = (state_other + avatar_other + avatar_self) / 3.0
        # permute from NHWC to NCHW and do convolution
        state_other = state_other.permute(2, 0, 1)
        state_other = F.leaky_relu(self.conv_other(state_other)).reshape(-1)

        # get action probs and values
        action_other_probs = self.action_other_layer(state_other)
        return labor_self, labor_other, labor_weights, action_other_probs


    def act(self, state, temb, memory, output_weights=False):
        '''
        called when sampling actions from the policy
        '''
        # split the state tensor into objects and avatar
        obj_state, avt_state_self, avt_state_other, rel_pos_self_x, rel_pos_self_y, rel_pos_other_x, rel_pos_other_y = torch.split(state, [state.shape[-1]-1-1-1-1-1-1,1,1,1,1,1,1], dim=-1)

        # embedding for the avatar object, which will not attend to text
        avatar_self = nonzero_mean(self.sprite_emb(avt_state_self))
        avatar_other = nonzero_mean(self.sprite_emb(avt_state_other))

        # take non_zero mean of embedded state which acts as query
        query = nonzero_mean(self.sprite_emb(obj_state))
        # attention
        key = self.txt_key(temb) # (bs, num_sent, sent_len, emb_dim)
        key_scale = self.scale_key(temb) # (bs, num sent, sent_len, 1)
        key = key * key_scale
        key = torch.sum(key, dim=1) # (bs, num_sent, sent_len, emb_dim) -> (bs, num_sent, emb_dim)
        
        value = self.txt_val(temb)
        val_scale = self.scale_val(temb)
        value = value * val_scale
        value = torch.sum(value, dim=1)
        state_emb, weights= self.attention(query, key, value)
        if output_weights:
            return weights
        
        state_emb = state_emb.view(self.state_h, self.state_w, -1)
        avatar_self = avatar_self.view(self.state_h, self.state_w, -1)
        avatar_other = avatar_other.view(self.state_h, self.state_w, -1)

        query_self = nonzero_mean(self.sprite_emb2(torch.cat((obj_state, rel_pos_self_x, rel_pos_self_y), dim=2)))
        labor_emb_self, weights_self = self.attention(query_self, key, value)
        labor_emb_self = labor_emb_self.view(self.state_h, self.state_w, -1)

        query_other = nonzero_mean(self.sprite_emb2(torch.cat((obj_state, rel_pos_other_x, rel_pos_other_y), dim=2)))
        labor_emb_other, weights_other = self.attention(query_other, key, value)
        labor_emb_other = labor_emb_other.view(self.state_h, self.state_w, -1)

        #obs_emb2, obs_emb3
        labor_weights = torch.cat((labor_emb_self, labor_emb_other), dim=2)
        labor_weights = self.labor_weight(labor_weights.permute(2, 0, 1))
        labor_prob = F.gumbel_softmax(labor_weights.permute(1, 2, 0), tau=1, hard=True)
        state_self = state_emb * labor_prob[:,:,0].view(self.state_h, self.state_w, -1)
        state_other = state_emb * labor_prob[:,:,1].view(self.state_h, self.state_w, -1)

        # Take the average between state_emb and avatar_emb in case of overlap
        state_self = (state_self + avatar_self + avatar_other) / 3.0
        # permute from NHWC to NCHW and do convolution
        state_self = state_self.permute(2, 0, 1)
        state_self = F.leaky_relu(self.conv_self(state_self)).reshape(-1)

        # get action probs and values
        action_self_probs = self.action_self_layer(state_self)
        
        state_other = (state_other + avatar_other + avatar_self) / 3.0
        # permute from NHWC to NCHW and do convolution
        state_other = state_other.permute(2, 0, 1)
        state_other = F.leaky_relu(self.conv_other(state_other)).reshape(-1)

        # get action probs and values
        action_other_probs = self.action_other_layer(state_other)

        if self.training:
            dist = Categorical(action_self_probs)
            action = dist.sample()

            # store everything in memory
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            memory.predlogprobs.append(action_other_probs)
            memory.texts.append(temb)
            return action.item()

        else:
            action = torch.argmax(action_self_probs).item()
            if random.random() < 0.05: # random action with 0.05 prob
                action = random.randrange(0, self.action_dim)
            return action

    
    def evaluate(self, state, action, temb):
        '''
        identical to self.act(), but with an additional batch dimension called by PPO algorithm when updating model params
        '''
        bs = state.shape[0] # batch size

        # split the state tensor into objects and avatar
        obj_state, avt_state_self, avt_state_other, rel_pos_self_x, rel_pos_self_y, rel_pos_other_x, rel_pos_other_y = torch.split(state, [state.shape[-1]-1-1-1-1-1-1,1,1,1,1,1,1], dim=-1)

        # embedding for the avatar object, which will not attend to text
        avatar_self = nonzero_mean(self.sprite_emb(avt_state_self))
        avatar_other = nonzero_mean(self.sprite_emb(avt_state_other))

        # take non_zero mean of embedded state which acts as query
        query = nonzero_mean(self.sprite_emb(obj_state))
        
        # attention
        key = self.txt_key(temb) # (bs, num_sent, sent_len, emb_dim)
        key_scale = self.scale_key(temb) # (bs, num sent, sent_len, 1)
        key = key * key_scale
        key = torch.sum(key, dim=2) # (bs, num_sent, sent_len, emb_dim) -> (bs, num_sent, emb_dim)
        
        value = self.txt_val(temb)
        val_scale = self.scale_val(temb)
        value = value * val_scale
        value = torch.sum(value, dim=2)
        
        state = self.batch_attention(query, key, value)
        
        # compress the channels from NKHWC to BHWC' where K is history length
        state = state.view(bs, self.state_h, self.state_w, -1)
        avatar_self = avatar_self.view(bs, self.state_h, self.state_w, -1)
        avatar_other = avatar_other.view(bs, self.state_h, self.state_w, -1)

        query_self = nonzero_mean(self.sprite_emb2(torch.cat((obj_state, rel_pos_self_x, rel_pos_self_y), dim=3)))
        labor_emb_self = self.batch_attention(query_self, key, value)
        labor_emb_self = labor_emb_self.view(bs, self.state_h, self.state_w, -1)

        query_other = nonzero_mean(self.sprite_emb2(torch.cat((obj_state, rel_pos_other_x, rel_pos_other_y), dim=3)))
        labor_emb_other = self.batch_attention(query_other, key, value)
        labor_emb_other = labor_emb_other.view(bs, self.state_h, self.state_w, -1)

        #obs_emb2, obs_emb3
        labor_weights = torch.cat((labor_emb_self, labor_emb_other), dim=3)
        labor_weights = self.labor_weight(labor_weights.permute(0, 3, 1, 2))
        labor_prob = F.gumbel_softmax(labor_weights.permute(0, 2, 3, 1), tau=1, hard=True)
        state_self = state * labor_prob[:,:,:,0].view(bs, self.state_h, self.state_w, -1)
        state_other = state * labor_prob[:,:,:,1].view(bs, self.state_h, self.state_w, -1)

        # Take the average between state_emb and avatar_emb in case of overlap
        state_self = (state_self + avatar_self + avatar_other) / 3.0

        # permute from NHWC to NCHW and do convolution
        state_self = state_self.permute(0, 3, 1, 2)
        state_self = F.leaky_relu(self.conv_self(state_self)).reshape(bs, -1)

        # get action probs and values
        action_self_probs = self.action_self_layer(state_self)
        dist = Categorical(action_self_probs)
        action_self_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state_self)

        state_other = (state_other + avatar_other + avatar_self) / 3.0
        # permute from NHWC to NCHW and do convolution
        state_other = state_other.permute(0, 3, 1, 2)
        state_other = F.leaky_relu(self.conv_other(state_other)).reshape(bs, -1)

        # get action probs and values
        action_other_probs = self.action_other_layer(state_other)

        return action_self_logprobs, torch.squeeze(state_value), dist_entropy, action_other_probs, labor_prob
