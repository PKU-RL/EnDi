import torch
import torch.nn as nn

import endi_messenger.envs.config as config
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from itertools import permutations


class ObservationBuffer:
    '''
    Maintains a buffer of observations along the 0-dim.
    
    Parameters:
    buffer_size
        How many previous observations to track in the buffer
    device
        The device on which buffers are loaded into

    '''
    def __init__(self, buffer_size, device,):
        self.buffer_size = buffer_size
        self.buffer = None
        self.device = device

    def _np_to_tensor(self, obs):
        return torch.from_numpy(obs).long().to(self.device)

    def reset(self, obs):
        # initialize / reset the buffer with the observation
        self.buffer = [self._np_to_tensor(obs) for _ in range(self.buffer_size)]

    def update(self, obs):
        # update the buffer by appending newest observation
        assert self.buffer, "Please initialize buffer first with reset()"
        del self.buffer[0] # delete the oldest entry
        self.buffer.append(self._np_to_tensor(obs)) # append the newest observation

    def get_obs(self):
        # get a stack of all observations currently in the buffer
        return torch.stack(self.buffer)


class PPO:
    def __init__(self, ModelCls, model_kwargs, device, lr, gamma, K_epochs, eps_clip, load_state=None, load_optim=None,
                optim_kwargs = {}, optimizer="Adam"
    ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.MseLoss = nn.MSELoss()
        self.policy = ModelCls(**model_kwargs).to(device)
        self.policy_old = ModelCls(**model_kwargs).to(device)
        
        for p in self.policy.parameters(): # xavier initialize model weights
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if load_state: # initialize model weights if one is specified
            self.policy.load_state_dict(torch.load(load_state, map_location=device))

        self.policy_old.load_state_dict(self.policy.state_dict())

        OptimCls = getattr(torch.optim, optimizer)
        self.optimizer = OptimCls(
            self.policy.parameters(),
            lr=lr,
            **optim_kwargs
        )

        if load_optim:
            self.optimizer.load_state_dict(torch.load(load_optim, map_location=device))
        
        self.policy.train()
        self.policy_old.train()


    def compute_supervised_loss(self, other_logits, actual_logits):
        loss = CrossEntropyLoss()
        result = loss(other_logits, actual_logits.long())
        return result

    def update(self, memory, mask, lloss=False, sloss=False, dloss=False, ploss=False, s_a=1.0):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        old_texts = torch.stack(memory.texts).to(self.device).detach()
        # old_predlogprobs = torch.stack(memory.predlogprobs).to(self.device).detach()
        if sloss:
            old_targetactions = torch.stack(memory.targetactions).to(self.device).detach()

        l_loss = 0
        s_loss = 0
        d_loss = 0
        p_loss = 0
        total_loss = 0
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            if lloss or dloss or ploss or sloss:
                logprobs, state_values, dist_entropy, action_other_probs, labor_prob = self.policy.evaluate(old_states, old_actions, old_texts)
            else:
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_texts)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) \
                - 0.01 * dist_entropy
            
            if lloss:
                labor_probm = torch.stack((labor_prob[:,:,:,0] * mask, labor_prob[:,:,:,1] * mask), dim=-1)
                labor_probm = labor_probm.reshape(-1, config.STATE_HEIGHT * config.STATE_WIDTH, 2).contiguous()

                labor_prior_loss = torch.square(torch.sum(labor_probm[:,:,0], dim=1) - torch.sum(labor_probm[:,:,1], dim=1))
                loss += labor_prior_loss   
                l_loss += labor_prior_loss.mean().item()        
            
            if dloss:
                _, _, _, rel_pos1x, rel_pos1y, rel_pos2x, rel_pos2y = torch.split(old_states, [old_states.shape[-1]-1-1-1-1-1-1,1,1,1,1,1,1], dim=-1)
                labor_probm = torch.stack((labor_prob[:,:,:,0] * mask, labor_prob[:,:,:,1] * mask), dim=-1)
                rel_pos1x = (rel_pos1x[:,-1,:,:,0] - config.STATE_WIDTH) / config.STATE_WIDTH * labor_probm[:,:,:,0]
                rel_pos1y = (rel_pos1y[:,-1,:,:,0] - config.STATE_HEIGHT) / config.STATE_HEIGHT * labor_probm[:,:,:,0]
                rel_pos2x = (rel_pos2x[:,-1,:,:,0] - config.STATE_WIDTH) / config.STATE_WIDTH * labor_probm[:,:,:,1]
                rel_pos2y = (rel_pos2y[:,-1,:,:,0] - config.STATE_HEIGHT) / config.STATE_HEIGHT * labor_probm[:,:,:,1]
                d1 = torch.square(rel_pos1x) + torch.square(rel_pos1y)
                d2 = torch.square(rel_pos2x) + torch.square(rel_pos2y)

                dis_loss = torch.square(torch.sum(d1, axis=(1,2)) - torch.sum(d2, axis=(1,2)))
                loss += dis_loss
                d_loss += dis_loss.mean().item()

            if ploss:
                _, _, _, rel_pos1x, rel_pos1y, rel_pos2x, rel_pos2y = torch.split(old_states, [old_states.shape[-1]-1-1-1-1-1-1,1,1,1,1,1,1], dim=-1)
                labor_probm = torch.stack((labor_prob[:,:,:,0] * mask, labor_prob[:,:,:,1] * mask), dim=-1)
                rel_pos1x = (rel_pos1x[:,-1,:,:,0] - config.STATE_WIDTH) / config.STATE_WIDTH
                rel_pos1y = (rel_pos1y[:,-1,:,:,0] - config.STATE_HEIGHT) / config.STATE_HEIGHT
                rel_pos2x = (rel_pos2x[:,-1,:,:,0] - config.STATE_WIDTH) / config.STATE_WIDTH
                rel_pos2y = (rel_pos2y[:,-1,:,:,0] - config.STATE_HEIGHT) / config.STATE_HEIGHT

                pathloss = []
                for k in range(len(labor_probm)):

                    d1 = torch.abs(rel_pos1x[k]) + torch.abs(rel_pos1y[k])
                    d2 = torch.abs(rel_pos2x[k]) + torch.abs(rel_pos2y[k])
                    s1 = (d1==0).int()
                    s2 = (d2==0).int()
                    id1 = torch.nonzero((mask + s1) * (labor_probm[k,:,:,0] + s1))
                    id2 = torch.nonzero((mask + s2) * (labor_probm[k,:,:,1] + s2))
                    tmp1 = []
                    tmp2 = []
                    for i,j in id1:
                        i,j = i.item(), j.item()
                        tmp1.append((torch.abs(rel_pos1x[k]-rel_pos1x[k][i, j])+torch.abs(rel_pos1y[k]-rel_pos1y[k][i, j]))[id1[:,0], id1[:,1]])
                    for i,j in id2:
                        tmp2.append((torch.abs(rel_pos2x[k]-rel_pos2x[k][i, j])+torch.abs(rel_pos2y[k]-rel_pos2y[k][i, j]))[id2[:,0], id2[:,1]])
                    l1 = []
                    l2 = []
                    tmp1 = torch.stack(tmp1)
                    tmp2 = torch.stack(tmp2)
                    if len(tmp1) > 1:
                        for i in permutations(range(1, len(tmp1)), len(tmp1) - 1):
                            tmp = 0
                            tmp += tmp1[0, i[0]]
                            for j in range(len(i) - 1):
                                tmp += tmp1[i[j], i[j + 1]]
                            l1.append(tmp)
                    else:
                        l1.append(torch.tensor(0))
                    
                    if len(tmp2) > 1:
                        for i in permutations(range(1, len(tmp2)), len(tmp2) - 1):
                            tmp = 0
                            tmp += tmp2[0, i[0]]
                            for j in range(len(i) - 1):
                                tmp += tmp2[i[j], i[j + 1]]
                            l2.append(tmp)
                    else:
                        l2.append(torch.tensor(0))
                    
                    pathloss.append(min(l1) + min(l2))
                
                pathloss = torch.stack(pathloss)
                loss += pathloss
                p_loss += pathloss.mean().item()

            if sloss:
                supervised_loss = self.compute_supervised_loss(action_other_probs, old_targetactions)
                loss += s_a * supervised_loss
                s_loss += supervised_loss.item()
            
            total_loss += loss.mean().item()
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # zero out the padding_idx for sprite embedder (temp fix for PyTorch bug)
        self.policy.sprite_emb.weight.data[0] = 0
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        s_loss = s_loss / self.K_epochs
        l_loss = l_loss / self.K_epochs
        d_loss = d_loss / self.K_epochs
        p_loss = p_loss / self.K_epochs
        total_loss = total_loss / self.K_epochs

        return s_loss, l_loss, d_loss, p_loss, total_loss


class TrainStats:
    '''
    Class for tracking agent reward and other statistics.
    Params:
        rewards_key: a dictionary that maps from reward (float) to a string that describes the meaning of the reward. E.g. {1:'resource', 2:'key', 3:'win'}
    '''
    def __init__(self, rewards_key):
        self._reserved = ['avg_reward', 'avg_length', 'episodes']
        self._rk = rewards_key.copy()
        self.stats = {}
        for event in rewards_key.values():
            if event in self._reserved:
                raise Exception(event + ' is a reserved event word')
            self.stats[event] = 0
        self.all_rewards = [] # end of episode rewards
        self.eps_reward = 0 # track rewards in episode
        self.steps = 0 # steps since last reset()
        self.total_steps = 0 # steps since object instantiation
        self.episodes = 0
        self.loss_steps = 0
        self.s_loss1 = []
        self.s_loss2 = []
        self.l_loss1 = []
        self.l_loss2 = []
        self.d_loss1 = []
        self.d_loss2 = []
        self.p_loss1 = []
        self.p_loss2 = []
        self.t_loss1 = []
        self.t_loss2 = []
        
    # return str stats
    def __str__(self):
        if self.episodes == 0:
            return 'No stats'
        stats = f'lengths: {self.steps/self.episodes:.2f} \t '
        stats += f'rewards: {sum(self.all_rewards)/self.episodes:.2f} \t '
        for key, val in self.stats.items():
            stats += (key + 's: ' + f'{val/self.episodes:.2f}' + ' \t ')
        return stats
        
    # adds reward to current set of stats
    def step(self, reward):
        self.steps += 1
        self.total_steps += 1
        self.eps_reward += reward
        for key, event in self._rk.items():
            if reward == key:
                self.stats[event] += 1

    # adds reward to current set of stats
    def log_loss(self, s_loss1, s_loss2, l_loss1, l_loss2, d_loss1, d_loss2, p_loss1, p_loss2, t_loss1, t_loss2):
        self.loss_steps += 1
        self.s_loss1.append(s_loss1)
        self.s_loss2.append(s_loss2)
        self.l_loss1.append(l_loss1)
        self.l_loss2.append(l_loss2)
        self.d_loss1.append(d_loss1)
        self.d_loss2.append(d_loss2)
        self.p_loss1.append(p_loss1)
        self.p_loss2.append(p_loss2)
        self.t_loss1.append(t_loss1)
        self.t_loss2.append(t_loss2)

    # end of episode
    def end_of_episode(self):
        self.episodes += 1
        self.all_rewards.append(self.eps_reward)
        self.eps_reward = 0
    
    # reset gamestats
    def reset(self):
        for key in self.stats.keys():
            self.stats[key] = 0
        self.episodes = 0
        self.all_rewards = []
        self.eps_reward = 0
        self.steps = 0
        self.episodes = 0
        self.loss_steps = 0
        self.s_loss1 = []
        self.s_loss2 = []
        self.l_loss1 = []
        self.l_loss2 = []
        self.d_loss1 = []
        self.d_loss2 = []
        self.p_loss1 = []
        self.p_loss2 = []
        self.t_loss1 = []
        self.t_loss2 = []

    # the running reward
    def running_reward(self):
        assert self.episodes > 0, 'number of episodes = 0'
        return sum(self.all_rewards) / self.episodes
        
    # compress all stats into single dict. Append is optional dict to append
    def compress(self, append=None, train=True):
        assert self.episodes > 0, 'No stats to compress'
        stats = {event: num/self.episodes for (event, num) in self.stats.items()}
        stats['step'] = self.total_steps
        stats['avg_reward'] = sum(self.all_rewards) / self.episodes
        stats['avg_length'] = self.steps / self.episodes
        stats['episodes'] = self.episodes
        if train:
            stats['supervised_loss_1'] = sum(self.s_loss1) / self.loss_steps
            stats['supervised_loss_2'] = sum(self.s_loss2) / self.loss_steps
            stats['labor_loss_1'] = sum(self.l_loss1) / self.loss_steps
            stats['labor_loss_2'] = sum(self.l_loss2) / self.loss_steps
            stats['dis_loss_1'] = sum(self.d_loss1) / self.loss_steps
            stats['dis_loss_2'] = sum(self.d_loss2) / self.loss_steps
            stats['path_loss_1'] = sum(self.p_loss1) / self.loss_steps
            stats['path_loss_2'] = sum(self.p_loss2) / self.loss_steps
            stats['total_loss_1'] = sum(self.t_loss1) / self.loss_steps
            stats['tatal_loss_2'] = sum(self.t_loss2) / self.loss_steps
        if append:
            stats.update(append)
        return stats