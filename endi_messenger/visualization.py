
import argparse
from audioop import avg
import hashlib
import torch
import numpy as np

import gym
from model import EnDi, Memory
from train_tools import ObservationBuffer, PPO, TrainStats
import endi_messenger
from transformers import AutoModel, AutoTokenizer
from utils.util import Encoder
from torch.nn import CrossEntropyLoss

def numpy_formatter(i: int):
    ''' Format function passed to numpy print to make things pretty.
    '''
    id_map = {}
    for ent in endi_messenger.envs.config.ALL_ENTITIES:
        id_map[ent.id] = ent.name[:2].upper()
    id_map[0] = '  '
    id_map[15] = 'A0'
    id_map[16] = 'AM'
    id_map[-1] = '@1'
    id_map[-2] = '@2'
    id_map[-3] = '#1'
    id_map[-4] = '#2'
    if i < 17:
        return id_map[i]
    else:
        return 'XX'

def print_instructions():
    ''' Print the Messenger instructions and header.
    '''
    print(f"\nMESSENGER\n")
    print("Read the manual to get the message and bring it to the goal.")
    print("A0 is you (agent) without the message, and AM is you with the message.")
    print("The following is the symbol legend (symbol : entity)\n")
    for ent in endi_messenger.envs.config.ALL_ENTITIES[:12]:
        print(f"{ent.name[:2].upper()} : {ent.name}")    
    print("\nNote when entities overlap the symbol might not make sense. Good luck!\n")

def get_grid(obs):
    ''' Print the observation to terminal
    '''
    grid = np.concatenate((obs['entities'], obs['avatar1'], obs['avatar2']), axis=-1)
    return np.sum(grid, axis=-1).astype('uint8')

def get_state_grid(obs):
    ''' Print the observation to terminal
    '''
    return obs['entities'][:,:,0].astype('uint8')

def print_manual(manual):
    ''' Print the manual to terminal
    '''
    man_str = f"Manual: {manual[0]}\n"
    for description in manual[1:]:
        man_str += f"        {description}\n"
    print(man_str)

def print_results(obs_stack):    
    obs_stack = np.array2string(obs_stack)\
        .replace('\n ','')\
        .replace('[[[','[[')\
        .replace(']]]',']]')\
        .replace('[[', '[')\
        .replace(']]', ']')
    print(obs_stack)

def clear_terminal():
    ''' Special print that will clear terminal after each step.
    Replace with empty return if your terminal has issues with this.
    ''' 
    # print(chr(27) + "[2J")
    print("\033c\033[3J")

def wrap_obs1(obs):
    """ Convert obs format returned by gym env (dict) to a numpy array expected by model
    """
    return np.concatenate(
        (obs["entities"], obs["avatar1"], obs["avatar2"], obs["rel_pos1x"], obs["rel_pos1y"], obs["rel_pos2x"], obs["rel_pos2y"]), axis=-1
    )

def wrap_obs2(obs):
    """ Convert obs format returned by gym env (dict) to a numpy array expected by model
    """
    return np.concatenate(
        (obs["entities"], obs["avatar2"], obs["avatar1"], obs["rel_pos2x"], obs["rel_pos2y"], obs["rel_pos1x"], obs["rel_pos1y"]), axis=-1
    )

def get_mask(obs):
    mask =  (obs["entities"] > 0).astype(np.int32)
    return torch.tensor(mask[:,:,0]).to(args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--seed", default=None, type=int, help="Set the seed for the model and training.")
    parser.add_argument("--device", default=0, type=int, help="cuda device ordinal to train on.")

    # Model arguments
    parser.add_argument("--load_state", default=None, help="Path to model state dict.")
    parser.add_argument("--latent_vars", default=128, type=int, help="Latent model dimension.")
    parser.add_argument("--hist_len", default=3, type=int, help="Length of history used by state buffer")
    parser.add_argument("--emb_dim", default=256, type=int, help="embedding size for text")

    # Environment arguments
    parser.add_argument("--stage", default="1", type=str, help="the stage to run experiment on")
    parser.add_argument("--max_steps", default=16, type=int, help="Maximum num of steps per episode")
    parser.add_argument("--step_penalty", default=0.0, type=float, help="negative reward for each step")
    
    # Training arguments
    parser.add_argument("--update_timestep", default=64, type=int, help="Number of steps before model update")
    parser.add_argument("--lr", default=0.00005, type=float, help="learning rate")
    parser.add_argument("--gamma", default=0.8, type=float, help="discount factor")
    parser.add_argument("--k_epochs", default=4, type=int, help="num epochs to update")
    parser.add_argument("--eps_clip", default=0.1, type=float, help="clip param for PPO")
    parser.add_argument("--optimizer", default="Adam", type=str, help="optimizer class to use")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay for optimizer")
    parser.add_argument("--max_time", default=1000, type=float, help="max train time in hrs")
    parser.add_argument("--max_eps", default=1e10, type=float, help="max training episodes")
    parser.add_argument("--freeze_attention", action="store_true", help="Do not update attention weights.")

    # Logging arguments
    parser.add_argument('--log_interval', default=5000, type=int, help='number of episodes between logging')
    parser.add_argument('--eval_interval', default=25000, type=int, help='number of episodes between eval')
    parser.add_argument('--eval_eps', default=1, type=int, help='number of episodes to run eval')
    parser.add_argument('--log_group', type=str, help="wandb log group")
    parser.add_argument('--entity', type=str, help="entity to log runs to on wandb")
    parser.add_argument('--check_script', action='store_true', help="run quickly just to see script runs okay.")

    parser.add_argument('--print_grid', default=1)
    parser.add_argument("--load_state1", default=None, help="Path to model state dict.")
    parser.add_argument("--load_state2", default=None, help="Path to model state dict.")

    args = parser.parse_args()
    
    # get hash of arguments minus seed
    args_dict = vars(args).copy()
    args_dict["device"] = None
    args_dict["seed"] = None
    args_dict["output"] = None
    args_hash = hashlib.md5(
        str(sorted(args_dict.items())).encode("utf-8")
    ).hexdigest()

    args.device = torch.device(f"cuda:{args.device}")

    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = f'msgr-train-v{args.stage}'

    #####################################################
    
    # Text Encoder
    encoder_model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = Encoder(model=encoder_model, tokenizer=tokenizer, device=args.device, max_length=36)

    model_kwargs = {
        "hist_len": args.hist_len,
        "n_latent_var": args.latent_vars,
        "emb_dim": args.emb_dim,
    }

    optim_kwargs = {
        "weight_decay": args.weight_decay
    }

    ppo1 = PPO(
        ModelCls = EnDi,
        model_kwargs = model_kwargs,
        device = args.device,
        lr = args.lr,
        gamma = args.gamma,
        K_epochs = args.k_epochs,
        eps_clip = args.eps_clip,
        load_state = args.load_state,
        optim_kwargs=optim_kwargs,
        optimizer=args.optimizer
    )

    ppo2 = PPO(
        ModelCls = EnDi,
        model_kwargs = model_kwargs,
        device = args.device,
        lr = args.lr,
        gamma = args.gamma,
        K_epochs = args.k_epochs,
        eps_clip = args.eps_clip,
        load_state = args.load_state,
        optim_kwargs=optim_kwargs,
        optimizer=args.optimizer
    )

    # freeze attention mechanism
    if args.freeze_attention:
        ppo1.policy.freeze_attention()
        ppo1.policy_old.freeze_attention()
        ppo2.policy.freeze_attention()
        ppo2.policy_old.freeze_attention()

    ppo1.policy.load_state_dict(torch.load(args.load_state1, map_location=args.device))
    ppo2.policy.load_state_dict(torch.load(args.load_state2, map_location=args.device))
    ppo1.policy.eval()
    ppo2.policy.eval()

    buffer1 = ObservationBuffer(buffer_size=3, device=args.device)
    buffer2 = ObservationBuffer(buffer_size=3, device=args.device)

    np.set_printoptions(formatter={'int': numpy_formatter})
    env = gym.make(env_name)

    for _ in range(args.eval_eps):
        obs, text = env.reset()
        if args.print_grid == 1:
            print_instructions()
            print_manual(text)
            obs_grid = np.array2string(get_grid(obs))\
                .replace('[[',' [')\
                .replace(']]',']')\
                .replace(' [', '[')
            print(obs_grid)
            print('\n')
        text = encoder.encode(text)
        obs1 = wrap_obs1(obs)
        obs2 = wrap_obs2(obs)  
        buffer1.reset(obs1)
        buffer2.reset(obs2)

        for t in range(args.max_steps):
            with torch.no_grad():
                action1 = ppo1.policy.act(buffer1.get_obs(), text, None)
                action2 = ppo2.policy.act(buffer2.get_obs(), text, None)
                labor1_self, labor1_other, _, _ = ppo1.policy.get_labor(buffer1.get_obs(), text)
                labor2_self, labor2_other, _, _ = ppo2.policy.get_labor(buffer2.get_obs(), text)

            obs, reward, done, _ = env.step((action1, action2))
            if args.print_grid == 1:

                obs_grid = get_grid(obs)
                state_grid = get_state_grid(obs)
                prints1 = np.stack([
                    obs_grid, 
                    labor1_self*-1, 
                    labor1_other*-2, 
                    labor1_self * state_grid], 
                axis=1)
                
                prints2 = np.stack([
                    np.zeros_like(obs_grid), 
                    labor2_other*-3, 
                    labor2_self*-4, 
                    labor2_self * state_grid], 
                axis=1)

                print_results(prints1)
                print()
                print_results(prints2)
                print()
                
            obs1 = wrap_obs1(obs)
            obs2 = wrap_obs2(obs)    
            if t == args.max_steps - 1 and reward < 1:
                reward = -1.0 # failed to complete objective
                done = True    
            win = False
            if done and reward >= 1:
                win =  True
            if t == args.max_steps - 1 and not done:
                win = False                
            if done:
                break
            buffer1.update(obs1)
            buffer2.update(obs2)

    print("============================================================================================")
