# Implementation of EnDi and Multi-Agent Messenger

Implementation of the EnDi model and Multi-Agent Messenger environment from the ICML 2023 paper: Entity Divider with Language Grounding in Multi-Agent Reinforcement Learning.

## Setup

The conda environment we use is provided in `environment.yml`. To create the same environment, run

```shell
conda env create -f environment.yml
conda activate endi-messenger
pip install -e .
```

## Running

To reproduce results of our paper, please follow the instructions below.

### Prepare wandb

We use [wandb](https://github.com/wandb/wandb) library to store the training results. Please follow this [link](https://github.com/wandb/wandb) for detailed instructions about wandb.

### Stage 1 curriculum

Train from scratch for stage 1.

```shell
python train.py --stage 1 --max_steps 16 --sloss --dloss --seed 1 --entity "<your_wandb_name>" --log_group "stage_1"  --output output/stage_1/s1
```

### Stage 2 curriculum

After stage 1 curriculum, run the following command for stage 2 curriculum.

```shell
python train.py --stage 2 --max_steps 32 --sloss --dloss --seed 1 --entity "<your_wandb_name>" --log_group "stage_2" --load_state1 output/stage_1/s1_max1.pth --load_state2 output/stage_1/s1_max2.pth --freeze_attention --output output/stage_2/s2
```

### Stage 3 curriculum

After stage 2 curriculum, run the following command for stage 3 curriculum.

```shell
python train.py --stage 3 --max_steps 64 --sloss --dloss --seed 1 --entity "<your_wandb_name>" --log_group "stage_3" --load_state1 output/stage_2/s2_max1.pth --load_state2 output/stage_2/s2_max2.pth --freeze_attention --output output/stage_3/s3
```

### Loss Functions

You may choose different loss functions. For standard EnDi, choose `--sloss --dloss` as the commands above. For other variants in the paper, please follow the instructions below:

 - EnDi(num): `--sloss --lloss`
 - EnDi-sup: `--dloss`
 - EnDi-reg: `--sloss`

### Results and Models

The results are stored with [wandb](https://github.com/wandb/wandb) library. Please follow this [link](https://github.com/wandb/wandb) to use it. The trained models will be stored in `output/`.

## Policy Visualization

After training, you can visualize our trained policy by running the following commands.

```shell
python visualization.py --stage 1 --load_state1 output/stage_1/s1_max1.pth --load_state2 output/stage_1/s1_max2.pth
python visualization.py --stage 2 --load_state1 output/stage_2/s2_max1.pth --load_state2 output/stage_2/s2_max2.pth
python visualization.py --stage 3 --load_state1 output/stage_3/s3_max1.pth --load_state2 output/stage_3/s3_max2.pth
```

## Human Play

You can play the multi-agent messenger by running the following commands. The number after "stage" can be selected as 1, 2 or 3, which correspond to the stages we used during training.

```shell
python play_msgr.py --stage 1
```

In each step, you need to enter the actions of two agents at the same time, separated by a space. The available actions are `[w,a,s,d,'']`, where `''` represents not entering any character.
