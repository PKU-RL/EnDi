'''
Classes that follows a gym-like interface and implements stage one of the Messenger
environment.
'''

import json
import random
from collections import namedtuple
from pathlib import Path

import numpy as np

from endi_messenger.envs.base import MessengerEnv, Position
import endi_messenger.envs.config as config
from endi_messenger.envs.manual import TextManual
from endi_messenger.envs.utils import games_from_json


Sprite = namedtuple("Sprite", ["name", "id", "position"])


class StageOne(MessengerEnv):
    def __init__(self, split, message_prob=0.2, shuffle_obs=True):
        '''
        message_prob:
            the probability that the avatar starts with the message
        shuffle_obs:
            shuffle the observation including the text manual
        '''
        super().__init__()
        self.message_prob = message_prob
        self.shuffle_obs = shuffle_obs
        this_folder = Path(__file__).parent
        
        # Get the games and manual
        games_json_path = this_folder.joinpath("games.json")
        if "train" in split and "mc" in split: # multi-combination games
            game_split = "train_multi_comb"
            text_json_path = this_folder.joinpath("texts", "text_train.json")
        elif "train" in split and "sc" in split: # single-combination games
            game_split = "train_single_comb"
            text_json_path = this_folder.joinpath("texts", "text_train.json")
        elif "val" in split:
            game_split = "val"
            text_json_path = this_folder.joinpath("texts", "text_val.json")
        elif "test" in split:
            game_split = "test"
            text_json_path = this_folder.joinpath("texts", "text_test.json")
        else:
            raise Exception(f"Split: {split} not understood.")

        # list of Game namedtuples
        self.all_games = games_from_json(json_path=games_json_path, split=game_split)
        
        # we only need the immovable and unknown descriptions, so just extract those.
        with text_json_path.open(mode="r") as f:
            descrip = json.load(f)
        
        self.descriptors = {}
        for entity in descrip:
            self.descriptors[entity] = {}
            for role in ("enemy", "message", "goal"):
                self.descriptors[entity][role] = []
                for sent in descrip[entity][role]["immovable"]:
                    self.descriptors[entity][role].append(sent)
                for sent in descrip[entity][role]["unknown"]:
                    self.descriptors[entity][role].append(sent)
        
        self.positions = [ # all possible entity locations
            Position(y=7, x=4),
            Position(y=7, x=6),
            Position(y=3, x=5),
            Position(y=5, x=3),
            Position(y=5, x=7),
        ]
        self.avatar1_start_pos = Position(y=5, x=5)
        self.avatar2_start_pos = Position(y=5, x=5)
        self.avatar1 = None
        self.avatar2 = None
        self.enemy = None
        self.message1 = None
        self.message2 = None
        self.neutral = None
        self.goal1 = None
        self.goal2 = None

    def _get_manual(self):
        enemy_str = random.choice(self.descriptors[self.enemy.name]["enemy"])
        key_str = random.choice(self.descriptors[self.message1.name]["message"])
        goal_str = random.choice(self.descriptors[self.goal1.name]["goal"])
        manual = [enemy_str, key_str, goal_str]
        if self.shuffle_obs:
            random.shuffle(manual)
        return manual

    def _get_obs(self):
        entities = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        avatar1 = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        avatar2 = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        
        x_offset1 = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        x_offset2 = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        y_offset1 = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        y_offset2 = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))

        for sprite in (self.enemy, self.message1, self.message2, self.goal1, self.goal2):
            entities[sprite.position.y, sprite.position.x, 0] = sprite.id

        for i in range(config.STATE_WIDTH):
            x_offset1[:, i] = i - self.avatar1.position.x + config.STATE_WIDTH
        for i in range(config.STATE_HEIGHT):
            y_offset1[i, :] = i - self.avatar1.position.y + config.STATE_HEIGHT            
        for i in range(config.STATE_WIDTH):
            x_offset2[:, i] = i - self.avatar2.position.x + config.STATE_WIDTH
        for i in range(config.STATE_HEIGHT):
            y_offset2[i, :] = i - self.avatar2.position.y + config.STATE_HEIGHT

        rel_pos1x = x_offset1
        rel_pos1y = y_offset1
        rel_pos2x = x_offset2
        rel_pos2y = y_offset1

        avatar1[self.avatar1.position.y, self.avatar1.position.x, 0] = self.avatar1.id
        avatar2[self.avatar2.position.y, self.avatar2.position.x, 0] = self.avatar2.id
        
        return {"entities": entities, "avatar1": avatar1, "avatar2": avatar2, 'rel_pos1x': rel_pos1x, 'rel_pos1y': rel_pos1y, 'rel_pos2x': rel_pos2x, 'rel_pos2y':rel_pos2y }

    def reset(self):
        self.game = random.choice(self.all_games)
        enemy, message1, goal1 = self.game.enemy, self.game.message, self.game.goal
        message2, goal2 = self.game.message, self.game.goal

        # randomly choose where to put enemy, key, goal
        shuffled_pos = random.sample(self.positions, 5)
        self.enemy = Sprite(name=enemy.name, id=enemy.id, position=shuffled_pos[0])
        self.message1 = Sprite(name=message1.name, id=message1.id, position=shuffled_pos[1])
        self.goal1 = Sprite(name=goal1.name, id=goal1.id, position=shuffled_pos[2])
        self.message2 = Sprite(name=message2.name, id=message2.id, position=shuffled_pos[3])
        self.goal2 = Sprite(name=goal2.name, id=goal2.id, position=shuffled_pos[4])
        
        if random.random() < self.message_prob:
            self.avatar1 = Sprite(
                name=config.WITH_MESSAGE.name,
                id=config.WITH_MESSAGE.id,
                position=self.avatar1_start_pos
            )
            self.avatar2 = Sprite(
                name=config.WITH_MESSAGE.name,
                id=config.WITH_MESSAGE.id,
                position=self.avatar2_start_pos
            )

        else: # decide whether avatar has message or not
            self.avatar1 = Sprite(
                name=config.NO_MESSAGE.name,
                id=config.NO_MESSAGE.id,
                position=self.avatar1_start_pos
            )
            self.avatar2 = Sprite(
                name=config.NO_MESSAGE.name,
                id=config.NO_MESSAGE.id,
                position=self.avatar2_start_pos
            )
        
        obs = self._get_obs()
        manual = self._get_manual()

        return obs, manual
    
    def _move_avatar1(self, action):
        if action == config.ACTIONS.stay:
            return
        
        elif action == config.ACTIONS.up: 
            if self.avatar1.position.y <= 0:
                return
            else:
                new_position = Position(
                    y = self.avatar1.position.y - 1,
                    x = self.avatar1.position.x
                )
                
        elif action == config.ACTIONS.down: 
            if self.avatar1.position.y >= config.STATE_HEIGHT - 1:
                return
            else:
                new_position = Position(
                    y = self.avatar1.position.y + 1,
                    x = self.avatar1.position.x
                )
                
        elif action == config.ACTIONS.left: 
            if self.avatar1.position.x <= 0:
                return
            else:
                new_position = Position(
                    y = self.avatar1.position.y,
                    x = self.avatar1.position.x - 1
                )
                
        elif action == config.ACTIONS.right: 
            if self.avatar1.position.x >= config.STATE_WIDTH - 1:
                return
            else:
                new_position = Position(
                    y = self.avatar1.position.y,
                    x = self.avatar1.position.x + 1
                )
                
        else:
            raise Exception(f"{action} is not a valid action.")
            
        self.avatar1 = Sprite(
                name=self.avatar1.name,
                id=self.avatar1.id,
                position=new_position
            )

    def _move_avatar2(self, action):
        if action == config.ACTIONS.stay:
            return
        
        elif action == config.ACTIONS.up: 
            if self.avatar2.position.y <= 0:
                return
            else:
                new_position = Position(
                    y = self.avatar2.position.y - 1,
                    x = self.avatar2.position.x
                )
                
        elif action == config.ACTIONS.down: 
            if self.avatar2.position.y >= config.STATE_HEIGHT - 1:
                return
            else:
                new_position = Position(
                    y = self.avatar2.position.y + 1,
                    x = self.avatar2.position.x
                )
                
        elif action == config.ACTIONS.left: 
            if self.avatar2.position.x <= 0:
                return
            else:
                new_position = Position(
                    y = self.avatar2.position.y,
                    x = self.avatar2.position.x - 1
                )
                
        elif action == config.ACTIONS.right: 
            if self.avatar2.position.x >= config.STATE_WIDTH - 1:
                return
            else:
                new_position = Position(
                    y = self.avatar2.position.y,
                    x = self.avatar2.position.x + 1
                )
                
        else:
            raise Exception(f"{action} is not a valid action.")
            
        self.avatar2 = Sprite(
                name=self.avatar2.name,
                id=self.avatar2.id,
                position=new_position
            )

    def _overlap(self, sprite_1, sprite_2):
        if (sprite_1.position.x == sprite_2.position.x and
           sprite_1.position.y == sprite_2.position.y):
            return True
        else:
            return False

    def step(self, actions):
        self._move_avatar1(actions[0])
        self._move_avatar2(actions[1])
        obs = self._get_obs()
        if self._overlap(self.avatar1, self.enemy) or self._overlap(self.avatar2, self.enemy):
            return obs, -1.0, True, None  # state, reward, done, info
        
        if self._overlap(self.avatar1, self.message1):
            if self._overlap(self.avatar2, self.message2):
                return obs, 1.0, True, None
            else:
                return obs, 0.5, True, None

        if self._overlap(self.avatar2, self.message1):
            if self._overlap(self.avatar1, self.message2):
                return obs, 1.0, True, None
            else:
                return obs, 0.5, True, None

        if self._overlap(self.avatar1, self.message2):
            if self._overlap(self.avatar2, self.message1):
                return obs, 1.0, True, None
            else:
                return obs, 0.5, True, None

        if self._overlap(self.avatar2, self.message2):
            if self._overlap(self.avatar1, self.message1):
                return obs, 1.0, True, None
            else:
                return obs, 0.5, True, None

        if self._overlap(self.avatar1, self.goal1):
            if self._overlap(self.avatar2, self.goal2):
                return obs, 1.0, True, None
            else:
                return obs, 0.5, True, None
        
        if self._overlap(self.avatar2, self.goal1):
            if self._overlap(self.avatar1, self.goal2):
                return obs, 1.0, True, None
            else:
                return obs, 0.5, True, None

        if self._overlap(self.avatar1, self.goal2):
            if self._overlap(self.avatar2, self.goal1):
                return obs, 1.0, True, None
            else:
                return obs, 0.5, True, None
        
        if self._overlap(self.avatar2, self.goal2):
            if self._overlap(self.avatar1, self.goal1):
                return obs, 1.0, True, None
            else:
                return obs, 0.5, True, None

        return obs, 0.0, False, None
