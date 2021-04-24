import torch
import time
import random
import numpy as np
from PIL import Image
from nn import Net, Trainer
from collections import deque
from grab_screen import screenshot
import torchvision.transforms as T
from game_engine import SnakeGame, Point

class Agent():
    def __init__(self, memory_capacity=0, lr=0):
        self.epsilon = 1
        self.e_decay = 0.995
        self.e_min = 0.01
        self.gamma = 0.99
        self.memory = deque(maxlen=memory_capacity)
        # self.net = Net(15, 256, 3)
        self.net = Net(3)
        self.trainer = Trainer(self.net, lr, self.gamma)
        # self.transform = T.ToTensor()

    def get_state(self, game):
        # super slow
        state = screenshot("Snake") # image is the state
        state = Image.fromarray(np.uint8(state)).convert("RGB")
        state = state.resize((224, 224))
        state = np.array(state).transpose(2, 0, 1)
        # state = self.transform(state)

        # head = game.snake[0]
        # tail = game.snake[-1]

        # point_l = Point(head.x - 20, head.y)
        # point_r = Point(head.x + 20, head.y)
        # point_u = Point(head.x, head.y - 20)
        # point_d = Point(head.x, head.y + 20)

        # dir_l = game.direction == "l"
        # dir_r = game.direction == "r"
        # dir_u = game.direction == "u"
        # dir_d = game.direction == "d"

        # state = [
        #     (dir_r and game.is_collision(point_r)) or
        #     (dir_l and game.is_collision(point_l)) or
        #     (dir_u and game.is_collision(point_u)) or
        #     (dir_d and game.is_collision(point_d)),

        #     (dir_u and game.is_collision(point_r)) or
        #     (dir_d and game.is_collision(point_l)) or
        #     (dir_l and game.is_collision(point_u)) or
        #     (dir_r and game.is_collision(point_d)),

        #     (dir_d and game.is_collision(point_r)) or
        #     (dir_u and game.is_collision(point_l)) or
        #     (dir_r and game.is_collision(point_u)) or
        #     (dir_l and game.is_collision(point_d)),

        #     dir_l,
        #     dir_r,
        #     dir_u,
        #     dir_d,

        #     game.food.x < game.head.x,
        #     game.food.x > game.head.x,
        #     game.food.y < game.head.y,
        #     game.food.y > game.head.y,

        #     tail.x < game.head.x,
        #     tail.x > game.head.x,
        #     tail.y < game.head.y,
        #     tail.y > game.head.y,
        # ]

        # state = np.array(state, dtype=int)
        # print(state.shape)
        return state

    def get_action(self, state):
        action = [0, 0, 0]
        if random.random() < self.epsilon:
            a = random.randint(0, 2)
            action[a] = 1
        else:
            with torch.no_grad():
                # use transforms and unsqueeze
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                # state = self.transform(state).unsqueeze(0)

                preds = self.net(state)
                a = torch.argmax(preds).item()
                action[a] = 1

        self.epsilon = max(self.epsilon * self.e_decay, self.e_min)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self, batch_size):
        if len(self.memory) > batch_size:
            mem_sample = random.sample(self.memory, batch_size)
        else:
            mem_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mem_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)