import torch
import random
import numpy as np
from agent import Agent
from game_engine import SnakeGame

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

save = False
episodes = 1_001
memory_capacity = 1_000_000
batch_size = 16
lr = 3e-4
print_every = 5#50
update_every = 20
agent = Agent(memory_capacity, lr)
game = SnakeGame()
ep_rewards = []
ep_scores = []
agent.net.train()

for episode in range(episodes):
    if episode % print_every == 0:
        speed = 200
    else:
        speed = 1000

    game.reset()
    ep_reward = 0
    done = False
    while not done:
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, done = game.play(action, speed, episode)
        next_state = agent.get_state(game)
        ep_reward += reward

        agent.train_short_memory(state, action, reward, next_state, done)
        agent.remember(state, action, reward, next_state, done)

    ep_rewards.append(ep_reward)
    ep_scores.append(game.score)

    if episode % update_every == 0:
        agent.trainer.update_target_net()

    if episode % print_every == 0:
        if save:
            torch.save(agent.net.state_dict(), f"checkpoint-{episode}.pth.tar")
        print(f"episode: {episode} | avg_reward: {np.mean(ep_rewards[-print_every:])} | avg_score: {np.mean(ep_scores[-print_every:])}")

    agent.train_long_memory(batch_size)
