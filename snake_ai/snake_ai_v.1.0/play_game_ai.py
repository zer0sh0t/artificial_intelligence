import torch
from agent import Agent
from game_engine import SnakeGame

speed = 40
agent = Agent()
game = SnakeGame(800, 800)
pretrained_net = torch.load("assets/checkpoint-700.pth.tar")
agent.net.load_state_dict(pretrained_net)

ep_reward = 0
done = False
while not done:
    state = agent.get_state(game)

    action = [0, 0, 0]
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        preds = agent.net(state)
        a = torch.argmax(preds).item()
        action[a] = 1

    reward, done = game.play(action, speed, 0)
    next_state = agent.get_state(game)
    ep_reward += reward

print(f"ep_reward: {ep_reward} | ep_score: {game.score}")
