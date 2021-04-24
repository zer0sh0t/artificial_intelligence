import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size)

        self.a_fc = nn.Linear(hidden_size, output_size)
        self.v_fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        a = self.a_fc(x)
        v = self.v_fc(x)

        return a + v - a.mean()


class Trainer():
    def __init__(self, net, lr, gamma):
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(self.net.parameters(), lr)

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        Q_preds = self.net(state)
        target_preds = torch.zeros_like(Q_preds)
        net_preds = torch.zeros_like(Q_preds)

        with torch.no_grad():
            for i in range(len(done)):
                Q = reward[i]
                if not done[i]:
                    Q = reward[i] + self.gamma * \
                        torch.max(self.net(next_state[i]))

                net_preds[i][torch.argmax(action[i]).item()] = Q

        for i in range(len(done)):
            Q = reward[i]
            if not done[i]:
                Q = reward[i] + self.gamma * \
                    torch.max(self.target_net(next_state[i]))

            target_preds[i][torch.argmax(action[i]).item()] = Q

        action_idx = action.max(1)[1].unsqueeze(1)
        Q_vals = Q_preds.gather(1, action_idx)

        net_actions = net_preds.max(1)[1].unsqueeze(1)
        target_vals = target_preds.gather(1, net_actions)

        self.opt.zero_grad()
        loss = F.smooth_l1_loss(Q_vals, target_vals.detach())
        loss.backward()

        for p in self.net.parameters():
            p.grad.data.clamp_(-1, 1)

        self.opt.step()
