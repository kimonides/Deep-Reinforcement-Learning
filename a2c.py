import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax()
        )

    def forward(self, x):
        x = x.view(1, -1)
        return self.fc(x)


class Baseline(nn.Module):

    def __init__(self, input_shape):
        super(Baseline, self).__init__()
        shape = 1
        for dim in input_shape:
            shape *= dim
        self.fc = nn.Sequential(
            nn.Linear(shape, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x.view(1, -1)
        return self.fc(x)


class Agent:

    def __init__(self, policy_net, baseline_net):
        self.policy_net = policy_net
        self.baseline = baseline_net

    def train(self, env, num_traj, iterations, gamma, base_epochs):
        iter_rewards = []
        for iter in range(iterations):
            print('Episode')
            trajectories = []
            ITER_REW = 0
            for _ in range(num_traj):
                rewards = []
                log_probs = []
                states = []
                s = env.reset()
                done = False
                while not done:
                    s = torch.FloatTensor([s]).cuda()
                    a = self.policy_net(s)
                    states.append(s)
                    del s
                    a2 = a.detach().cpu().numpy()
                    vec = [0, 1]
                    u = np.random.choice(vec, 1, replace=False, p=a2[0])
                    log_probs.append(a[0][u])
                    del a
                    sp, r, done, _ = env.step(u[0])
                    if done:
                        if len(rewards) < 50:
                            r = -200
                    ITER_REW += r
                    rewards.append(r)
                    # env.render()
                    s = sp
                trajectories.append({'log_probs': log_probs, 'rewards': rewards, 'states': states})
            # self.update_baseline(base_epochs, trajectories, gamma)
            self.update_parameters(trajectories, gamma)
            print("ITERATION:", iter + 1, "AVG REWARD:", ITER_REW / num_traj)
            iter_rewards.append(ITER_REW/num_traj)
        return iter_rewards

    def update_parameters(self, trajectories, gamma):
        c = 0.01
        loss = torch.tensor([0]).float().cuda()
        optim = torch.optim.Adam(list(self.policy_net.parameters()) + list(self.baseline.parameters()), lr=0.01)
        for trajectory in trajectories:
            for t in range(len(trajectory['rewards'])):
                r_t = torch.tensor([0]).float().cuda()
                log_prob = trajectory['log_probs'][t]
                temp = trajectory['rewards'][t:t + 20]
                for i, reward in enumerate(temp[:-1]):
                    r_t += gamma ** i * reward
                critic_estimate = self.baseline(trajectory['states'][t])[0]
                r_t += gamma ** i * self.baseline(trajectory['states'][i+1])[0]
                advantage_t = r_t - critic_estimate
                loss += (-log_prob * advantage_t) + (c * (critic_estimate - r_t) ** 2)
                if t % 20 == 0:
                    optim.zero_grad()
                    loss.backward()
                    # print("\nPolicy Gradients\n")
                    # for name, param in self.policy_net.named_parameters():
                    #     print(name, param.grad.data.sum())
                    # print("\nCritic Gradients\n")
                    # for name, param in self.baseline.named_parameters():
                    #     print(name, param.grad.data.sum())
                    optim.step()
                    loss = torch.tensor([0]).float().cuda()


def main():
    env = gym.make('CartPole-v0')
    policy_net = PolicyNet(env.observation_space.shape, env.action_space.n).to(torch.device('cpu'))
    base_net = Baseline(env.observation_space.shape).to(torch.device('cpu'))
    # policy_net.load_state_dict(torch.load('./policynet'))
    # base_net.load_state_dict(torch.load('./basenet'))
    agent = Agent(policy_net, base_net)
    rews = agent.train(env, 32, 200, 0.99, 5)
    plt.plot(rews)
    plt.show()


main()