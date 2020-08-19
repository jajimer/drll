import torch as T
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gym
import pandas as pd
plt.style.use('seaborn')


class PolicyNetwork(nn.Module):
    def __init__(self, s_dim, n_actions, lr):
        super(PolicyNetwork, self).__init__()
        self.input_dim = s_dim
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr)
        self.device = 'cuda:0' if T.cuda.is_available() else 'cpu'
        self.net.to(self.device)

    def forward(self, obs):
        probs = self.net(obs)
        dist = T.distributions.Categorical(probs)
        return dist


class Agent(object):
    def __init__(self, input_dim, n_actions, lr = 3e-4, gamma = 0.99, normalize = True):
        self.rewards_memory = []
        self.log_actions_memory = []
        self.gamma = gamma
        self.normalize = normalize

        self.policy = PolicyNetwork(input_dim, n_actions, lr)

    def get_action(self, obs):
        x = T.Tensor([obs]).to(self.policy.device)
        dist = self.policy(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.log_actions_memory.append(log_prob)

        return action.item()

    
    def store_rewards(self, r):
        self.rewards_memory.append(r)
    
    def learn(self):

        def get_returns(rewards, gamma):
            G = np.zeros_like(rewards, dtype=np.float64)
            for t in range(len(rewards)):
                G_sum = 0
                discount = 1
                for k in range(t, len(rewards)):
                    G_sum += rewards[k] * discount
                    discount *= gamma
                G[t] = G_sum
            return G

        self.policy.optimizer.zero_grad()
        G = get_returns(self.rewards_memory, self.gamma)
        if self.normalize:
            G = (G - G.mean()) / (G.std() + 1e-9)
        gradients = [-log * g for log, g in zip(self.log_actions_memory, G)]
        loss = T.stack(gradients).sum().to(self.policy.device)
        loss.backward()
        self.policy.optimizer.step()

        self.rewards_memory = []
        self.log_actions_memory = []



if __name__ == '__main__':

    def run(env_name, lr, gamma, normalize, N = 2000):
        env = gym.make(env_name)
        agent = Agent(gamma=gamma, lr=lr, 
                                input_dim=env.observation_space.shape[0],
                                n_actions=env.action_space.n,
                                normalize = normalize)

        scores = []
        for i in range(N):
            done = False
            observation = env.reset()
            score = 0
            while not done:
                action = agent.get_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.store_rewards(reward)
                observation = observation_
            agent.learn()
            scores.append(score)

            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

        return scores


    for envs in ['CartPole-v0', 'LunarLander-v2']:
        N = 2000
        score_norm = run(envs, 3e-4, 0.9, True, N)
        score_no_norm = run(envs, 3e-4, 0.9, False, N)

        window = 20
        mean_score_norm = [elem for elem in pd.Series.rolling(pd.Series(score_norm), window).mean()]
        mean_score_no_norm = [elem for elem in pd.Series.rolling(pd.Series(score_no_norm), window).mean()]

        plt.plot(mean_score_norm, label = 'Normalized')
        plt.plot(mean_score_no_norm, label = 'Unnormalized')
        plt.xlabel('Episode')
        plt.ylabel('Avg. score')
        plt.legend()
        name = envs.split('-')[0]
        plt.savefig('reinforce_%s.png' % name)
        plt.close()