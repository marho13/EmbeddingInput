import torch
from torch import nn
from model import small_size
from memory import *
import copy
#from resnetRL import resnet18
# device = torch.device('cpu')


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, netsize=small_size, a=0):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        device = torch.device("cuda")#:2" if a<4 else "cuda:3")

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = netsize(state_dim, action_dim, has_continuous_action_space, action_std_init).float().to(device)#resnet18(pretrained=False)#netsize(state_dim, action_dim, has_continuous_action_space, action_std_init).float().to(device)#, non_blocking=True)# self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)

        self.device = device

        self.policy_old = netsize(state_dim, action_dim, has_continuous_action_space, action_std_init).float().to(device)#resnet18(pretrained=False)#netsize(state_dim, action_dim, has_continuous_action_space, action_std_init).float().to(device)#, non_blocking=True)#self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        #print(self.device)#state)
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.cuda.FloatTensor(state).to(self.device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state.detach())
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.cuda.FloatTensor(state).to(self.device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state.detach())
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.cpu().item()

    def save_rew_terminal(self, rew, term):
        self.buffer.rewards.append(rew)
        self.buffer.is_terminals.append(term)

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = torch.stack(deltas)
        tempDelta = deltas.clone()
        gaes = copy.deepcopy(tempDelta.detach())
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return gaes, target#torch.vstack(gaes), torch.vstack(target)

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # advantages, target = self.get_gaes(rewards, self.buffer.is_terminals, old_logprobs, logprobs)
            # Finding Surrogate Loss
            #print(len(self.buffer.rewards), len(self.buffer.is_terminals))
            #print(rewards.shape, state_values.shape)
            advantages = rewards - state_values
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            #grad = self.policy.get_gradients()
            #self.optimizer.zero_grad()
            #self.policy.set_gradients(grad)
            # self.policy_old.set_gradients(gradOld)
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return loss.mean()

    def policyOldUp(self):
        self.policy_old.load_state_dict(self.policy.state_dict())

    def getTrainingMem(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        return old_states, old_actions, old_logprobs, rewards
    
    def trainKepochsBatch(self, old_states, old_actions, old_logprobs, rewards):
        batchSize = len(rewards)//32
        
        self.optimizer.zero_grad()
        outputLoss = 0.0
        for b in range(32):
            start = b*batchSize
            stop = (b+1)*batchSize

            batch_old_states = old_states[start:stop]
            batch_old_actions = old_actions[start:stop]
            batch_old_logprobs = old_logprobs[start:stop]
            batch_rewards = rewards[start:stop]
            
            logprobs, state_values, dist_entropy = self.policy.evaluate(batch_old_states, batch_old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - batch_old_logprobs.detach())

            advantages = batch_rewards - state_values
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, batch_rewards) - 0.01 * dist_entropy

            loss.mean().backward()

            outputLoss += loss.mean()

        return outputLoss/32



    def trainKepochs(self, old_states, old_actions, old_logprobs, rewards):
        # Optimize policy for K epochs
        # Evaluating old actions and values
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

        # match state_values tensor dimensions with rewards tensor
        state_values = torch.squeeze(state_values)

        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(logprobs - old_logprobs.detach())

        # Finding Surrogate Loss
        advantages = rewards - state_values
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        # final loss of clipped objective PPO
        loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        return loss.mean()

    def clearBuffer(self):
        self.buffer.clear()


    def getLossGrad(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
        return loss.mean()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def get_weights(self):
        return copy.copy({k: v.data.cpu() for k, v in self.policy.state_dict().items()})

    def set_weights(self, weights):
        self.policy.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in copy.copy(self.policy.parameters()):
            #grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(p.grad.data.numpy())#copy.copy(grad))
        return grads

    def set_gradients(self, gradients):
        #print(type(gradients[0]), type(self.policy.parameters()[0]))
        for g, p in zip(gradients, self.policy.parameters()):
            if g is not None:
                print(type(p.grad), type(g))
                p.grad = torch.cuda.FloatTensor(copy.copy(g))#from_numpy(copy.copy(g)).float()
