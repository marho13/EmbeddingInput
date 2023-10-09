import ray
import time
from ppo import PPO
import gym
from memory import Memory
import copy
import numpy as np
import torch
from embedding import get_embedding

@ray.remote(max_restarts=-1, max_task_retries=2, num_gpus=0.25, memory=16096*1024*1024)
class DataWorker(object):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, has_continuous_action_space, action_std, env_name, netsize, a):
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        #
        # self.seed = seed
        self.device = torch.device('cuda')
        time.sleep(a*5)
        self.model = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, has_continuous_action_space, action_std, netsize, a)#.to(device)
        self.model.policy.to(self.device, non_blocking=True)
        self.model.policy_old.to(self.device, non_blocking=True)
        self.env = gym.make(env_name)
        self.env.reset()#seed=env_seed)
        self.memory = Memory()
        self.maxSteps = 1000

    def gatherReplay(self, weights):
        self.model.policy.set_weights(weights)
        r, t = [], 0
        while t < 2:
            rew, timestep = self.run_one_episode()#self.performNActions(1000)
            r.append(rew)
            t += 1
        return sum(r)/len(r)

    def getLoss(self, weights):
        self.model.policy.set_weights(weights)
        a, b, c, d = self.model.getTrainingMem()
        #loss = self.model.trainKepochs(a, b, c, d)
        loss = self.model.trainKepochsBatch(a, b, c, d)
        return [copy.copy(self.model.policy.get_gradients()), loss]

    def calcGrad(self, weights):
        self.model.set_weights(weights)
        rew = self.performNActions(1000)
        self.model.update()
        return self.model.get_weights(), rew

    def update(self):
        return self.model.update()

    def policyOldUp(self):
        self.model.policyOldUp()

    def run_one_episode(self):
        env = self.env
        steps = 0
        state = self.resetter()
        done = False
        episode_rew = 0.0
        #print(len(state), len(state[0]))
        while (steps < self.maxSteps) and (done == False):
            state, rew, done = self.training_loop(state, env)
            episode_rew += rew
            #if steps % 100 == 0:
            #    print("Steps: {}/{}".format(steps, self.maxSteps))
            #    print("Done: {}".format(done))

            steps += 1

        return episode_rew, steps

    def training_loop(self, image, env):
        #emb = get_embedding(image)
        action = self.model.select_action(torch.from_numpy(image).float().to(self.device))#emb.to(self.device))
        state, rew, done, _, __ = env.step(action)
        self.model.save_rew_terminal(rew, done)
        state = np.double(np.swapaxes(np.expand_dims(state, axis=0), 1, -1))
        return state, rew, done

    def log_stats(self, rew, loss, e):
        self.logger.log({"reward": rew}, step=e)
        self.logger.log({"loss": loss}, step=e)

    def resetter(self):
        state, _ = self.env.reset()
        return np.double(np.swapaxes(np.expand_dims(state, axis=0), 1, -1))


    def performNActions(self, N):
        state, info = self.env.reset()
        # print(state, info)
        state = np.ravel(state)
        totRew = 0.0
        for t in range(N):
            if len(state) == 2: state = state[0]
            # state = np.ravel(state)
            # print(len(state))
            # print(state.shape)
            state = np.reshape(state, [1, 3, 96, 96])
            prevState = torch.from_numpy(state.copy())
            # s = self.numpyToTensor(state)
            action = self.model.select_action(torch.from_numpy(state).float())
            # action[0] = (action[0]*2) - 1
            # print(action)
            action = torch.argmax(action, dim=-1)
            actionTranslated = self.translateAction(action.item())
            state, rew, done, doney, info = self.env.step(actionTranslated) #rew, done, info, _ = self.env.step(action)#.item())#actionTranslated)
            state = np.ravel(state)
            # action = torch.from_numpy(action) #state', 'action', 'next_state', 'reward
            prevState = torch.unsqueeze(prevState, 0)
            stateMem = torch.unsqueeze(torch.from_numpy(state.copy()), 0)
            rew = torch.unsqueeze(torch.tensor(rew), 0)
            done = torch.unsqueeze(torch.tensor(done), 0)
            self.model.buffer.rewards.append(rew)
            self.model.buffer.is_terminals.append(done)
            # self.memory.push(prevState, action, stateMem, rew)
            totRew += rew
            if done or doney:
                # print(t)
                break
                # state = self.env.reset()
        return totRew, t

    def translateAction(self, action):
        translationDictionary = {0:[-1.0, 0.0, 0.0], 1:[ +1.0, 0.0, 0.0 ], 2:[ 0.0, 0.0, 1.0 ], 3:[ 0.0, 1.0, 0.0 ],
                                 4:[0.0, 0.0, 0.0]}#, 5:[1.0, 0.0, 0.0], 6:[0.0, 1.0, 0.0], 7:[0.0, 0.5, 0.0],
                                 #8:[0.0, 0.0, 0.5], 9:[0.0, 0.0, 1.0]}
        return translationDictionary[action]

    def applyGrads(self, grad):
        self.model.optimizer.zero_grad()
        self.model.set_gradients(grad)
        self.model.optimizer.step()

    def memClear(self):
        self.model.clearBuffer()

    def getActionLayer(self):
        state = self.env.reset()
        # s = self.numpyToTensor(state)
        return self.model.policy.act(state)

    def getValueLayer(self):
        state = self.env.reset()
        # s = self.numpyToTensor(state)
        action, dist = self.model.policy.act(state)
        # print(action)
        logprobs, stateval, distentropy, actionprobs, actionlogprobs = self.model.policy.tester(state, action)
        return logprobs, stateval, distentropy, actionprobs, actionlogprobs

    def getWeights(self):
        return self.model.policy.get_weights()

    def setWeights(self, weight):
        self.model.policy.set_weights(weight)

    def numpyToTensor(self, state):
        s = np.expand_dims(state, axis=0)
        s = np.swapaxes(s, 1, -1)
        return torch.from_numpy(s.copy())
