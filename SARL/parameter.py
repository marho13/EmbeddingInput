import ray
import torch
import numpy as np
from PPO import PPO
from model import small_size
from torchsummary import summary
#from resnetRL import resnet18
#from RLResnetModel import resnet18

class ParameterServer(object):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, has_continuous_action_space, action_std, numAgents, netsize, a=0):
        self.device = torch.device('cuda')
        #print(self.device)
        self.model = netsize(state_dim, action_dim, has_continuous_action_space, action_std).to(self.device)#small_size(state_dim, action_dim, has_continuous_action_space, action_std).to(self.device)#, non_blocking=True)#PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, has_continuous_action_space, action_std, netsize)#resnet18(pretrained=False).to(self.device)#small_size(state_dim, action_dim, has_continuous_action_space, action_std).to(self.device)#, non_blocking=True)#PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, has_continuous_action_space, action_std, netsize)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_actor)
        self.num_agents = numAgents
        #self.model.policy.to(self.device)
        #self.model.policy_old.to(self.device)
        #self.model.get_gradients()

    def getActorGrad(self, grad):  # My idea could decrease the number in np.divide
        avg_grad = self.getAvgGrad(*grad)#avg_grad = self.calcAvgGrad(*grad)
        output = []
        for g in grad:
            temp = np.add(g, avg_grad)#avg_grad)
            output.append(temp)

        return output

    def avgWeights(self, weight):
        sumWeights = {}
        for w in weight[0]:
            sumWeights.update({w:np.zeros(list(weight[0][w].size()))})
        for w in weight:
            for x in w:
                sumWeights[x] += w[x].numpy()


        for w in weight[0]:
            sumWeights[w] = torch.from_numpy(sumWeights[w]/self.num_agents)

        self.set_weights(sumWeights)


    def getActorGradSum(self, grad):
        sum_grad = self.getSumGrad(*grad)
        output = []
        for g in grad:
            temp = np.add(g, sum_grad)
            output.append(temp)

        return output

    def listCalcAvgGrad(self, gradients): #(64,4), (64, ), (2,64), (2, ), (64, 4), (64, ), (1, 64), (1, )
        nnSize = [[64, 1000], [64], [3, 64], [3], [64, 1000], [64], [1, 64], [1]]
        output = []
        for g in range(len(gradients[0])):
            output.append([])
            if len(nnSize[g]) == 2:
                for i in range(nnSize[g][0]):
                    output[-1].append([])
                    for j in range(nnSize[g][1]):
                        tempVal = 0.0
                        for agent in range(len(gradients)):
                            tempVal += gradients[agent][g][i][j]
                        output[-1][-1].append(tempVal)
            else:
                for i in range(nnSize[g][0]):
                    tempVal = 0.0
                    for agent in range(len(gradients)):
                        tempVal += gradients[agent][g][i]
                    output[-1].append(tempVal)
        self.updater(output)
        return self.model.get_weights()

    def rewardweighted(self, grad, rew):
        output = self.getWeightedGrads(grad, rew)
        self.listCalcAvgGrad(output)#self.calcAvgGrad(*output)
        return self.model.get_weights()
    
    def lossweighted(self, grad, loss):
        output = self.getLossWeightedGrads(grad, loss)
        self.calcGrad(*output)#self.listCalcAvgGrad(output)#self.calcAvgGrad(*output)
        return self.model.get_weights()

    def rewardUpscaledweighted(self, grad, rew):
        output = self.getWeightedGrads(grad, rew)
        self.upscaleGrad(*output)
        return self.model.get_weights()

    def getLossWeightedGrads(self, grad, l):
        totLoss = self.getTotRew(l)
        output = []
        for g in range(len(grad)):
            weight = (1/8) + (l[g]/totLoss)
            output.append([])
            for x in range(len(grad[g])):
                output[-1].append((grad[g][x]*weight))

        return output

    def getWeightedGrads(self, grad, rew):
        reward, miny = self.MoveRewards(rew)
        totRew = self.getTotRew(reward)
        output = []
        for g in range(len(grad)):
            if reward[g] == 0.0:
                weight = (1/8) + (1/totRew)
            else:
                weight = (1/8) + reward[g]/totRew
            output.append([])
            for x in range(len(grad[g])):
                output[g].append((grad[g][x] * weight))
        return output

    def MoveRewards(self, rew):
        minimum = min(rew)
        output = []
        for r in rew:
            output.append(r+minimum)
        return output, minimum

    def getTotRew(self, rew):
        sum = 0
        for r in rew:
            sum += r
        return sum*2
    
    def upscaleGrad(self, *grad):
        summed_gradients = [
                np.stack(grad_zip).sum(axis=0)
                for grad_zip in zip(*grad)
                ]
        summed_gradients = np.multiply(summed_gradients, (self.num_agents/2))
        self.updater(summed_gradients)
        return self.model.get_weights()
        
    
    def calcAvgGrad(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        summed_gradients = [s/self.num_agents for s in summed_gradients]
        self.updater(summed_gradients)
        return self.model.get_weights()

    def calcGrad(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.updater(summed_gradients)
        return self.model.get_weights()
    
    def getSumGrad(self, *grad):
        sum_grad = [
                np.stack(grad_zip).sum(axis=0)
                for grad_zip in zip(*grad)
                ]
        return sum_grad

    def getAvgGrad(self, *gradients):
        sum_grad = [
                np.stack(grad_zip).sum(axis=0)
                for grad_zip in zip(*gradients)
                ]
        return np.divide(sum_grad, self.num_agents)

    def updater(self, gradient):
        #self.model.optimizer.zero_grad()
        #summary(self.model, (1000, 1))
        #summary(small_size, (3, 96, 9))
        self.model.set_gradients(gradient)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def returnModel(self):
        return self.model
