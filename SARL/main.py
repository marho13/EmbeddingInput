from embedding import get_embedding
from ppo import PPO
import gym
import wandb
import numpy as np
from model import small_size
#from resnet import resnet18
from RLResnetModel import resnet18
from worker import DataWorker
from parameter import ParameterServer
import ray

class TrainNetwork:
    def __init__(self, num_epochs, ts, hp, env):
        self.epochs = num_epochs
        self.num_runs = 10
        self.maxSteps = ts
        self.param = hp
        self.k_epoch = hp["epochs"]
        self.model = PPO(hp["state_dim"], hp["action_dim"], hp["lr_act"], hp["lr_crit"], hp["gamma"], hp["epochs"],
                         hp["clip"], hp["continuous"], 0.6, resnet18, 0)
        self.env = gym.make(env)
        self.logger = wandb.init(project="classifier_embedding_{}".format(env), group="resnet_transfer")
        self.has_continuous_action_space = True
        self.env_name = env
        self.runInitializations(hp["state_dim"], hp["action_dim"], hp["lr_act"], hp["lr_crit"], hp["gamma"], hp["epochs"], hp["clip"], 0.6, env, 8, resnet18, "resnet")#self.iterate_n_epochs()

    def iterate_n_epochs(self):
        rewards = []
        for e in range(self.epochs):
            rew, loss = self.run_n_agent_episode()
            #loss = self.model.update()
            
            rewards.append(rew)
            print("Episode {}, gave reward: {} & avg {}".format(e, rew, sum(rewards[-50:])/50))
            self.log_stats(rew, loss, e)

    def initialiseRun(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std, env_name, num_agents, netsize):
        self.worker = [DataWorker.remote(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                                         self.has_continuous_action_space, action_std, env_name, netsize, a) for a in range(
            num_agents)]  # DataWorker.remote(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, self.has_continuous_action_space, action_std, env_name)
        self.ps = ParameterServer(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                                  self.has_continuous_action_space, action_std, num_agents, netsize)
        self.episode = 0
        self.time_step = 0

    
    def runInitializations(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std, env_name, num_agents, netsize, netstring):
        for a in range(self.num_runs):
            self.logger = wandb.init(project="classifier_embedding_{}".format(self.env_name), group="resnet_transfer")#resnet18")
            print("Run {}:".format(a))
            self.initialiseRun(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std, env_name, num_agents, netsize)
            self.iterate_n_epochs()
            self.logger.finish()
        self.logger.finish()



    def run_n_agent_episode(self):
        current_weights = self.ps.get_weights()
        r = ray.get([w.gatherReplay.remote(current_weights) for w in self.worker])#ray.get(self.worker.gatherReplay())#
        #print(r)
        rew = [a for a in r]
        #loss = []
        for _ in range(self.k_epoch):
            output = ray.get([w.getLoss.remote(current_weights) for w in self.worker])#ray.get(self.worker.getLoss.remote(current_weights))#
            grad, loss = [], []
            for a in range(len(output)):
                grad.append(output[a][0])#.detach())
                loss.append(output[a][1].item())
            #current_weights = self.ps.calcAvgGrad(*grad)
            current_weights = self.ps.lossweighted(grad, loss)#self.ps.appGrad(g)#
        [w.memClear.remote() for w in self.worker]#self.worker.memClear()#.remote()
        [w.policyOldUp.remote() for w in self.worker]#self.worker.policyOldUp()
        return (sum(rew)/len(rew)), (sum(loss)/len(loss))

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
        
        return episode_rew

    def training_loop(self, image, env):
        emb = get_embedding(image)
        action = self.model.select_action(emb)
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
        

###Hyperparameters###
state_dim = 1000
action_dim = 3  # Chosen by the environment

lr_actor = 0.00003
lr_critic = 0.0001

gamma = 0.99
clip = 0.2

ppo_epochs = 10
continuous_action_space = True

hyper_parameters = {"state_dim": state_dim, "action_dim": action_dim, "lr_act": lr_actor, "lr_crit": lr_critic,
                    "gamma": gamma, "epochs": ppo_epochs, "clip": clip, "continuous": continuous_action_space}

###Env hyperparameters###
epochs = 300000
timesteps = 1000
env_name = "CarRacing-v2"

TrainNetwork(num_epochs=epochs, ts=timesteps, hp=hyper_parameters, env=env_name)
