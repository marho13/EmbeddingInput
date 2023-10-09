# from temp.model import ConvNet
from EmbeddingInput.Udacity.ppo import PPO

# from temp.memory import Memory
import torch
from EmbeddingInput.Udacity.embedder import get_embedding
import EmbeddingInput.Udacity.environment as environment

from EmbeddingInput.Udacity.model import small_size

import numpy as np
# from EmbeddingInput.Udacity.parameter import ParameterServer
# from EmbeddingInput.Udacity.worker import DataWorker
import ray
import wandb
import gym
from datetime import datetime
import time


def selfContainedmain():  # Fix the Ros nodes (2 )

    wandy = wandb.init(project="FFI",
                       config={
                           "batch_size": 16,
                           "learning_rate": 0.0003,
                           "dataset": "FFI",
                           "model": "Highest reward single actor weight",
                       })
    env = environment.giveEnv(
        "G:/UdacityML/self_driving_car_nanodegree_program.exe")  # /Downloads/BoatSimulatorBuild-main/LinuxBoatBuild.x86_64")#, allow_multiple_obs=True) #
    # image = env.observation_space.spaces[0]
    state_dim = 1
    maxRew = 0.0
    for n in env.observation_space.spaces[0].shape:
        state_dim *= n
    # state_dim = [state_dim*n for n in (env.observation_space.spaces[0].shape)]#*env.observation_space.shape[1]*env.observation_space.shape[2]#8 #164x164x3
    action_dim = 3
    lr_act = 0.0012
    lr_crit = 0.004
    # betas = (0.9, 0.99)
    gamma = 0.99
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PPO(state_dim, action_dim, lr_act, lr_crit, gamma, K_epochs=10, eps_clip=0.2,
                has_continuous_action_space=True)
    # memory = Memory(2000)

    for episode in range(10000):
        env.reset()
        time.sleep(1)
        state = env.reset()
        time.sleep(1)
        # env.step([0.0,0.0,0.0])
        img = np.ravel(np.array(state[0], dtype=np.float16))
        imugnss = np.array(state[1], dtype=np.float16)
        totRew = 0.0

        for step in range(500):
            prevState = torch.from_numpy(img.copy())  # img.copy())
            state, rew, done = training_loop(img, env)

            totRew += rew
            # print(type(done.item()), done.item())
            if done:
                print("Episode {}, gave reward: {:2f}".format(episode, totRew.item()))
                wandy.log({"Reward": totRew.item()})
                if totRew.item() > maxRew:
                    maxRew = totRew.item()
                    torch.save(model, "FFI_model")
                # env.reset()
                model.update()
                # memory = Memory(2000)
                break


class TrainNetwork:
    def __init__(self, num_epochs, ts, hp, env):
        self.epochs = num_epochs
        self.num_runs = 10
        self.maxSteps = ts
        self.param = hp
        self.k_epoch = hp["epochs"]
        self.model = PPO(hp["state_dim"], hp["action_dim"], hp["lr_act"], hp["lr_crit"], hp["gamma"], hp["epochs"],
                         hp["clip"], hp["continuous"], 0.6, small_size, 0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = environment.giveEnv("/home/martin_holen/Downloads/LinuxBuild/LinuxBoatBuild.x86_64")
        self.logger = wandb.init(project="classifier_embedding_{}".format(env), group="embedding_mid")
        self.has_continuous_action_space = False
        self.env_name = env
        self.runInitializations(hp["state_dim"], hp["action_dim"], hp["lr_act"], hp["lr_crit"], hp["gamma"],
                                hp["epochs"], hp["clip"], 0.6, env, 8, small_size,
                                "small_size")  # self.iterate_n_epochs()

    def iterate_n_epochs(self):
        rewards = []
        print("Starting to train")
        for e in range(self.epochs):
            # print("Episode")
            rew, loss = self.train_episode()  # self.run_n_agent_episode()
            # loss = self.model.update()

            rewards.append(rew)
            print("Episode {}, gave reward: {} & avg {}".format(e, rew, sum(rewards[-50:]) / 50))
            self.log_stats(rew, loss, e)

    def initialiseRun(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std, env_name,
                      num_agents, netsize):
        self.episode = 0
        self.time_step = 0

    def train_episode(self):
        episodeRew = 0.0
        for e in range(2):
            done = False
            state = self.resetter()
            while not done:
                state, reward, done = self.training_loop(state, self.env)

                episodeRew += reward
        loss = self.model.update()
        return episodeRew / 16, loss

    def runInitializations(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std,
                           env_name, num_agents, netsize, netstring):
        for a in range(self.num_runs):
            self.logger = wandb.init(project="classifier_embedding_{}".format(self.env_name),
                                     group="embedding_mid")  # resnet18")
            print("Run {}:".format(a))
            self.initialiseRun(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std,
                               env_name, num_agents, netsize)
            self.iterate_n_epochs()
            self.logger.finish()
        self.logger.finish()

    def training_loop(self, image, env):
        # print(len(image), len(image[0]), len(image[0][0]), len(image[0][0][0]))
        emb = get_embedding(image)
        action = self.model.select_action(
            emb.to(self.device))  # torch.from_numpy(image).float().to('cuda'))#emb.to('cuda'))
        # print(action)#action = torch.argmax(action)
        state, rew, done, _ = env.step(action)
        # print(rew, done, _)
        self.model.save_rew_terminal(rew, done)
        # print(len(state), len(state[0]), len(state[0][0]), len(state[0][0][0]))
        # print(state)
        state = np.float32(np.swapaxes(np.expand_dims(state[0], axis=0), 1, -1)) / 256.0
        return state, rew, done

    def log_stats(self, rew, loss, e):
        self.logger.log({"reward": rew}, step=e)
        self.logger.log({"loss": loss}, step=e)

    def resetter(self):
        state, _ = self.env.reset()
        return np.float32(np.swapaxes(np.expand_dims(state, axis=0), 1, -1))


def main():
    envName = "/home/martin_holen/Document/linuxBuild/LinuxBoatBuild.x86_64"
    env = environment.giveEnv(envName)  # , allow_multiple_obs=True #

    image = env.observation_space.spaces[0]

    state_dim = image.shape[0] * image.shape[1] * image.shape[2]  # 8 #164x164x3
    action_dim = 3
    n_latent_var = 64
    lr = 0.001
    betas = (0.9, 0.99)
    gamma = 0.99

    device = "cpu"
    numAgents = 4
    epochs = 1

    ray.init(ignore_reinit_error=True)

    print("Running synchronous parameter server training.")
    #
    wandy = wandb.init(project="Federated-learning-PPO5L-{}-SGD".format(envName),
                       config={
                           "batch_size": 16,
                           "learning_rate": lr,
                           "dataset": envName,
                           "model": "Highest reward single actor weight",
                       })

    for run in range(10):
        ps = ParameterServer(state_dim, action_dim, n_latent_var, lr, betas, gamma, envName, numAgents)
        workers = [DataWorker.remote(state_dim, action_dim, n_latent_var, lr, betas, gamma, envName) for i in
                   range(numAgents)]

        current_weights = ps.get_weights()
        print("Run {}".format(run))
        for i in range(epochs):
            gradients = ray.get([worker.compute_gradients.remote(current_weights) for worker in workers])
            grads, loss, reward = [], [], []
            for output in gradients:
                grads.append(output[0])
                loss.append(output[1].item())
                reward.append(output[2].item())

            avgLoss = sum(loss) / numAgents
            avgRew = sum(reward) / numAgents
            wandb.log({"training loss": avgLoss}, step=i)
            wandb.log({"training reward": avgRew}, step=i)

            current_weights = ps.highestRewGrad(grads, reward)

            if i % 10 == 9:
                rew = ps.performNActions(1000)
                print("Epoch {}, gave reward {}".format(i, rew.item()))
                wandb.log({"testing reward": rew.item()}, step=i)

    wandy.finish()

    ray.shutdown()


class runner:
    def __init__(self, env_name, max_ep_len, has_continuous_action_space, k_epochs, eps_clip, gamma, lr_actor,
                 lr_critic, action_std, action_std_decay_rate, min_action_std, action_std_decay_freq, log_f_name):

        random_seed = 0  # set random seed if required (0 = no random seed)
        self.has_continuous_action_space = has_continuous_action_space
        # env = environment.giveEnv("/home/martin_holen/LinuxBuild/LinuxBoatBuild.x86_64")

        num_agents = 1
        num_runs = 10000

        if random_seed:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", random_seed)
            torch.manual_seed(random_seed)
            # env.seed(random_seed)
            np.random.seed(random_seed)

        print("training environment name : " + env_name)

        # state_dim = env.observation_space.shape[0]

        # if self.has_continuous_action_space:
        #     action_dim = env.action_space.shape[0]
        # else:
        #     action_dim = env.action_space.n
        action_dim = 4
        state_dim = 500 * 500 * 1
        directory = "PPO_preTrained/"

        self.ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                             self.has_continuous_action_space, action_std)
        # self.env = env#gym.make(env_name)

        self.time_step = 0
        self.episode = 0
        self.run_num_pretrained = 0

        self.max_ep_len = max_ep_len
        self.update_timestep = self.max_ep_len * 4

        self.action_std = action_std
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        self.action_std_decay_freq = action_std_decay_freq

        self.save_model_freq = int(1e5)
        self.checkpoint_path = directory + "{}/PPO_{}_{}_{}.pth".format(env_name, env_name, random_seed,
                                                                        self.run_num_pretrained)
        self.start_time = datetime.now().replace(microsecond=0)

        self.log_f = open(log_f_name, "w+")
        self.log_f.write('episode,timestep,reward\n')
        self.k_epoch = k_epochs
        self.episode_rew = []
        self.print_freq = 5
        self.num_runs = num_runs

        self.wandy = wandb.init(project=env_name,
                                config={
                                    "batch_size": 16,
                                    "learning_rate_critic": lr_critic,
                                    "learning_rate_actor": lr_actor,
                                    "dataset": env_name,
                                    "model": "standard",
                                })

        self.runInitializations(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std,
                                env_name, num_agents)

    def initialiseRun(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std,
                      env_name, num_agents):
        print(env_name + "Damn")
        self.worker = [DataWorker.remote(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                                         self.has_continuous_action_space, action_std, env_name) for _ in
                       range(num_agents)]

        self.ps = ParameterServer(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                                  self.has_continuous_action_space, action_std, num_agents)

    def runInitializations(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std,
                           env_name, num_agents):
        for _ in range(self.num_runs):
            self.initialiseRun(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std,
                               env_name, num_agents)
            self.workerMax(maxTi=int(3e6))
        self.wandy.finish()

    def workerRun(self):
        current_weights = self.ps.get_weights()
        r = ray.get([w.gatherReplay.remote() for w in self.worker])  # ray.get(self.worker.gatherReplay())#
        rew = [a.item() for a in r]
        loss = []
        for _ in range(self.k_epoch):
            output = ray.get([w.getLoss.remote(current_weights) for w in
                              self.worker])  # ray.get(self.worker.getLoss.remote(current_weights))#
            grad = []
            for a in range(len(output)):
                grad.append(output[a][0])
                loss.append(output[a][1].item())
            current_weights = self.ps.calcAvgGrad(*grad)
            # current_weights = self.ps.rewardweighted(grad, rew)#self.ps.appGrad(g)#
        [w.memClear.remote() for w in self.worker]  # self.worker.memClear()#.remote()
        [w.policyOldUp.remote() for w in self.worker]  # self.worker.policyOldUp()
        return (sum(rew) / len(rew)), (sum(loss) / len(loss))

    def workerMax(self, maxEp=1e11, maxTi=1e14):
        rew, loss = [], []
        while True:
            if self.episode > maxEp or self.time_step > maxTi:
                self.log_f.close()
                # self.env.close()
                # print total training time
                print(
                    "============================================================================================")
                end_time = datetime.now().replace(microsecond=0)
                print("Started training at (GMT) : ", self.start_time)
                print("Finished training at (GMT) : ", end_time)
                print("Total training time  : ", end_time - self.start_time)
                print(
                    "============================================================================================")
            else:
                r, l = self.workerRun()
                self.wandy.log({"Reward": r}, step=self.episode)
                self.wandy.log({"Loss": l}, step=self.episode)
                loss.append(l)
                rew.append(r)
                self.episode += 1
                if self.episode % self.print_freq == 0:
                    avg = sum(rew[-self.print_freq:]) / len(rew[-self.print_freq:])
                    # print(len(rew), rew[-100:], avg)
                    print("Episode {} gave reward {}".format(self.episode, avg))


if __name__ == '__main__':
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
    epochs = 10000
    timesteps = 500

    TrainNetwork(num_epochs=epochs, ts=timesteps, hp=hyper_parameters, env="Boat")
    # selfContainedmain()
