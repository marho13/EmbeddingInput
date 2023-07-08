from embedding import get_embedding
from ppo import PPO
import gym
import wandb

class train_network:
    def __init__(self, num_epochs, timesteps, hp, env_name):
        self.epochs = num_epochs
        self.timesteps = timesteps
        self.param = hp
        self.model = PPO(hp["state_dim"],hp["action_dim"], hp["lr_act"], hp["lr_crit"], hp["gamma"], hp["epochs"],
                         hp["clip"], hp["continuous"])
        self.env = gym.make(env_name)
        self.logger = wandb.init(project="classifier_embedding_{}".format(env_name), group="")

    def iterate_n_epochs(self):
        for e in range(self.epochs):
            rew = self.run_one_episode()
            loss = self.model.update()

            self.log_stats(rew, loss, e)

    def run_one_episode(self):
        env = self.env
        steps = 0
        state, info = env.reset()
        done = False
        episode_rew = 0.0

        while steps > self.timesteps and not done:
            state, rew, done, doney, info = self.training_loop(state, env)
            episode_rew += rew

        return episode_rew

    def training_loop(self, image, env):
        emb = get_embedding(image)
        action = self.model.select_action(emb)
        return env.step(action)

    def log_stats(self, rew, loss, e):
        self.logger.log({"reward": rew}, step=e)
        self.logger.log({"loss": loss}, step=e)


#Hyperparameters
state_dim = 1000
action_dim = 3 #Chosen by the environment

lr_actor = 0.00003
lr_critic = 0.0001

gamma = 0.99
clip = 0.2

ppo_epochs = 10
continuous_action_space = True

hyper_parameters = {"state_dim":state_dim, "action_dim":action_dim, "lr_act": lr_actor, "lr_crit": lr_critic,
                    "gamma":gamma, "epochs":ppo_epochs, "clip":clip, "continuous":continuous_action_space}

#Env hyperparameters
epochs = 30000
timesteps = 1000
env_name = "CarRacing-v2"

train_network(num_epochs=epochs, timesteps=timesteps, hp=hyper_parameters, env_name=env_name)
