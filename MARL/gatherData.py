import numpy as np
import gym
from matplotlib.image import imsave

def saveImage(data, num):
    imsave("file-{}.jpg".format(num), data)


def resetter(env):
    img = env.reset()
    for _ in range(40):
        img = env.step([0.0, 0.0, 0.0])
    return img
env = gym.make("CarRacing-v2")
img = resetter(env)

saveImage(img[0], 0)
for _ in range(20):
    img = env.step(np.array([0.5, 0.5, 0.0]))

saveImage(img[0], 1)

