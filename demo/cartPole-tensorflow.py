import numpy as np
import tensorflow as tf
import gyn
env = gym.make('CartPole-0')
env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes<10:
    env.render()
    obervation,reward,done,_ = env.step(np.random.randint(0,2))
    reward_sum += reward
    if done:
        random_episodes += 1
        print('reward for this episode was:',reward_sum)
        reward_sum = 0
        env.reset()
