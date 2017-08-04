import gym
env=['CartPole-v0','MountainCar-v0','MsPacman-v0','Hopper-v1']
env = gym.make(env[2])
env.reset()

for _ in range(100000):
    print _
    env.render()
    env.step(env.action_space.sample()) # take a random action
    #env.reset()



