import gym
from gym import wrappers
env = gym.make('Copy-v0')
env = wrappers.Monitor(env, './copy-v0', force=True)
for i_episode in range(600):
    observation = env.reset()
    done=False
    while not done:
    #for t in range(100):
        action = (1,1,observation)
        observation, reward, done, info = env.step(action)
     
     #       print("Episode finished after {} timesteps".format(t+1))
     #       break
    #env.render()
    #raw_input()
