import gym
from gym import wrappers
import pandas as pd
import numpy as np

EPSILON=0.1
ACTIONS=[x for x in range(6)]

def build_q_table(n_states,actions):
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))),
        columns=actions,
        )
    return table

def choose_action(state,q_table):
    state_actions = q_table.iloc[state,:]
    if np.random.uniform()>EPSILON or state_actions.all() == 0:
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name


if __name__ == '__main__':
    env = gym.make('Taxi-v2')
    env = wrappers.Monitor(env, './Taxi-v2', force=True)
    
    q_table = build_q_table(25,[x for x in range(6)])
    for episode in range(100000):
        observation = env.reset()
        observation = (x//100)*5+(x%10//2)
        done=False
        while not done:
            action = choose_action(observation,q_table)
            observation,reward,done,info = env.step(action)
            observation = (x//100)*5+(x%10//2)
            #env.render()
