#encoding:utf8



# '---o-----T' our environment


import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 10
ACTIONS = ['left','right']
EPSILON = 0.9
ALPHA = 0.1
LAMBDA = 0.9 #衰减度
GAMMA = 0.9    # 奖励递减值
MAX_EPISODES = 20
FRESH_TIME = 0.1

def build_q_table(n_states,actions):
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))),
        columns=actions,

        )
    #print table
    return table

def choose_action(state,q_table):
    state_actions = q_table.iloc[state,:]
    if np.random.uniform()>EPSILON or state_actions.all() == 0:
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name

def get_env_feedback(S,A):
    if A=='right':
        if S==N_STATES -2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S+1
            R = 0
    else:
        R = 0
        if S ==0:
            S_ = S
        else:
            S_ = S-1
    return S_,R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction))
        time.sleep(2)
        print('\r                                ')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction))
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPISODES):     # 回合
        step_counter = 0
        S = 3   # 回合初始位置
        is_terminated = False   # 是否回合结束
        update_env(S, episode, step_counter)    # 环境更新
        while not is_terminated:

            A = choose_action(S, q_table)   # 选行为
            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
            q_predict = q_table.ix[S, A]    # 估算的(状态-行为)值
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #  实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R     #  实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode

            q_table.ix[S, A] += ALPHA * (q_target - q_predict)  #  q_table 更新
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter+1)  # 环境更新

            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
