"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/3/12 23:47
@File : AssetAllocation.py
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import copy


class Env(ABC):
    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class DiscreteEnv(Env):

    def __init__(self, state_dim, action_dim, transition_P, initial_dis):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transition_P = transition_P  # transition probs, P[s][a] == [(prob, s', r, done), ...]
        # 转移概率矩阵，是个字典，每个key为state，value为state对应action组成的字典，action字典key为action，value为（prob,s',r,done）元组
        self.initial_state_dis = initial_dis  # 应该是一个列表，每个元素是初始state分布的概率
        self.s = None  # current state

    def step(self, action):
        transition = self.transition_P[self.s][action]  # [(prob, s', r, done), ...]
        i = np.random.choice(len(transition), p=[t[0] for t in transition])
        p, next_s, r, d = transition[i]
        self.s = next_s
        return next_s, r, d, {'prob': p}

    def reset(self):
        self.s = np.random.choice(self.state_dim, p=self.initial_state_dis)
        return self.s


class AssetAllocation(DiscreteEnv):
    # Parameter Settings
    aversion_rate: float = 0.01
    riskless_return: float = 0.05
    risky_return: dict = {0.4: 0.5, 0.6: -0.3}  # 风险资产的Bernoulli分布参数 [p: corresponding_return]
    action_space = [0.2, 0.8]

    def __init__(self, init_wealth=10, T=10):
        self.t = None
        self.s = None
        self.initial_wealth = init_wealth
        transition_p = {}  # transition_p[time][wealth][action] = [(prob, next_w, reward, done), ...]
        state_space = {}  # state_space[t] = [state1, state2, ...]
        # Generate state space to store state between time step
        for i in range(T):
            if i == 0:
                state_space[i] = [init_wealth]
            else:
                state_space[i] = []

        # Generate state space and transition prob
        for i in range(1, T):
            transition_p[i - 1] = {}
            for w in state_space[i - 1]:  # 提取出前一时刻的wealth
                transition_p[i - 1][w] = {act: [] for act in self.action_space}  #

                for act in self.action_space:  # 每种wealth有不同action
                    for p, r in self.risky_return.items():  # 每个action有两种可能
                        w_ = w * act * (1 + r) + w * (1 - act) * (1 + self.riskless_return)
                        w_ = np.round(w_, 3)  # 保留3位小数
                        state_space[i] += [w_]  # next state
                        reward = 0
                        transition_p[i - 1][w][act] += [(p, w_, reward, False)]

        # Final state transfer to itself
        transition_p[9] = {}
        for w_ in state_space[9]:
            transition_p[9][w_] = {act: [] for act in self.action_space}
            for act in self.action_space:
                reward = self._reward_compute(w_, self.aversion_rate)
                transition_p[9][w_][act] += [(1, w_, reward, True)]

        state_dim = sum([len(state_space[x]) for x in state_space])  # number of state
        action_dim = len(self.action_space)
        init_dis = [1] + [0] * (state_dim - 1)  # Initial state distribution

        self.state_space = state_space
        super(AssetAllocation, self).__init__(state_dim, action_dim, transition_p, init_dis)

    @staticmethod
    def _reward_compute(wealth, aversion_rate):
        return (-np.exp(- aversion_rate * wealth)) / aversion_rate  # CARA Function

    def reset(self):
        self.s = self.initial_wealth
        self.t = 0
        return f'state wealth: {self.s}, t: {self.t}'

    def step(self, action):
        transition = self.transition_P[self.t][self.s][action]  # [(prob, s', r, done), ...]
        i = np.random.choice(len(transition), p=[t_[0] for t_ in transition])
        p, next_s, r, d = transition[i]
        self.s = next_s
        self.t += 1
        return next_s, r, d, {'prob': p}


class AgentValueIteration:
    DISCOUNT_FACTOR = 0.7

    def __init__(self, env):
        self.env = env
        # Initialize state value & random policy
        self.value_dict = {}  # value_dict[time] = {state1: 0, state2: 0, ...}
        self.policy = {}  # policy[time] = {state1: act1, state2:act2, ...}

        for t_, s_list in env.state_space.items():
            self.value_dict[t_] = {ss_: 0 for ss_ in s_list}
            self.policy[t_] = {ss_: np.random.choice(self.env.action_space) for ss_ in s_list}

    def _action_computation(self, t_, s_, mode=1):
        """
        Used to get the best action and the corresponding action_value at each state (time, wealth)
        :param t_: state t
        :param s_: state wealth
        :param mode: return the whole action value dict (used to plot the state-action table) or just the best
        :return:
        """
        action_values = {act: 0 for act in self.env.action_space}
        for a_ in self.env.action_space:
            for prob, next_state, reward, done in self.env.transition_P[t_][s_][a_]:
                if done:
                    action_values[a_] += prob * (reward + self.DISCOUNT_FACTOR * self.value_dict[t_][next_state])
                    # break
                else:
                    action_values[a_] += prob * (reward + self.DISCOUNT_FACTOR * self.value_dict[t_ + 1][next_state])
        best_action = max(action_values, key=lambda x: action_values[x])
        best_action_value = action_values.get(best_action)
        if mode == 1:
            return best_action, best_action_value
        else:
            return action_values

    def value_iteration(self):
        rep_times = 0

        while True:
            delta = 0
            for t, s_list in self.env.state_space.items():
                for s in s_list:
                    best_a, best_a_v = self._action_computation(t, s)
                    delta = max(delta, np.abs(best_a_v - self.value_dict[t][s]))
                    self.value_dict[t][s] = best_a_v
                    self.policy[t][s] = best_a
            if delta < 0.001:
                print(f'Value Iteration Done!  Total round: {rep_times}')
                break
            print(f'Iteration round:{rep_times}, delta:{delta}')
            rep_times += 1

        return self.policy

    def state_action_value(self):
        """
        Used to draw the state_action value table
        :return: pd.Dataframe( act1, act2, act3
             time1   state1     V1    V2    V3
             time1   state2     V4    V5    V6
             time2   state3     V7    V8    V9)
        """
        s_a_table = {}
        for t, s_list in self.env.state_space.items():
            s_a_table[t] = pd.DataFrame(columns=['state'] + ample.action_space)
            for s in s_list:
                act_vs = self._action_computation(t, s, mode=0)
                a_l = [s] + list(act_vs.values())
                s_a_table[t].loc[len(s_a_table[t]), :] = a_l
        s_a_table = pd.concat(s_a_table.values(), keys=s_a_table.keys(), axis=0)
        s_a_table = s_a_table.reset_index().rename(columns={'level_0': 't'}).drop(columns='level_1')
        s_a_table = s_a_table.set_index(['t', 'state'])
        return s_a_table

    def policy_table(self):
        """Draw the self-policy in dataframe"""
        table = {}
        for t, s_a in self.policy.items():
            table[t] = pd.DataFrame(columns=['state'] + ample.action_space)
            for s, a in s_a.items():
                a_l = [s] + [1 if i == a else 0 for i in ample.action_space]
                table[t].loc[len(table[t]), :] = a_l
        table = pd.concat(table.values(), keys=table.keys(), axis=0)
        table = table.reset_index().rename(columns={'level_0': 't'}).drop(columns='level_1')
        table = table.set_index(['t', 'state'])
        return table


class AgentPolicyIteration:
    DISCOUNT_FACTOR = 0.7

    def __init__(self, env):
        self.env = env
        # Initialize state value & random policy
        self.value_dict = {}
        self.policy = {}

        for t_, s_list in env.state_space.items():
            self.value_dict[t_] = {ss_: 0 for ss_ in s_list}
            self.policy[t_] = {ss_: [(1, np.random.choice(self.env.action_space))] for ss_ in s_list}
            # 每个t时刻的下policy是该时刻下state作为key，（prob，act）作为value的字典的字典

    def _action_computation(self, t_, s_, mode=1):
        """Same as above"""
        action_values = {act: 0 for act in self.env.action_space}
        for a_ in self.env.action_space:
            for prob, next_state, reward, done in self.env.transition_P[t_][s_][a_]:
                if done:
                    action_values[a_] += prob * (reward + self.DISCOUNT_FACTOR * self.value_dict[t_][next_state])
                    # break
                else:
                    action_values[a_] += prob * (reward + self.DISCOUNT_FACTOR * self.value_dict[t_ + 1][next_state])
        best_action = max(action_values, key=lambda x: action_values[x])
        best_action_value = action_values.get(best_action)
        if mode == 1:
            return best_action, best_action_value
        else:
            return action_values

    def _policy_evaluation(self, truncated_k):
        """
        This function is used to evaluate the self-policy, and update the corresponding state value,
        which is the self.value_dict.
        Attention:
            - Since I use truncated policy iteration method, the state value don't need to iter to
            convergence but only k steps.
            - Updating self.value_dict directly means that the initial state value of each policy
            only needs to inherit the final policy value of the previous iteration.
        :param truncated_k: iter steps.
        :return: Difference of the policy state value between each iteration
        """
        rep_times = 0
        delta = 0
        while rep_times < truncated_k:  # Iter the state value computation k times
            delta = 0
            for t, s_list in self.env.state_space.items():  # for each state(time, s)
                for s in s_list:
                    s_policy_value = 0
                    for a_prob, act in self.policy[t][s]:
                        for t_prob, next_s, reward, done in self.env.transition_P[t][s][act]:
                            if done:
                                s_policy_value += a_prob * t_prob * (
                                            reward + self.DISCOUNT_FACTOR * self.value_dict[t][next_s])
                            else:
                                s_policy_value += a_prob * t_prob * (
                                            reward + self.DISCOUNT_FACTOR * self.value_dict[t + 1][next_s])

                    delta = max(delta, np.abs(s_policy_value - self.value_dict[t][s]))
                    self.value_dict[t][s] = s_policy_value  # Update the policy state value
            rep_times += 1
        return delta

    def _policy_improvement(self):
        for t, s_list in self.env.state_space.items():
            for s in s_list:
                best_a, best_a_v = self._action_computation(t, s)
                self.policy[t][s] = [(1, best_a)]  # greedy policy

    def policy_iteration(self, truncated_k=15):
        rep_times = 0
        while True:
            # Policy evaluation
            delta = self._policy_evaluation(truncated_k=truncated_k)
            # Policy improvement
            self._policy_improvement()
            if delta < 0.001:
                print(f'Policy Iteration Done! delta:{delta} Total round: {rep_times}')
                break
            print(f'Iteration round:{rep_times}, delta:{delta}')
            rep_times += 1
        return self.policy

    def policy_table(self):
        """Draw the policy in dataframe, todo in future"""
        ...


class AgentMC:
    DISCOUNT_FACTOR = 0.7
    EPSILON = 0.05

    def __init__(self, env):
        self.env = env
        # Initialize state vale & policy
        self.policy = {}  # self.policy[t] = {state:[(act1_prob, act1), (act2_prob, act2), ...]}
        self.Q_table = {}  # self.Q_table[(t, state)] = {act1: [0], act2: [0], ...}
        for t_, s_list in env.state_space.items():
            # Initial Policy is deterministic
            self.policy[t_] = {ss_: [(1, np.random.choice(self.env.action_space))] for ss_ in s_list}
            for ss_ in s_list:
                # Since MC method needs to compute the mean of sampling Q-value, using list to store each sample result
                self.Q_table[(t_, ss_)] = {act: [0] for act in self.env.action_space}

    def episode(self, policy):
        """
        Generate trajectory under particular policy
        :param policy: policy[t] = {state:[(act1_prob, act1), (act2_prob, act2), ...]}
        :return: trajectory [(t, state, action, reward), ...]
        """
        self.env.reset()
        s, t = self.env.s, self.env.t
        trajectory = []
        while True:
            a = []
            p = []
            for p_a in policy[t][s]:  # [(prob1, act1), (prob2, act2)]
                p.append(p_a[0])
                a.append(p_a[1])
            act = np.random.choice(a, p=p)

            next_s, r, done, _ = self.env.step(act)
            trajectory.append((t, s, act, r))

            if done:
                break
            s = next_s
            t += 1
        return trajectory

    def _best_action_computation(self, t, s):  # Compute the best action based on current Q-table
        action_values = {}
        for act in self.env.action_space:
            action_values[act] = np.mean(self.Q_table[(t, s)][act])  # MC method needs the mean of sampling q-value
        best_a = max(action_values, key=lambda x: action_values[x])
        best_a_v = action_values.get(best_a)
        return best_a, best_a_v

    def _policy_improvement(self, t, s, best_a_):
        self.policy[t][s] = []
        act_d = self.env.action_dim
        exploring_p = self.EPSILON / act_d  # Epsilon-greedy Updating
        exploiting_p = 1 - (((act_d - 1) * self.EPSILON) / act_d)
        for act in self.env.action_space:
            if act == best_a_:
                self.policy[t][s] += [(exploiting_p, act)]
            else:
                self.policy[t][s] += [(exploring_p, act)]

    def mc_iteration(self, episode_num=10000):
        for ith in range(1, episode_num + 1):
            if ith % 1000 == 0:
                print(f'Episode Round: {ith} / {episode_num}')

            g = 0
            trajectory = self.episode(self.policy)  # Generate the episode sample

            for step in trajectory[::-1]:  # Backward Computation
                t, s, act, reward = step

                g = self.DISCOUNT_FACTOR * g + reward  # g(s,a)
                self.Q_table[(t, s)][act] += [g]

                # action value Evaluation on this state
                best_a, _ = self._best_action_computation(t, s)
                # epsilon-policy improvement
                self._policy_improvement(t, s, best_a)
        return self.policy


class AgentSARSA:
    DISCOUNT_FACTOR = 0.7
    EPSILON = 0.05

    def __init__(self, env):
        self.env = env
        self.policy = {}  # self.policy[t][state] = [(prob1, act1),(...)]
        self.Q_table = {}  # self.Q_table[(t, state)] = {act1: [0], act2: [0], ...}

        self.exploring_p = self.EPSILON / env.action_dim
        self.exploiting_p = 1 - (((env.action_dim - 1) * self.EPSILON) / env.action_dim)

        for t_, s_list in env.state_space.items():
            self.policy[t_] = {}
            for ss_ in s_list:
                self.Q_table[(t_, ss_)] = {act: 0 for act in env.action_space}
                # Equal initial action Prob
                self.policy[t_][ss_] = [(1 / env.action_dim, act) for act in env.action_space]

    def _best_action_computation(self, t, s):
        action_values = self.Q_table[(t, s)]
        best_a = max(action_values, key=lambda x: action_values[x])
        best_a_v = action_values.get(best_a)
        return best_a, best_a_v

    def _policy_action_choose(self, t, s):  # Choose action under self.policy
        a = []
        p = []
        for p_a in self.policy[t][s]:
            a.append(p_a[1])
            p.append(p_a[0])
        return np.random.choice(a, p=p)

    def _policy_improvement(self, best_a):
        p_a = []
        for act in self.env.action_space:
            if act == best_a:
                p_a.append((self.exploiting_p, act))
            else:
                p_a.append((self.exploring_p, act))
        return p_a

    def sarsa_iteration(self, episode_num=10000, alpha=0.1):

        for ith in range(1, episode_num + 1):
            if ith % 1000 == 0:
                print(f'Episode Round: {ith} / {episode_num}')

            self.env.reset()
            while True:
                # Generate experience (s, a, r, s_t+1, a_t+1)
                t, s = self.env.t, self.env.s
                act = self._policy_action_choose(t, s)
                next_s, r, done, _ = self.env.step(act)
                if done:
                    break
                next_act = self._policy_action_choose(t + 1, next_s)

                # Compute the TD target and update q table
                td_target = r + self.DISCOUNT_FACTOR * self.Q_table[(t + 1, next_s)][next_act]
                self.Q_table[(t, s)][act] += alpha * (self.Q_table[(t, s)][act] - td_target)

                best_a, _ = self._best_action_computation(t, s)
                self.policy[t][s] = self._policy_improvement(best_a)  # Update self-policy


class AgentQLearning:
    DISCOUNT_FACTOR = 0.7
    EPSILON = 0.05

    def __init__(self, env):
        self.env = env
        self.policy = {}
        self.Q_table = {}

        self.exploring_p = self.EPSILON / env.action_dim
        self.exploiting_p = 1 - (((env.action_dim - 1) * self.EPSILON) / env.action_dim)

        for t_, s_list in env.state_space.items():
            self.policy[t_] = {}
            for ss_ in s_list:
                self.Q_table[(t_, ss_)] = {act: 0 for act in env.action_space}
                # Equal initial action Prob
                self.policy[t_][ss_] = [(1 / env.action_dim, act) for act in env.action_space]

    def reset(self):
        self.policy = {}
        self.Q_table = {}
        for t_, s_list in self.env.state_space.items():
            self.policy[t_] = {}
            for ss_ in s_list:
                self.Q_table[(t_, ss_)] = {act: 0 for act in self.env.action_space}
                # Equal initial action Prob
                self.policy[t_][ss_] = [(1 / self.env.action_dim, act) for act in self.env.action_space]

    def _policy_action_choose(self, t, s, policy=None):  # Choose action under self.policy
        a = []
        p = []
        if policy:
            for p_a in policy[t][s]:
                a.append(p_a[1])
                p.append(p_a[0])
            return np.random.choice(a, p=p)
        else:
            for p_a in self.policy[t][s]:
                a.append(p_a[1])
                p.append(p_a[0])
            return np.random.choice(a, p=p)

    def _best_action_computation(self, t, s):
        action_values = self.Q_table[t, s]
        best_a = max(action_values, key=lambda x: action_values[x])
        best_a_v = action_values.get(best_a)
        return best_a, best_a_v

    def _policy_improvement(self, best_a):
        p_a = []
        for act in self.env.action_space:
            if act == best_a:
                p_a.append((self.exploiting_p, act))
            else:
                p_a.append((self.exploring_p, act))
        return p_a

    def qlearning_iteration(self, episode_num=10000, alpha=0.1):  # ON-policy
        for ith in range(1, episode_num + 1):
            if ith % 1000 == 0:
                print(f'Episode Round: {ith} / {episode_num}')

            self.env.reset()
            while True:
                t, s = self.env.t, self.env.s
                act = self._policy_action_choose(t, s)
                next_s, r, done, _ = self.env.step(act)
                if done:
                    break
                _, best_next_a_v = self._best_action_computation(t + 1, next_s)

                td_target = r + self.EPSILON * best_next_a_v
                self.Q_table[(t, s)][act] += alpha * (self.Q_table[(t, s)][act] - td_target)

                best_a, _ = self._best_action_computation(t, s)
                self.policy[t][s] = self._policy_improvement(best_a)

    def qlearning_iteration_Off_policy(self, episode_num=10000, alpha=0.1):
        """
        Off-policy version, code is as same as above, only use different sampling policy to draw data.
        :param episode_num:
        :param alpha:
        :return:
        """
        # Sampling Policy just as same as the initial policy
        sample_policy = copy.deepcopy(self.policy)

        for ith in range(1, episode_num + 1):
            if ith % 1000 == 0:
                print(f'Episode Round: {ith} / {episode_num}')

            self.env.reset()
            while True:
                t, s = self.env.t, self.env.s
                act = self._policy_action_choose(t, s, sample_policy)
                next_s, r, done, _ = self.env.step(act)
                if done:
                    break
                _, best_next_a_v = self._best_action_computation(t + 1, next_s)

                td_target = r + self.EPSILON * best_next_a_v
                self.Q_table[(t, s)][act] += alpha * (self.Q_table[(t, s)][act] - td_target)

                best_a, _ = self._best_action_computation(t, s)
                self.policy[t][s] = self._policy_improvement(best_a)


if __name__ == '__main__':
    # Generate Environment
    ample = AssetAllocation()
    ample.reset()

    # Model Based Method
    ## Dynamic Programming
    ### Value Iteration
    ample_vi = AgentValueIteration(ample)
    ample_vi.value_iteration()
    print(ample_vi.policy)  # Check the convergent policy

    ### Policy Iteration
    ample_pi = AgentPolicyIteration(ample)
    ample_pi.policy_iteration()
    print(ample_pi.policy)

    # Model Free Method
    ## Monte Carlo Method
    ### MC Exploring Start with epsilon-greedy method
    ample_mc = AgentMC(ample)
    ample_mc.mc_iteration()
    print(ample_mc.policy)

    ## TD Method
    ### TD SARSA
    ample_sarsa = AgentSARSA(ample)
    ample_sarsa.sarsa_iteration()
    print(ample_sarsa.policy)

    ### TD Q-learning
    ample_qlearning = AgentQLearning(ample)
    ample_qlearning.qlearning_iteration()
    print(ample_qlearning.policy)
