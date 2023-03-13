# Asset_Allocation_in_Reinforcement_Learning
HKUST MAFM course project for the Asset Allocation in Reinforcement Learning


## Introduction
This project is one of my class assignments completed at HKUST M.Sc. in 
Financial Mathematics (MAFM) program , 2023 Spring, for the course MAFM 5370 
Reinforcement Learning with Financial Applications.
  
This project consider the simplest situation in asset allocation area and 
using several fundamental method in RL to achieve the optimal investment policy. 
  
The project contains the code of: Environment settings, Dynamics Programming, 
MC Algorithm and TD Algorithm.

Learning Materials is from my class teacher: Chak Wong.  
Most of the Algorithme Pseudocode is from Bilibili course: 
[Mathematical Foundations of Reinforcement Learning](https://www.bilibili.com/video/BV1KY4y1N7H8/?spm_id_from=333.788&vd_source=c6859ec5158d515b50f001aba53cc8f9)
and its Book GitHub link: 
[Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathmatical-Foundation-of-Reinforcement-Learning)

_Attention: there may be some errors in the code, since I'm still a beginner
in RL, please raise issue if you find sth wrong._

****

## Problem Description
> Consider the discrete-time asset allocation example in section 8.4 of Rao and Jelvis.  
> Suppose the single-time-step return of the risky asset from time t to t+1 
> as 𝑌_𝑡=𝑎, 𝑝𝑟𝑜𝑏=𝑝,  𝑎𝑛𝑑 𝑏, 𝑝𝑟𝑜𝑏=(1−𝑝) .  Suppose that T =10, 
> see the TD method to find the Q function, and hence the optimal strategy.
  
The **objective** of the project is to find the optimal policy, which describe how much money(action) you 
should invest in the risky asset under different time & total wealth(states).

## **Project Layout**
The project code can be divided into 8 classes, first three classes complete the definition of
the basic environment of the asset allocation, the rest classes complete the implementation of 
different RL algorithms.

### **Model Assumption & Env Setting**
Considering the fact that, in asset allocation, our investment decision is based on the current amount 
of money and the time due to maturity, I set the time and amount of wealth as the state in the Env 
system. Besides, due to the constraints of algorithm running speed, I simply set two action to do 
in each state, which means the agent can either invest 0.2 or 0.8 of his total wealth into risky asset. 
Based on the above assumption, I simply view it as a finite discrete problem in reinforcement learning.
  
The Env setting is as below:
- **State**: (t, wealth)
- **Action**: (0.2, 0.8)
- **Reward**: CARA utility function of wealth  - exp(- a * w) / a
- Risky returns ~ Bernoulli {0.4: go up 0.5, 0.6: go down -0.3}
- Riskless return: 0.05

Since the prob and act is known and state is finite, I first generate all the possible wealth states at different time 
using For loop, then use Dict to restore the transition probabilistic matrix and 
the states space(see the AssetAllocation class). I also implement the step and reset function in the class. After finish 
the env setting, we can generate agent algorithm.


### **Algorithm Code**
I complete 5 different algorithm in the project. First two are model-based method, the reset are model free method. In 
each algorithm class, I first initialize the basic elements such as: state value, policy and Q_table, depends on what it 
needs. Secondly, I customize some functions such as find best action & action value etc. in each class, the usage of each 
function is well explained in the annotations and their names.

#### **Dynamic Programming**
Under this algorithm, we assume that we already know the transfer prob, so we can access the env.transition_P to get the 
transition probabilistic at each state (time, wealth_t). Therefore, we can easily compute state value and action value, 
and thus get the optimal state-action value and the optimal policy through the iteration form of 
the Bellman Optimal Equation.
I complete the **value iteration method** and the **truncated policy iteration method**.

- Value Iteration is just the iterative form to solve BOE.
- Policy Iteration first need to evaluate the initial policy, which is like Value Iteration but under certain policy. After
the evaluation, we need to improve the policy using greedy method.(Truncated Policy don't need to do the evaluation until
convergence but only k steps)

_The Pseudocode can be checked in the above Book link._

#### **MC Exploring Start with Epsilon-Greedy policy**
Under this algorithm, we can only estimate the transition prob using data sampling. Thus, I write an episode function to 
generate the data. The rest is just as the pseudocode.

The core idea of the code is to generate the corresponding policy and state of the instance when 
the class is created, and then update the policy through mc_iteration function to get the 
optimal policy and final q-table.

_The Pseudocode can be checked in the above Book link._

#### **TD SARSA**
Unlike the MC method have to compute the mean of q value, which means we have to update policy after 
we get all data, the TD SARSA method allow us to compute q value and update policy once we have 
only one data.

Due to the problem request and env setting, the First-Visit and Every-Visit method can not be deployed in this project, 
hope to compete them in the future.

_The Pseudocode can be checked in the above Book link._

#### **TD Q-Learning**
Unlike the TD SARSA have to get the experience of (s, a, r, s, a), the TD Q-learning method only needs
(s, a, r, s), which decrease the variance and become more effective.

I write the **On-Policy** and **Off-Policy** iteration method of the Q-Learning algorithm. The only difference 
between these two method is whether the update policy is as same as the sampling policy.

_The Pseudocode can be checked in the above Book link._

### **Result Analysis**
The codes running results can be checked in the [jupyter file](test0309.ipynb).

All the algorithm's convergence results are the same: take action 0.2 at all (time, wealth) states. 
Considering that the expectation returns of each action is same at all states, the interpretation of 
this result may be that the agent should only take the action which has higher expectation return no matter at what states.

As for the compares of different algorithms:
- The truncated policy iteration cause less iterative round than the value iteration, guess its due to the setting of 
truncated k.
- Some states in MC method can not converge to the optimal policy, which may have been affected by the number of 
sampling episode.
***

_To be done:_
- _Complete MC Exploring Starts with importance sampling_
- _Try Expected SARSA and n-step SARSA algorithm_
- _Complicate the env setting and expand it to continuous state-action problem._
