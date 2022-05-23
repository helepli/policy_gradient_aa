# Policy Gradient REINFORCE with the Actor-Advisor (Policy Intersection + Learning correction)

The Actor-Advisor [[1]](#1) is a Policy Shaping method based on the Policy Intersection formula [[2]](#2), adapted to Policy Gradient methods [[3]](#3). At acting time, the agent samples the mixture of its policy and of an advisory policy, following the Policy Shaping formula in [[2]](#2):

<img src="https://latex.codecogs.com/svg.latex?\Large&space;a_t\sim\pi_L(s_t)\times\pi_A(s_t)=\frac{\pi_L(s_t)\,\pi_A(s_t)}{\pi_L(s_t)\cdot\pi_A(s_t)}" title="\Large a_t\sim\pi_L(s_t)\times\pi_A(s_t)=\frac{\pi_L(s_t)\,\pi_A(s_t)}{\pi_L(s_t)\cdot\pi_A(s_t)}" />

The adapation required for Policy Gradient to allow an advisory policy to be mixed with the policy it is currently learning is to incoprorate the advisory policy in the loss:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;loss=-\sum\limits_{t=0}^{T}G_t\log(\pi_{\theta}(a_t|s_t)\times\pi_A(a_t|s_t))" title="\Large loss=-\sum\limits_{t=0}^{T}G_t\log(\pi_{\theta}(a_t|s_t)\times\pi_A(a_t|s_t))" />

This implementation of REINFORCE is from https://github.com/sonic1sonic/Monte-Carlo-Policy-Gradient-REINFORCE/blob/master/REINFORCE.py

In this experiment, we train an agent without the guidance of an advisory policy for 2000 episodes on Lunar Lander. We then freeze it and save it to use it as an advisor. We launch two fresh REINFORCE agents learning learnign while being advised/guided by our advisor agent trained before, which we load. One of our two advisees only uses the Policy Shaping formula at acting time, without the learning correction in the loss, the other uses both the Policy Shaping formula at acting time and the learning correction in the loss.

Our results show that REINFORCE needs a learning correction to be able to learn well and exploit the advice of the previously trained advisor agent, otherwise performance plummets.

## References
<a id="1">[1]</a>
Plisnier, H., Steckelmacher, D., Brys, T., Roijers, D., Nowé, A., "Directed Policy Gradient for Safe Reinforcement Learning with Human Advice", 2018, European Workshop On Reinforcement Learning 14 (EWRL14)
<a id="2">[2]</a>
Griffith, S., Subramanian, K., Scholz, J., Isbell, C. L., Andrea T. L., "Policy shaping: Integrating human feedback with reinforcement learning", Advances in neural information processing systems 2013
<a id="3">[3]</a>
Sutton, R., McAllester, D., Singh, S., Mansour, Y., "Policy Gradient Methods for Reinforcement Learning with Function Approximation", Neural Information Processing Systems (NIPS) 2000

