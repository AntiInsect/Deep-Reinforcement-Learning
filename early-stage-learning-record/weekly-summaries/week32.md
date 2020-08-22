# Week32

- A Survey of Exploration Strategies in Reinforcement Learning
- Reinforcement Learning without rewards
    1. a form of “unsupervised” RL.
    2. The idea of MaxEnt RL can enhance the standard RL when there is no reward in environment (sparse reward or poor-shaped reward)
- Provably Efficient Maximum Entropy Exploration
- Diagnosing Bottlenecks in Deep Q-learning Algorithms
    1. Address the overestimation problem in Q-leanring and show the advantage of adding higher-entropy to the resulting policy
- An Information-Theoretic Optimality Principle for Deep Reinforcement Learning
    1. By adapting concepts from information theory, it introduces an intrinsic penalty signal encouraging reduced Q-value estimates. The resultant algorithm encompasses a wide range of learning outcomes containing deep Q-networks as a special case.
- Entropy Regularization with Discounted Future State Distribution in Policy Gradient Methods
    1. regularize the policy gradient objective with entropy
    2. providing a practically feasible algorithm to estimate the normalized discounted weighting of states, i.e, the discounted future state distribution.
    3. exploration can be achieved by entropy regularization with the discounted state distribution in policy gradients, where a metric for maximal coverage of the state space can be based on the entropy of the induced state distribution.
- Bridging the Gap Between Value and Policy Based Reinforcement Learning
    1. Entropy as a bridge of Value and Policy Based methods
    2. a new connection between value and policy based reinforcement learning (RL) based on a relationship between softmax temporal value consistency and policy optimality under entropy regularization.
- Combining Policy Gradient and Q-learning
    1. an equivalency between action-value fitting techniques and actor-critic algorithms, showing that regularized policy gradient techniques can be interpreted as advantage function learning
    algorithms.