# week-two-report

## This week

* finish the tutorial on the openAI website ( see notes in appendix )
* mainly learn about the Policy Gradient ( see notes in appendix )
* read the first two chapters of the book “Reinforcement Learning : an introduction” to gain learn more detailed basic knowledge and gain more intuition
* read the paper “Resource Management with Deep Reinforcement Learning”, discovered and solved  almost all blind spots and misunderstandings about reinforcement learning ( see notes in appendix )
* review Conditional Expectation and Probability Distribution

## Important Getaway

* the Tensorflow placeholder : a more basic concept than variable, similar to an uninitialized variable (which we will not use at once). It allows us to create our operations and build our computation graph, without needing the data
* RL unlike other ML :
* focus more on goal-directed learning from interaction
* learn like an infant — without no explicit teacher but with a direct sensorimotor connection to its environment
* involving interaction between an active decision-making and its environment, within which the agent seeks to achieve a goal despite uncertainty about its environment
* the learner is NOT told which actions to take, but instead must discover which actions yield the most reward by trying them ( evaluate the actions taken rather than instructs by giving correct actions )
* DIFFERENT from supervised learning ( which learns from a training set of LABELED examples provided by a knowledge external supervisor ) — an agent must learn from its own experience and so unsupervised learning ( which try to find the hidden structure ) — the goal of an agent is to maximize a reward signal
* key features : 👈
  * trial-and-error search and delayed reward
  * trade-off between exploration and exploitation
  * it explicit considers the WHOLE problem of a goal-directed agent interacting with an uncertain environment ( do not isolate subproblems )
  * reinforcement learning methods and evolutionary methods are very different 👈
  * evolutionary methods evaluate the “lifetime” behavior of many non-learning agents, each using a different policy for the interacting with its environment, and select those that are able to obtain the most reward
  * variant : methods that search in spaces of policies defined by a collection of numerical parameters. They estimate the directions in the parameters should be adjusted in order to most rapidly improve a policy’s performance

## Workflow enhancement
  
* use Endnote to store the papers for neat organisation
