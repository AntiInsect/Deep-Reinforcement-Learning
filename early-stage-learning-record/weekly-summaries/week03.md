# week-three-report

* Due to the coming deadline of convex optimization course project, I adjusted the content of this week to balance the workload. However, besides reading reference and articles, I tried the RL course taught by ::David Silver:: on youtube. I have finished the first 3 Lectures.
* PS: more detailed notes are in the repo (marked and modified directly on the original slides)

## Class 1

* Characteristics of Reinforcement Learning
  * There is no supervisor, only a reward signal
  * Feedback is delayed, not instantaneous
  * Time really matters (sequential, non i.i.d data)
  * Agent’s actions affect the subsequent data it receives
* Sequential Decision Making
* Different State
  * Agent State
  * Environment State
  * Information State
  * Fully Observable Environments
  * Partially Observable Environments
* Learning and Planning
* Exploration and Exploitation
* Prediction and Control

## Class 2 : Markov Decision Processes

* MDP features for RL
* Markov Process
* Markov Reward Process
  * Return
  * Why discount factor
  * Value Function
  * Bellman Equation for MRPs
* Markov Decision Process
  * Policy
  * Value Function
  * Bellman Expectation Equation
  * Optimal Value Function
  * Optimal Policy

## Class 3 : Planning by Dynamic Programming

* General Idea About DP
* Policy Evaluation (synchronous backups)
* Policy Iteration (feedback process)
  * Modified Policy Iteration
  * Generalized Policy Iteration
* Principle of Optimality
* Deterministic Value Iteration
* Value Iteration
* Synchronous Dynamic Programming Algorithms
* Asynchronous Dynamic Programming
