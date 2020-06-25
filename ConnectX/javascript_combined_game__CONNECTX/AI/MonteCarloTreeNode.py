import numpy as np


class TreeNode(object):

    def __init__(self, prior_prob, parent=None):
        self.parent = parent
        self.children = {}  # key=action, value=TreeNode
        self.reward = 0  #  Total simulation reward of the node.
        self.visited_times = 0  #  Total number of visits of the node.
        self.prior_prob = prior_prob  #  prior probability of the move.

    def is_root(self):
        """
        Whether the node is the root node.
        :return: <Bool>
        """
        return self.parent is None

    def expand(self, action, probability):
        """
        Expand node.
        :param action: Selected extended action.
        :param probability: prior probability of the move.
        :return: <TreeNode> 
        """
        if action in self.children:
            return self.children[action]

        child_node = TreeNode(prior_prob=probability,
                              parent=self)
        self.children[action] = child_node

        return child_node

    def UCT_function(self, c=5.0):
        greedy = c * self.prior_prob * np.sqrt(self.parent.visited_times) / (1 + self.visited_times)
        if self.visited_times == 0:
            return greedy
        return self.reward / self.visited_times + greedy

    def choose_best_child(self, c=5.0):
        """
        According to the UCT function, select an optimal child node.
        :param c: greedy value.
        :return: <(action(x_axis, y_axis), TreeNode)>  An optimal child node.
        """
        return max(self.children.items(), key=lambda child_node: child_node[1].UCT_function(c))

    def backpropagate(self, value):
        """
        Backpropagate, passing the result to the parent node.
        :param value: The value to be backpropagated.
        """
        self.visited_times += 1
        self.reward += value

        if not self.is_root():
            self.parent.backpropagate(-value)
