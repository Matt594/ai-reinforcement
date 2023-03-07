# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()


    def runValueIteration(self):
        # Write value iteration code here
        for _ in range(self.iterations):
            updated_values = {}
            for state in self.mdp.getStates():
                max_q_value = -float('inf')
                if self.mdp.isTerminal(state):
                    max_q_value = 0
                else: 
                    for action in self.mdp.getPossibleActions(state):
                        q_value = self.computeQValueFromValues(state, action)
                        if q_value > max_q_value:
                            max_q_value = q_value
                updated_values[state] = max_q_value
                
            for key in updated_values:
                self.values[key] = updated_values[key]        


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

          Qk+1(s,a) = sum(T(s,a,s') * [R(s,a,s') + (discount * (Vk(s'))])
        """
        
        total_quality_value = 0
        for new_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, new_state)
            quality_value = self.getValue(new_state)
            total_quality_value += prob * (reward + (self.discount * quality_value))
        return total_quality_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        max_action = None
        max_q_value = -float('inf')
        if self.mdp.isTerminal(state):
            return max_action
        for action in self.mdp.getPossibleActions(state):
            q_value = self.computeQValueFromValues(state, action)
            if q_value > max_q_value:
                max_action = action
                max_q_value = q_value
        return max_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for i in range(self.iterations):
            max_q_value = -float('inf')
            state = self.mdp.getStates()[i % len(self.mdp.getStates())]
            if self.mdp.isTerminal(state):
                max_q_value = 0
            else: 
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    if q_value > max_q_value:
                        max_q_value = q_value
            self.values[state] = max_q_value
   

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Compute predecessors of all states
        predecessors = dict()
        for predecessor in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(predecessor):
                for successor, prob in self.mdp.getTransitionStatesAndProbs(predecessor, action):
                    if prob > 0.0:
                        if successor not in predecessors:
                            predecessors[successor] = set()
                        predecessors[successor].add(predecessor)
        # Initialize priority queue
        pq = util.PriorityQueue()
        # Find the diff of every state and update the priority queue
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                max_q_value = -float('inf')
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    if q_value > max_q_value:
                        max_q_value = q_value
                diff = abs(self.values[state] - max_q_value)
                pq.update(state, -diff)
        
        # Loop through iterations
        for _ in range(self.iterations):
            # Terminate if empty
            if pq.isEmpty():
                return
            # Pop state off the priority queue
            state = pq.pop()
            assert not self.mdp.isTerminal(state)
            # Update value of state
            max_q_value = -float('inf')
            for action in self.mdp.getPossibleActions(state):
                q_value = self.computeQValueFromValues(state, action)
                if q_value > max_q_value:
                    max_q_value = q_value
            self.values[state] = max_q_value
            # Find the diff of every predecessor and update the priority queue
            for predecessor in predecessors[state]:
                assert not self.mdp.isTerminal(predecessor)
                max_q_value = -float('inf')
                for action in self.mdp.getPossibleActions(predecessor):
                    q_value = self.computeQValueFromValues(predecessor, action)
                    if q_value > max_q_value:
                        max_q_value = q_value
                diff = abs(self.values[predecessor] - max_q_value)
                if diff > self.theta:
                    pq.update(predecessor, -diff)

