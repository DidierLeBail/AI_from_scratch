"""Implements a simple 1d version of the Bayesian information gain (BIG) framework described [here](https://telecom-paris.hal.science/hal-03323514/document).
"""

from typing import List, Callable, Union
import numpy as np

class Prior:
    """A prior is a distribution over integers.
    """

    def __init__(self,
        nb_thetas: int,
        probas: List[int]
    ):
        self.nb_thetas = nb_thetas
        self.probas = probas

    def draw(self, rng: np.random.Generator):
        return rng.choice(self.nb_thetas, p=self.probas)

class User:
    """In the BIG framework, a user is characterized by:
    - a probabilistic policy
    - a hidden parameter (goal)
    - an observed action
    
    Attributes
    ----------
    goal : int
        the user target
    _prior : Prior
        the user true prior (the target is drawn from it)
    _policy : Distribution
        the distribution conditioned on `_theta` and the stimulus received by the user,
        according to which the user picks an action
    """

    def __init__(self,
        prior: Prior,
        policy: Callable
    ):
        self.goal = None
        self._prior = prior
        self._policy = policy

    def pick_goal(self, rng: np.random.Generator):
        """The user picks a goal."""
        self.goal = self._prior.draw(rng)

    def pick_action(self, stimulus: int):
        """Apply the user policy:
        it determines the use action from its observation of the world (`stimulus`) and its own hidden state (`self.goal`).
        Note that contrary to a RL agent, this hidden state does not change, because it reflects the user goal.
        """
        return self._policy(stimulus, self.goal)

class Computer:
    """In the BIG framework, the computer tries to guess the user's goal by sending to the user
    the stimulus that maximally informs about the user goal.

    It is characterized by:
    - a model for the user policy
    - a prior over the user goals
    - a policy for returning a stimulus to the user

    This policy aims at maximizing the information gain about the user goal and requires to define a stimulus space.
    Here we take this space to be the range of integers from 0 to `nb_stim - 1`.
    Note that the user goal takes its values into the same space.

    Parameters
    ----------
    prior : Prior
        the prior about the user goal, that here lives in the same space as the stimulus
        (it is an integer between 0 and `nb_stim - 1`) ;
        this prior is updated according to the user model and user action
    model : Callable
        the model describing the user, as a distribution over user actions conditioned on user goal and stimulus ;
        is not updated, and may differ from the actual user policy
    nb_stim : int
        the nb of possible stimulus (size of the stimulus space)
    nb_actions : int
        size of the user action space
    
    Attributes
    ----------
    goals : numpy.ndarray
        list of all possible values for the user goal
        (here same values as the stimulus)
    """

    def __init__(self,
        prior: Prior,
        model: Callable,
        nb_stim: int,
        nb_actions: int
    ):
        self.prior = prior
        self.model = model
        self.nb_stim = nb_stim
        self.goals = np.arange(nb_stim, dtype=int)
        self.nb_actions = nb_actions
    
    def init_stimulus(self, rng: np.random.Generator):
        """Here drawing a stimulus is the same as drawing a goal."""
        return self.prior.draw(rng)

    def update_prior(self, stimulus: int, user_action: int):
        """Updates `self.prior` the distribution over the user's goals.
        """
        self.prior.probas *= self.model(user_action, stimulus, self.goals)
        self.prior.probas /= np.sum(self.prior.probas)

    def utility(self, stimulus: int):
        """Computes the utility of `stimulus`:
        in the BIG framework, it is the information gain about the user goal when conditioned on `stimulus`, averaged over the user actions.
        It is the mutual information between the user goal and action, conditioned on the stimulus.
        """
        # compute the entropy of the distribution of user actions conditioned on the user stimulus
        p_yx = np.zeros(self.nb_actions)
        for user_action in range(self.nb_actions):
            p_yx[user_action] = np.sum( self.prior.probas * self.model(user_action, stimulus, self.goals) )
        H_yx = - np.sum( p_yx * np.log2(p_yx) )

        # compute the entropy of the distribution of user actions conditioned on user goal and stimulus
        q_yx = np.zeros(self.nb_actions)
        for user_action in range(self.nb_actions):
            mod = self.model(user_action, stimulus, self.goals)
            q_yx[user_action] = np.sum( self.prior.probas * mod * np.log2(mod) )
        H_yThetaX = - np.sum(q_yx)

        # deduce the mutual information between user goal and action, conditioned on stimulus
        return H_yx - H_yThetaX

    def pick_stimulus(self, past_stimulus: int, user_action: int):
        # update the prior over the user goal, given the user reaction to last stimulus
        self.update_prior(past_stimulus, user_action)
        
        # compute the utility of each stimulus
        utilities = np.zeros(self.nb_stim)
        for stim in range(self.nb_stim):
            utilities[stim] = self.utility(stim)
        
        # pick the stimulus of highest utility
        return np.argmax(utilities)

if __name__ == "__main__":
    # define the user
    nb_goals = 20
    probas = np.ones(nb_goals) / nb_goals
    true_prior = Prior(nb_goals, probas)

    nb_stim = nb_goals
    nb_actions = 3
    def true_policy(stimulus: int, goal: int):
        """Returns the user action given stimulus and goal."""
        return np.heaviside(stimulus - goal, 2).astype(int)

    user = User(true_prior, true_policy)

    # define the computer
    prior = Prior(nb_goals, probas)

    def model(action: int, stimulus: int, goals: Union[int, np.ndarray]):
        """Returns the proba of a user action given a stimulus and user goal.
        
        Note that the probas associated to different goals can be computed in a single pass,
        if `goals` is a `numpy.ndarray`.

        Notes
        -----
        At the boundary, the model should take into account that one of the user actions is impossible.
        Otherwise, it takes much longer for the computer to guess the user goal when this goal matches with one of the boundaries
        (it takes a linear time instead of a logarithmic time).
        """
        true_a = true_policy(stimulus, goals)
        a1 = (true_a + 1) % 3
        a2 = (true_a + 2) % 3

        lamb = 3 / 4
        lamb1 = 1 / 8
        lamb2 = 1 / 8

        tab = action + np.zeros_like(true_a)
        old_proba = lamb * (tab == true_a) + lamb1 * (tab == a1) + lamb2 * (tab == a2)

        # the action 1 is impossible and so the probas of 0 and 2 must be renormalized
        if stimulus == 0:
            if action == 1:
                return np.zeros_like(goals) + 1e-14
            
            norm = np.where(goals > 0, 1 - lamb1, 1 - lamb2)
            return old_proba / norm
        
        # the action 0 is impossible
        elif stimulus == nb_stim - 1:
            if action == 0:
                return np.zeros_like(goals) + 1e-14
            
            norm = np.where(goals < nb_stim - 1, 1 - lamb2, 1 - lamb1)
            return old_proba / norm
            
        return old_proba

    computer = Computer(prior, model, nb_stim, nb_actions)

    # make the two interact until the user goal has been reached
    rng = np.random.default_rng(seed=None)
    user.pick_goal(rng)
    
    print(f"goal chosen by the user: {user.goal}")
    print()

    stim = computer.init_stimulus(rng)

    print(f"first stimulus sent by the computer: {stim}")
    print()

    nb_trials = 0
    while stim != user.goal:
        action = user.pick_action(stim)
        stim = computer.pick_stimulus(stim, action)
        #print(f"new stimulus sent by the computer: {stim}")
        #print()
        nb_trials += 1
    
    print(f"user goal guessed by the computer in {nb_trials} trials!")
    print(f"average nb of trials if dichotomy: {np.round(np.log2(100), 2)}")
