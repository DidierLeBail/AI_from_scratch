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
    theta : int
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
        self.theta = None
        self._prior = prior
        self._policy = policy

    def pick_goal(self, rng: np.random.Generator):
        """The user picks a goal."""
        self.theta = self._prior.draw(rng)

    def pick_action(self, stimulus: int):
        """Apply the user policy:
        it determines the use action from its observation of the world (`stimulus`) and its own hidden state (`self._theta`).
        Note that contrary to a RL agent, this hidden state does not change, because it reflects the user goal.
        """
        return self._policy(stimulus, self.theta)

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
    prior : numpy.ndarray
        the prior about the user goal, that here lives in the same space as the stimulus
        (it is an integer between 0 and `nb_stim - 1`) ;
        this prior is updated according to the user model and user action ;
        here the prior consists in a list with `self.prior[i] = proba of self.thetas[i]`
    model : Distribution
        the modeling of the user, as a distribution over user actions conditioned on user goal and stimulus ;
        is not updated, and may differ from the actual user policy
    nb_stim : int
        the nb of possible stimulus (size of the stimulus space)
    nb_actions : int
        size of the user action space
    
    Attributes
    ----------
    thetas : numpy.ndarray
        list of all possible values for the user goal
        (here same values as the stimulus)
    """

    def __init__(self,
        prior: np.ndarray,
        model: Callable,
        nb_stim: int,
        nb_actions: int
    ):
        self.prior = prior
        self.model = model
        self.nb_stim = nb_stim
        self.thetas = np.arange(nb_stim, dtype=int)
        self.nb_actions = nb_actions
    
    def update_prior(self, stimulus: int, user_action: int):
        """Updates `self.prior` the distribution over the user's goals.
        """
        self.prior = self.prior * self.model(user_action, stimulus, self.thetas)
        self.prior /= np.sum(self.prior)

    def utility(self, stimulus: int):
        """Computes the utility of `stimulus`:
        in the BIG framework, it is the information gain about the user goal when conditioned on `stimulus`, averaged over the user actions.
        It is the mutual information between the user goal and action, conditioned on the stimulus.
        """
        # compute the entropy of the distribution of user actions conditioned on the user stimulus
        p_yx = np.zeros(self.nb_actions)
        for user_action in range(self.nb_actions):
            p_yx[user_action] = np.sum( self.prior * self.model(user_action, stimulus, self.thetas) )
        H_yx = - np.sum( p_yx * np.log2(p_yx) )

        # compute the entropy of the distribution of user actions conditioned on user goal and stimulus
        q_yx = np.zeros(self.nb_actions)
        for user_action in range(self.nb_actions):
            mod = self.model(user_action, stimulus, self.thetas)
            q_yx[user_action] = np.sum( self.prior * mod * np.log2(mod) )
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
    nb_thetas = 100
    probas = np.ones(nb_thetas) / nb_thetas
    true_prior = Prior(nb_thetas, probas)

    nb_stim = nb_thetas
    nb_actions = 3
    def true_policy(stimulus, theta):
        """Returns the user action given stimulus and goal."""
        return ( 2 * np.heaviside(stimulus - theta, 0.5) ).astype(int)

    user = User(true_prior, true_policy)

    # define the computer
    prior = probas

    def model(action: int, stimulus: int, thetas: Union[int, np.ndarray]):
        """Returns the proba of a user action given a stimulus and user goal.
        
        Note that the probas associated to different goals can be computed in a single pass,
        if `thetas` is a `numpy.ndarray`.
        """
        true_a = true_policy(stimulus, thetas)
        a1 = (true_a + 1) % 3
        a2 = (true_a + 2) % 3

        lamb = 3 / 4
        lamb1 = 1 / 8
        lamb2 = 1 / 8

        tab = action + np.zeros_like(true_a)

        return lamb * (tab == true_a) + lamb1 * (tab == a1) + lamb2 * (tab == a2)

    computer = Computer(prior, model, nb_stim, nb_actions)

    # make the two interact until the user goal has been reached
    rng = np.random.default_rng(seed=None)
    user.pick_goal(rng)
    
    print(f"goal chosen by the user: {user.theta}")
    print()

    stim = rng.choice(computer.nb_stim, p=computer.prior)

    print(f"first stimulus sent by the computer: {stim}")
    print()

    nb_trials = 0
    while stim != user.theta:
        action = user.pick_action(stim)
        stim = computer.pick_stimulus(stim, action)
        #print(f"new stimulus sent by the computer: {stim}")
        #print()
        nb_trials += 1
    
    print(f"user goal guessed by the computer in {nb_trials} trials!")
    print(f"average nb of trials if dichotomy: {np.round(np.log2(100), 2)}")
