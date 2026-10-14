from collections import defaultdict

from numpy.typing import NDArray


class RewardFunction:
    """
    Class for reward functions in Verigym environments.
    It is based on dicts in dicts.
    All states and actions are flattened to integers indices.
    ✅ Indexing
    ❌ Slicing

    Example:
    ```
    R_array  # numpy array of shape (n_states, n_actions)
    R = RewardFunction.from_array(R_array)
    s, a = 0, 0
    R[s, a] == R[s][a]  # reward for taking action a in state s
    R[s]
    ```
    """

    R_dict: defaultdict[int, dict[int, float]]
    n_states: int
    n_actions: int

    def __init__(self, n_states: int, n_actions: int, R_dict=None):
        if R_dict is not None:
            # check if structure of R_dict is correct
            assert isinstance(R_dict, defaultdict), (
                f"R_dict must be a defaultdict, is {type(R_dict)}"
            )
            # check for each state, there is a dict of actions
            for s in R_dict:
                assert isinstance(s, int), (
                    f"State index must be an integer, got {type(s)}"
                )
                assert isinstance(R_dict[s], dict), (
                    f"R_dict[{s}] must be a dict, is {type(R_dict[s])}"
                )
                # check for each action, there is a float reward
                for a in R_dict[s]:
                    assert isinstance(a, int), (
                        f"Action index must be an integer, got {type(a)}"
                    )
                    assert isinstance(R_dict[s][a], float), (
                        f"R_dict[{s}][{a}] must be a float, is {type(R_dict[s][a])}"
                    )
            self.R_dict = R_dict
        else:
            self.R_dict = defaultdict(lambda: defaultdict(float))
        self.n_states = n_states
        self.n_actions = n_actions

    def __getitem__(self, idx):
        # Normalize to tuple
        if not isinstance(idx, tuple):
            idx = (idx,)

        if len(idx) == 1:
            (s,) = idx
            return self.R_dict[s]
        if len(idx) == 2:
            s, a = idx
            return self.R_dict[s][a]
        raise IndexError("Too many indices for RewardFunction")

    @classmethod
    def from_array(cls, array: NDArray) -> "RewardFunction":
        """
        Create a RewardFunction from a 2D numpy array.

        Parameters
        ----------
        array : NDArray
            A 2D numpy array of shape (n_states, n_actions) representing the reward function.

        Returns
        -------
        RewardFunction
            An instance of RewardFunction initialized with the values from the array.
        """
        n_states, n_actions = array.shape
        R_dict = defaultdict(dict)
        for s in range(n_states):
            for a in range(n_actions):
                R_dict[s][a] = array[s, a]
        return cls(n_states=n_states, n_actions=n_actions, R_dict=R_dict)

    @classmethod
    def from_dict(cls, R_dict: dict, n_states: int, n_actions: int) -> "RewardFunction":
        for s in R_dict:
            for a in R_dict[s].keys():
                if isinstance(R_dict[s][a], list):
                    if len(R_dict[s][a]) > 1:
                        raise NotImplementedError(
                            "currently cannot handle multiple reward functions"
                        )
                    R_dict[s][a] = R_dict[s][a][0]
        return cls(n_states=n_states, n_actions=n_actions, R_dict=R_dict)

class IntervalRewardFunction(RewardFunction):
    """
    Class for reward functions with interval bounds.
    """
    R_dict: defaultdict[int, dict[int, tuple]]
    n_states: int
    n_actions: int

    def __init__(self, n_states, n_actions, R_dict=None):
        if R_dict is not None:
            # check if structure of R_dict is correct
            assert isinstance(R_dict, defaultdict), (
                f"R_dict must be a defaultdict, is {type(R_dict)}"
            )
            # check for each state, there is a dict of actions
            for s in R_dict:
                assert isinstance(s, int), (
                    f"State index must be an integer, got {type(s)}"
                )
                assert isinstance(R_dict[s], dict), (
                    f"R_dict[{s}] must be a dict, is {type(R_dict[s])}"
                )
                # check for each action, there is a float reward
                for a in R_dict[s]:
                    assert isinstance(a, int), (
                        f"Action index must be an integer, got {type(a)}"
                    )
                    assert isinstance(R_dict[s][a], tuple), (
                        f"R_dict[{s}][{a}] must be a tuple, is {type(R_dict[s][a])}"
                    )
            self.R_dict = R_dict
        else:
            self.R_dict = defaultdict(lambda: defaultdict(float))
        self.n_states = n_states
        self.n_actions = n_actions

    @classmethod
    def from_array(cls, array: NDArray) -> "IntervalRewardFunction":
        """
        Create an IntervalRewardFunction from a 3D numpy array.

        Parameters
        ----------
        array : NDArray
            A 3D numpy array of shape (n_states, n_actions) representing the reward function.

        Returns
        -------
        RewardFunction
            An instance of RewardFunction initialized with the values from the array.
        """
        n_states, n_actions = array.shape[:2]
        R_dict = defaultdict(dict)
        for s in range(n_states):
            for a in range(n_actions):
                R_dict[s][a] = (array[s, a][0], array[s, a][1])
        return cls(n_states=n_states, n_actions=n_actions, R_dict=R_dict)


    @classmethod
    def from_reward_function(cls, reward_function: RewardFunction) -> "IntervalRewardFunction":
        """
        Create an IntervalRewardFunction with point intervals [r, r] from a RewardFunction.
        Keeps the state-action pairs of `reward_function`; all other pairs read as (0.0, 0.0) (see `from_dict`).

        Parameters
        ----------
        reward_function : RewardFunction
            The reward function with point rewards.

        Returns
        -------
        IntervalRewardFunction
            An instance of IntervalRewardFunction where lower bound == upper bound for all rewards.
        """
        R_dict = defaultdict(dict)
        for s, actions in reward_function.R_dict.items():
            for a, reward in actions.items():
                R_dict[s][a] = (reward, reward)
        return cls.from_dict(R_dict, reward_function.n_states, reward_function.n_actions)

    def point_estimate(self) -> RewardFunction:
        """
        Returns the midpoints (lower + upper) / 2 of the reward intervals, used wherever a point estimate is required.
        Keeps the state-action pairs of the intervals, and does not change the intervals. Like the interval of a
        state-action pair without a stored interval (see `from_dict`), its point estimate reads as 0.0.

        Returns
        -------
        RewardFunction
            The point estimate.
        """
        R_dict = defaultdict(lambda: defaultdict(float))
        for s, actions in self.R_dict.items():
            for a, (lb, ub) in actions.items():
                R_dict[s][a] = float((lb + ub) / 2)
        return RewardFunction(self.n_states, self.n_actions, R_dict)

    @classmethod
    def from_dict(cls, R_dict: dict, n_states: int, n_actions: int, fillval=0.0) -> "IntervalRewardFunction":
        """
        Create an IntervalRewardFunction from a dict mapping s -> a -> (lower, upper).
        Only the given state-action pairs are stored; all other pairs read as (fillval, fillval).
        """
        R_sparse_dict = defaultdict(lambda: defaultdict(lambda: (fillval, fillval)))
        for s, R_s in R_dict.items():
            for a, r in R_s.items():
                R_sparse_dict[s][a] = r
        return cls(n_states=n_states, n_actions=n_actions, R_dict=R_sparse_dict)
