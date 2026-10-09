from verigym.policy.policy import PolicyClass
from verigym.abstraction.abstractionmapper import AbstractionMapper
from warnings import warn

class PrismPolicy(PolicyClass):
    """
    For PrismPolicy, the external policy object is a file.

    Notes
    -----
    1. Assumes that the policy file was generated in PRISM with the option
    states=false.
    Otherwise, states are represented by valuations instead of indices, 
    which cannot be mapped back using the abstraction mapper.
    2. Assumes memoryless deterministic strategies.
    """
    def __init__(self, 
        policy_path: str, 
        abstraction_mapper: AbstractionMapper,
        action_map: dict = None
    ):
        """
        Initializes a policy from PRISM-readable output.
        
        Parameters:
            policy : str
                The path to the PRISM policy output file. Should be a `.tra` file. We currently do not support `.dot` files.
            abstraction_mapper : AbstractionMapper
                Maps the state/action spaces of the PRISM model to the gym environment to deploy the policy on.
            action_map : dict(str: int)
                A mapping from PRISM action label to discrete action index in the gym space.
        """
        parsed_policy = self._init_policy(policy_path)

        # Use action mapping if given, otherwise treat labels as indexes
        if action_map is None:
            self.action_label_to_idx = lambda label: int(label)
        elif isinstance(action_map, dict):
            self.action_label_to_idx = lambda label: action_map[label]

        super().__init__(policy=parsed_policy, abstraction_mapper=abstraction_mapper)
    
    def _init_policy(self, policyfile):
        parsed_policy = {}
        with open(policyfile, "r") as pf:
            policy_str = pf.readlines()
        if policyfile.endswith(".dot"):
            raise NotImplementedError("We currently do not support .dot policies.")
            # These behave a bit weird, it looks like they summarize states with the same behaviors for visualization purposes.

        # remove empty line at the end to avoid parsing errors
        if len(policy_str[-1]) == 0:
            policy_str = policy_str[:-1]

        elif policyfile.endswith(".tra"):
            model_info = policy_str[0].split(" ")
            n_states = int(model_info[0])

            policy_str = policy_str[1:] # the first row just shows n states and n choices
            for line in policy_str:
                line_list = line.strip().split(" ")
                # each line is: state idx, next state idx, prob, action label
                state = int(line_list[0])
                action_label = line_list[3]
                parsed_policy[state] = action_label
            
            assert len(parsed_policy.keys()) == n_states, f"states in policy: {len(parsed_policy.keys())}, states in model: {n_states}"

        else: # action list
            for line in policy_str:
                line_list = line.strip().split("=")
                state = int(line_list[0])
                action_label = line_list[1]
                parsed_policy[state] = action_label
        
        return parsed_policy

    def _action_from_policy(self, obs):       
        obs_enum = self.abstraction_mapper._state_abstraction_map.abstract_to_enum(obs) 
        if obs_enum not in self.policy.keys():
            warn(f"Abstract state {obs} has no action in this policy: state may be terminal or unreachable.")
            action_enum = 0 # default
        else:
            prism_action = self.policy[obs_enum]
            action_enum = self.action_label_to_idx(prism_action)
        action = self.abstraction_mapper._action_abstraction_map.enum_to_abstract(action_enum)
        return action
