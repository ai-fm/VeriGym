from verigym.policy.policy import PolicyClass
from verigym.abstraction.abstractionmapper import AbstractionMapper

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
    def __init__(self, policy, action_map, abstraction_mapper: AbstractionMapper):
        parsed_policy = self._init_policy(policy)
        self.action_label_to_idx = action_map

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
                state = int(line_list[0])-1 # indexing starts at 1 here
                action_label = line_list[1]
                parsed_policy[state] = action_label
        
        return parsed_policy


    def _action_from_policy(self, obs):
        if obs not in self.policy.keys():
            return None # terminal state, no action available
        else:
            action_name = self.policy[obs]
            action_index = self.action_label_to_idx[action_name]
            return action_index
