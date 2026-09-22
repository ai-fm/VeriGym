"""
    Explicit_policy(state_to_action)

Deterministic, state-based policy for a `UMB_MDP` or `UMB_POMDP`: `state_to_action[s]` is the (1-based) action taken
in state `s`, or 0 for states without choices. States and actions use the numbering of the UMB model.
"""
struct Explicit_policy <: Policy
    state_to_action::Vector{Int64}
end

POMDPs.action(p::Explicit_policy, s::Int64) = p.state_to_action[s]

"""
Explicit policy for `u`, where `policy_action(s)` is the action index chosen in state `s`. States without choices
get 0 and terminal states the action of their first choice, without calling `policy_action`. An action that is not
available in a state is replaced by the state's first choice, which is how the model treats it anyway.
"""
function explicit_actions(u::UMB_Model, policy_action)
    state_to_action = zeros(Int64, u.nr_states)
    replaced = 0
    for s in 1:u.nr_states
        available = actions(u, s)
        isempty(available) && continue
        if isterminal(u, s)
            state_to_action[s] = first(available)
            continue
        end
        a = policy_action(s)
        if !(a in available)
            a = first(available)
            replaced += 1
        end
        state_to_action[s] = a
    end
    replaced > 0 && @warn "The policy chose an unavailable action in $replaced states; using their first choice instead."
    return Explicit_policy(state_to_action)
end

"""
    Explicit_policy(policy, m::Union{UMB_MDP,UMB_POMDP})

Tabulate a deterministic `policy` that acts on the states `1:nr_states` of `m` (it is called once per state).
"""
Explicit_policy(policy::Policy, m::UMB_MDP) = explicit_actions(m, s -> action(policy, s))
Explicit_policy(policy::Policy, m::UMB_POMDP) = explicit_actions(m, s -> action(policy, s))

"""
    Explicit_policy(policy, m::MDP, u=UMB_MDP(m))

Tabulate a deterministic `policy` for the original POMDPs.jl MDP `m`, numbered as in `u = UMB_MDP(m)`: state `i` is
`ordered_states(m)[i]` and action `i` is `ordered_actions(m)[i]`. A fresh initial state of `u` gets its only action.
"""
function Explicit_policy(policy::Policy, m::MDP, u::UMB_MDP=UMB_MDP(m))
    S = ordered_states(m)
    u.nr_states in (length(S), length(S) + 1) ||
        throw(ArgumentError("u has $(u.nr_states) states, which does not match the $(length(S)) states of m"))
    return explicit_actions(u, s -> s <= length(S) ? actionindex(m, action(policy, S[s])) : 1)
end
