Base.@kwdef struct UMB_POMDP <: POMDP{Int64, Int64, Int64}
    nr_states::Int64
    nr_actions::Int64
    nr_observations::Int64
    initial_states::Vector{Int64}

    choice_to_branches::Vector{UnitRange{Int64}}
    branch_to_state::Vector{Int64}
    branch_to_probability::Vector{Float64}

    state_to_choices::Vector{UnitRange{Int64}}
    choice_to_action::Vector{Int64}

    state_to_observations::Vector{Int64}
    branch_to_reward::Vector{Float64}

    discount::Float64 = DEFAULT_DISCOUNT  # not part of UMB
end

"Fully observable counterpart of `UMB_POMDP`: the same fields without observations."
Base.@kwdef struct UMB_MDP <: MDP{Int64, Int64}
    nr_states::Int64
    nr_actions::Int64
    initial_states::Vector{Int64}

    choice_to_branches::Vector{UnitRange{Int64}}
    branch_to_state::Vector{Int64}
    branch_to_probability::Vector{Float64}

    state_to_choices::Vector{UnitRange{Int64}}
    choice_to_action::Vector{Int64}

    branch_to_reward::Vector{Float64}

    discount::Float64 = DEFAULT_DISCOUNT  # not part of UMB
end

"Either explicit model; everything except observations is shared."
const UMB_Model = Union{UMB_MDP, UMB_POMDP}

"Drop the observations of `m`. Only equivalent to `m` if every state has its own observation."
UMB_MDP(m::UMB_POMDP) = UMB_MDP(; (f => getfield(m, f) for f in fieldnames(UMB_MDP))...)

"Fully observable POMDP version of `m`: every state is its own observation."
UMB_POMDP(m::UMB_MDP) = UMB_POMDP(; (f => getfield(m, f) for f in fieldnames(UMB_MDP))...,
    nr_observations=m.nr_states, state_to_observations=collect(1:m.nr_states))

# POMDPs.jl interface, shared by UMB_MDP and UMB_POMDP except for observations. All indices are 1-based.
# Transitions and rewards are defined for every action: an action that has no choice in a state behaves like that
# state's first choice, and a state without choices is a zero-reward self-loop.

POMDPs.states(m::UMB_Model) = 1:m.nr_states
POMDPs.actions(m::UMB_Model) = 1:m.nr_actions
"Actions of the choices of state `s`."
POMDPs.actions(m::UMB_Model, s::Int64) = view(m.choice_to_action, m.state_to_choices[s])
POMDPs.observations(m::UMB_POMDP) = 1:m.nr_observations

POMDPs.stateindex(::UMB_Model, s::Int64) = s
POMDPs.actionindex(::UMB_Model, a::Int64) = a
POMDPs.obsindex(::UMB_POMDP, o::Int64) = o

POMDPs.discount(m::UMB_Model) = m.discount

"""
Index of the (first) choice of state `s` that takes action `a`. Falls back to the first choice of `s` if `a` is not
available, and returns `nothing` if `s` has no choices.
"""
function choice_index(m::UMB_Model, s::Int64, a::Int64)
    choices = m.state_to_choices[s]
    isempty(choices) && return nothing
    for c in choices
        m.choice_to_action[c] == a && return c
    end
    return first(choices)
end

function POMDPs.transition(m::UMB_Model, s::Int64, a::Int64)
    c = choice_index(m, s, a)
    c === nothing && return SparseCat([s], [1.0])
    branches = m.choice_to_branches[c]
    targets = view(m.branch_to_state, branches)
    probabilities = view(m.branch_to_probability, branches)
    allunique(targets) && return SparseCat(targets, probabilities)
    return merge_duplicates(targets, probabilities)
end

"Sum the probabilities of duplicate targets (POMDPTools' `pdf(::SparseCat, x)` only returns the first match)."
function merge_duplicates(targets::AbstractVector{Int64}, probabilities::AbstractVector{Float64})
    unique_targets = unique(targets)
    merged = [sum(p for (t, p) in zip(targets, probabilities) if t == u) for u in unique_targets]
    return SparseCat(unique_targets, merged)
end

POMDPs.observation(m::UMB_POMDP, ::Int64, sp::Int64) = Deterministic(m.state_to_observations[sp])
POMDPs.initialobs(m::UMB_POMDP, s::Int64) = Deterministic(m.state_to_observations[s])

"Expected reward of taking action `a` in state `s`."
function POMDPs.reward(m::UMB_Model, s::Int64, a::Int64)
    c = choice_index(m, s, a)
    c === nothing && return 0.0
    return sum(m.branch_to_probability[b] * m.branch_to_reward[b] for b in m.choice_to_branches[c]; init=0.0)
end

"Reward of the branch from `s` via `a` to `sp` (probability-weighted if several branches lead to `sp`, 0 if none)."
function POMDPs.reward(m::UMB_Model, s::Int64, a::Int64, sp::Int64)
    c = choice_index(m, s, a)
    c === nothing && return 0.0
    weighted_reward, probability = 0.0, 0.0
    for b in m.choice_to_branches[c]
        if m.branch_to_state[b] == sp
            weighted_reward += m.branch_to_probability[b] * m.branch_to_reward[b]
            probability += m.branch_to_probability[b]
        end
    end
    return probability > 0 ? weighted_reward / probability : 0.0
end

"Uniform over the initial states; use a fresh initial state to encode other distributions."
function POMDPs.initialstate(m::UMB_Model)
    n = length(m.initial_states)
    return SparseCat(m.initial_states, fill(1 / n, n))
end

"A state is terminal if all its branches are zero-reward self-loops."
function POMDPs.isterminal(m::UMB_Model, s::Int64)
    for c in m.state_to_choices[s], b in m.choice_to_branches[c]
        (m.branch_to_state[b] == s && iszero(m.branch_to_reward[b])) || return false
    end
    return true
end
