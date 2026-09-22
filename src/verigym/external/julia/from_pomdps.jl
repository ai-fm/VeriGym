# Converting POMDPs.jl models into the explicit UMB types.
# MDPs keep their numbering: state i is `ordered_states(m)[i]` and action i is `ordered_actions(m)[i]`.
# Converting POMDPs is not implemented yet (see coding/findings.md for the design).

"""
Merge branches to the same target: probabilities are summed and rewards weighted by probability.
Branches with zero probability are dropped. Returns `(targets, probabilities, rewards)`.
"""
function merge_branches(targets::AbstractVector{Int64}, probabilities::AbstractVector{<:Real}, rewards::AbstractVector{<:Real})
    position = Dict{Int64,Int64}()
    merged_targets, merged_probabilities, weighted_rewards = Int64[], Float64[], Float64[]
    for (t, p, r) in zip(targets, probabilities, rewards)
        p > 0 || continue
        i = get!(position, t) do
            push!(merged_targets, t)
            push!(merged_probabilities, 0.0)
            push!(weighted_rewards, 0.0)
            length(merged_targets)
        end
        merged_probabilities[i] += p
        weighted_rewards[i] += p * r
    end
    return merged_targets, merged_probabilities, weighted_rewards ./ merged_probabilities
end

"""
    UMB_MDP(m::MDP) -> UMB_MDP

Convert a discrete POMDPs.jl MDP (with `states`, `stateindex`, `actions` and `actionindex`) into a `UMB_MDP`.
State `i` is `ordered_states(m)[i]` and action `i` is `ordered_actions(m)[i]`.

Branch rewards are `reward(m, s, a, sp)`. Terminal states become zero-reward self-loops.
If `initialstate(m)` is uniform over its support, that support becomes the initial states. Otherwise, a fresh state
`n + 1` (the only initial state) leads to the initial distribution in one step with reward 0. Values from that
state are then `discount(m)` times the expected value of the initial distribution.
"""
function UMB_MDP(m::MDP)
    S = ordered_states(m)
    n = length(S)
    choice_to_branches = UnitRange{Int64}[]
    branch_to_state, branch_to_probability, branch_to_reward = Int64[], Float64[], Float64[]
    state_to_choices = UnitRange{Int64}[]
    choice_to_action = Int64[]

    function add_choice!(action, targets, probabilities, rewards)
        first_branch = length(branch_to_state) + 1
        append!(branch_to_state, targets)
        append!(branch_to_probability, probabilities)
        append!(branch_to_reward, rewards)
        push!(choice_to_branches, first_branch:length(branch_to_state))
        push!(choice_to_action, action)
    end

    for s in S
        first_choice = length(choice_to_action) + 1
        if isterminal(m, s)
            add_choice!(1, [stateindex(m, s)], [1.0], [0.0])
        else
            for a in actions(m, s)
                outcomes = vec([(sp, p) for (sp, p) in weighted_iterator(transition(m, s, a)) if p > 0])
                add_choice!(actionindex(m, a), merge_branches([stateindex(m, sp) for (sp, _) in outcomes],
                    [p for (_, p) in outcomes], [reward(m, s, a, sp) for (sp, _) in outcomes])...)
            end
        end
        push!(state_to_choices, first_choice:length(choice_to_action))
    end

    initial = vec(collect(weighted_iterator(initialstate(m))))  # some distributions iterate as a matrix
    targets, probabilities, _ = merge_branches([stateindex(m, s) for (s, _) in initial], [p for (_, p) in initial],
        zeros(length(initial)))
    isempty(targets) && error("the initial state distribution is empty")
    if all(p -> p ≈ first(probabilities), probabilities)
        initial_states = sort(targets)
        nr_states = n
    else
        add_choice!(1, targets, probabilities, zeros(length(targets)))
        push!(state_to_choices, length(choice_to_action):length(choice_to_action))
        initial_states = [n + 1]
        nr_states = n + 1
    end

    return UMB_MDP(; nr_states, nr_actions=length(ordered_actions(m)), initial_states, choice_to_branches,
        branch_to_state, branch_to_probability, state_to_choices, choice_to_action, branch_to_reward,
        discount=discount(m))
end

UMB_MDP(m::UMB_MDP) = m
