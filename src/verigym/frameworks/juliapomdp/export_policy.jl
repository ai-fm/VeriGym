# Writing and reading Explicit_policies in PRISM's strategy format (`-exportstrat file:type=actions,states=false`):
# one `state=action` line per state, with 0-based state indices into the UMB model (as in the UMB file itself and
# PRISM's output) and action labels, as read by VeriGym's `PrismPolicy`; the models in memory stay 1-based.
# States without choices, and states whose action is unlabeled (Storm's terminal self-loops), are omitted.
# Models without action labels use 0-based action indices as labels.

"Label written for action `a` of `p`: its action label, or the 0-based index if `p` has no labels."
policy_label(p::Explicit_policy, a::Integer) = isempty(p.action_labels) ? string(a - 1) : p.action_labels[a]

"""
    write_policy(path_or_io, p::Explicit_policy)

Write `p` in PRISM's strategy format: a line `s=label` for every state with a labelled action, with 0-based states.
"""
function write_policy(io::IO, p::Explicit_policy)
    isempty(p.action_labels) && @warn "The policy has no action labels; writing 0-based model action indices instead."
    for (s, a) in enumerate(p.state_to_action)
        a == 0 && continue
        label = policy_label(p, a)
        isempty(label) && continue
        println(io, s - 1, '=', label)
    end
end

function write_policy(path::AbstractString, p::Explicit_policy)
    open(io -> write_policy(io, p), path, "w")
    return path
end

"""
    read_policy(path_or_io, m::Union{UMB_MDP,UMB_POMDP}) -> Explicit_policy

Read a policy written by [`write_policy`](@ref) (PRISM strategy format with 0-based states and action labels) for the
model `m`. Labels are matched to the action labels of `m`, or read as 0-based action indices if `m` has none.
States that are not listed get 0.
"""
function read_policy(io::IO, m::UMB_Model)
    state_to_action = zeros(Int64, m.nr_states)
    for (number, line) in enumerate(eachline(io))
        line = strip(line)
        isempty(line) && continue
        fields = split(line, '=')
        length(fields) == 2 || error("line $number is not of the form `state=action`: $line")
        s = parse(Int64, fields[1]) + 1
        1 <= s <= m.nr_states || error("line $number: state $(s - 1) is outside 0:$(m.nr_states - 1)")
        if isempty(m.action_labels)
            a = parse(Int64, fields[2]) + 1
            1 <= a <= m.nr_actions || error("line $number: action $(a - 1) is outside 0:$(m.nr_actions - 1)")
        else
            a = findfirst(==(fields[2]), m.action_labels)
            a === nothing && error("line $number: unknown action label \"$(fields[2])\"")
        end
        state_to_action[s] = a
    end
    return Explicit_policy(state_to_action, m.action_labels)
end

read_policy(path::AbstractString, m::UMB_Model) = open(io -> read_policy(io, m), path)
