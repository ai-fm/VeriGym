# Writing and reading Explicit_policies in PRISM's strategy format (`-exportstrat file:type=actions,states=false`):
# one `state:action` line per state. States and actions are 0-based indices into the UMB model (as in its files);
# actions are choice-action indices, since our UMB files carry no action labels. States without choices are omitted.

"""
    write_policy(path_or_io, p::Explicit_policy)

Write `p` in PRISM's strategy format: a line `s:a` for every state with an action, with 0-based UMB indices.
"""
function write_policy(io::IO, p::Explicit_policy)
    for (s, a) in enumerate(p.state_to_action)
        a == 0 && continue
        println(io, s - 1, ':', a - 1)
    end
end

function write_policy(path::AbstractString, p::Explicit_policy)
    open(io -> write_policy(io, p), path, "w")
    return path
end

"""
    read_policy(path_or_io, nr_states) -> Explicit_policy

Read a policy written by [`write_policy`](@ref) (PRISM strategy format with 0-based state and action indices) for a
model with `nr_states` states. States that are not listed get 0.
"""
function read_policy(io::IO, nr_states::Integer)
    state_to_action = zeros(Int64, nr_states)
    for (number, line) in enumerate(eachline(io))
        line = strip(line)
        isempty(line) && continue
        fields = split(line, ':')
        length(fields) == 2 || error("line $number is not of the form `state:action`: $line")
        s, a = parse(Int64, fields[1]) + 1, parse(Int64, fields[2]) + 1
        1 <= s <= nr_states || error("line $number: state $(s - 1) is outside 0:$(nr_states - 1)")
        a >= 1 || error("line $number: negative action $(a - 1)")
        state_to_action[s] = a
    end
    return Explicit_policy(state_to_action)
end

read_policy(path::AbstractString, nr_states::Integer) = open(io -> read_policy(io, nr_states), path)
