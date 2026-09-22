# Reading and writing UMB files (spec v1.0, https://github.com/pmc-tools/umb/blob/main/spec.md) as UMB_MDPs and UMB_POMDPs.
# Files use 0-based indices and little-endian values; the models are 1-based.
# The discount factor is not part of UMB: it is neither written nor read.

## Binary encoding --------------------------------------------------------------------------------

"Decode a file of fixed-size little-endian values of type `T`."
function read_values(::Type{T}, bytes::AbstractVector{UInt8}) where {T}
    length(bytes) % sizeof(T) == 0 || error("file size $(length(bytes)) is not a multiple of sizeof($T)")
    return ltoh.(collect(reinterpret(T, bytes)))
end

write_values(values::AbstractVector) = collect(reinterpret(UInt8, htol.(values)))

"Decode a bool file with `n` entries: bit `i` holds entry `i`, padded to a multiple of 64 bits."
function read_bitset(bytes::AbstractVector{UInt8}, n::Integer)
    length(bytes) >= cld(n, 8) || error("bitset of $(length(bytes)) bytes is too small for $n entries")
    return BitVector([(bytes[i ÷ 8 + 1] >> (i % 8)) & 0x01 == 0x01 for i in 0:n-1])
end

function write_bitset(bits::AbstractVector{Bool})
    bytes = zeros(UInt8, cld(length(bits), 64) * 8)
    for i in findall(bits)
        bytes[(i - 1) ÷ 8 + 1] |= 0x01 << ((i - 1) % 8)
    end
    return bytes
end

"Convert a CSR file (0-based offsets) to 1-based index ranges."
function csr_to_ranges(csr::AbstractVector{UInt64})
    (!isempty(csr) && csr[1] == 0 && issorted(csr)) || error("invalid CSR mapping: $csr")
    return [Int(csr[i])+1:Int(csr[i+1]) for i in 1:length(csr)-1]
end

"Convert 1-based index ranges to a CSR file; the ranges must be contiguous and start at 1."
function ranges_to_csr(ranges::AbstractVector{UnitRange{Int64}})
    csr = zeros(UInt64, length(ranges) + 1)
    for (i, r) in enumerate(ranges)
        first(r) == csr[i] + 1 || error("ranges are not contiguous from 1: $ranges")
        csr[i+1] = last(r)
    end
    return csr
end

"Signed or unsigned integer from little-endian 64-bit words."
function words_to_integer(words::AbstractVector{UInt64}, signed::Bool)
    value = sum(BigInt(w) << (64 * (k - 1)) for (k, w) in enumerate(words))
    bits = 64 * length(words)
    return signed && (words[end] >> 63) == 1 ? value - (BigInt(1) << bits) : value
end

"""
Decode continuous numeric values declared with `type` (a type object from `index.json`) as `Float64`.
Supports `double` and `rational` of any size that is a multiple of 128 (signed numerator, then unsigned denominator).
"""
function read_numeric(bytes::AbstractVector{UInt8}, type::AbstractDict)
    name = type["type"]
    if name == "double"
        return read_values(Float64, bytes)
    elseif name == "rational"
        size = get(type, "size", 128)
        size % 128 == 0 || error("rational size $size is not a multiple of 128")
        words = reshape(read_values(UInt64, bytes), size ÷ 64, :)
        half = size ÷ 128
        if half == 1
            return [reinterpret(Int64, w[1]) / w[2] for w in eachcol(words)]
        end
        return [Float64(words_to_integer(w[1:half], true) // words_to_integer(w[half+1:end], false)) for w in eachcol(words)]
    end
    error("values of type \"$name\" are not supported")
end

## Archive ----------------------------------------------------------------------------------------

"`:gzip`, `:xz` or `:none`, from the magic bytes."
function detect_compression(bytes::AbstractVector{UInt8})
    length(bytes) >= 3 && bytes[1:3] == [0x1f, 0x8b, 0x08] && return :gzip
    length(bytes) >= 6 && bytes[1:6] == [0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00] && return :xz
    return :none
end

function decompress(bytes::Vector{UInt8})
    kind = detect_compression(bytes)
    kind == :gzip && return transcode(GzipDecompressor, bytes)
    kind == :xz && return transcode(XzDecompressor, bytes)
    return bytes
end

function compress(bytes::Vector{UInt8}, kind::Symbol)
    kind == :gzip && return transcode(GzipCompressor, bytes)
    kind == :xz && return transcode(XzCompressor, bytes)
    kind == :none && return bytes
    throw(ArgumentError("unknown compression $kind (use :gzip, :xz or :none)"))
end

"All regular files of a tar, keyed by their path."
function read_tar(bytes::Vector{UInt8})
    files = Dict{String,Vector{UInt8}}()
    mktempdir() do dir
        Tar.extract(IOBuffer(bytes), dir; set_permissions=false)
        for (root, _, names) in walkdir(dir), name in names
            path = joinpath(root, name)
            files[join(splitpath(relpath(path, dir)), "/")] = read(path)
        end
    end
    return files
end

"""
Write a ustar archive of regular files in the given order (`Tar.create` sorts the paths, but `index.json`
should be the first entry).
"""
function write_tar(io::IO, entries::AbstractVector{<:Pair{String,Vector{UInt8}}})
    octal(value, width) = Vector{UInt8}(string(value; base=8, pad=width - 1)) # width - 1 digits + NUL
    for (path, data) in entries
        ncodeunits(path) <= 100 || error("path too long for a ustar header: $path")
        header = zeros(UInt8, 512)
        place!(offset, field) = copyto!(header, offset + 1, field, 1, length(field))
        place!(0, Vector{UInt8}(path))
        place!(100, octal(0o644, 8))                    # mode
        place!(108, octal(0, 8))                        # uid
        place!(116, octal(0, 8))                        # gid
        place!(124, octal(length(data), 12))            # size
        place!(136, octal(0, 12))                       # mtime
        place!(148, fill(UInt8(' '), 8))                # checksum, computed with spaces
        place!(156, [UInt8('0')])                       # regular file
        place!(257, Vector{UInt8}("ustar\0" * "00"))    # magic + version
        place!(148, [octal(sum(Int, header), 7); UInt8(' ')])
        write(io, header, data, zeros(UInt8, mod(-length(data), 512)))
    end
    write(io, zeros(UInt8, 1024))
end

## Import -----------------------------------------------------------------------------------------

"Error unless the index describes a discrete-time model with at most one player and probabilities."
function check_supported(ts::AbstractDict)
    ts["time"] == "discrete" || error("only discrete-time models are supported, not \"$(ts["time"])\" time")
    ts["#players"] <= 1 || error("games (#players = $(ts["#players"])) are not supported")
    haskey(ts, "branch-probability-type") || error("models without branch probabilities are not supported")
    if ts["#observations"] > 0
        get(ts, "observations-apply-to", nothing) == "states" ||
            error("only observations on states are supported, not on \"$(get(ts, "observations-apply-to", nothing))\"")
        haskey(ts, "observation-probability-type") && error("only deterministic observations are supported")
    end
end

function check_length(name::AbstractString, values::AbstractVector, n::Integer)
    length(values) == n || error("$name has $(length(values)) entries, expected $n")
end

"1-based ranges from the CSR file `name`; if absent, index `i` maps to `i:i`."
function read_ranges(files::AbstractDict, name::AbstractString, n::Integer)
    ranges = haskey(files, name) ? csr_to_ranges(read_values(UInt64, files[name])) : [i:i for i in 1:n]
    check_length(name, ranges, n)
    return ranges
end

"""
Action of every choice, and the number of actions. Falls back to numbering the choices within each state if the
file has no choice actions or a state has several choices with the same action.
"""
function read_choice_actions(files::AbstractDict, ts::AbstractDict, state_to_choices, nr_choices::Integer)
    nr_actions = Int(ts["#choice-actions"])
    if nr_actions > 0
        name = "actions/choices/values.bin"
        actions = haskey(files, name) ? Int.(read_values(UInt32, files[name])) .+ 1 : ones(Int64, nr_choices)
        check_length(name, actions, nr_choices)
        all(allunique(view(actions, r)) for r in state_to_choices) && return actions, nr_actions
        @warn "Some states have several choices with the same action; numbering the choices of each state instead."
    end
    actions = [c - first(r) + 1 for r in state_to_choices for c in r]
    return actions, maximum(length, state_to_choices; init=0)
end

"Observation of every state and the number of observations, or `nothing` if the model has no observations."
function read_observations(files::AbstractDict, ts::AbstractDict, nr_states::Integer)
    nr_observations = Int(ts["#observations"])
    nr_observations == 0 && return nothing
    name = "observations/states/values.bin"
    haskey(files, "observations/states/distribution-mapping.bin") && error("only deterministic observations are supported")
    observations = haskey(files, name) ? Int.(read_values(UInt64, files[name])) .+ 1 : ones(Int64, nr_states)
    check_length(name, observations, nr_states)
    all(in(1:nr_observations), observations) || error("$name contains observations outside 0:$(nr_observations - 1)")
    return observations, nr_observations
end

"Identifier of the reward annotation to use: `reward` (identifier or alias), or the only one if `nothing`."
function select_reward(rewards::AbstractDict, reward::Union{Nothing,AbstractString})
    if reward === nothing
        isempty(rewards) && return nothing
        length(rewards) == 1 && return only(keys(rewards))
        error("the model has several rewards ($(join(sort(collect(keys(rewards))), ", "))); select one with `reward`")
    end
    haskey(rewards, reward) && return reward
    for (id, annotation) in rewards
        get(annotation, "alias", nothing) == reward && return id
    end
    error("the model has no reward \"$reward\"")
end

"Reward of every branch: the selected reward's state, choice and branch values added together."
function read_branch_rewards(files::AbstractDict, index::AbstractDict, reward, state_to_choices, choice_to_branches, nr_branches::Integer)
    rewards = get(get(index, "annotations", Dict()), "rewards", Dict())
    branch_rewards = zeros(nr_branches)
    id = select_reward(rewards, reward)
    id === nothing && return branch_rewards
    for entity in rewards[id]["applies-to"]
        folder = "annotations/rewards/$id/$entity/"
        haskey(files, folder * "distribution-mapping.bin") && error("stochastic rewards are not supported")
        haskey(files, folder * "values.bin") || continue  # absent: all zero
        values = read_numeric(files[folder * "values.bin"], rewards[id]["type"])
        if entity == "states"
            check_length(folder * "values.bin", values, length(state_to_choices))
            for (s, choices) in enumerate(state_to_choices), c in choices
                branch_rewards[choice_to_branches[c]] .+= values[s]
            end
        elseif entity == "choices"
            check_length(folder * "values.bin", values, length(choice_to_branches))
            for (c, branches) in enumerate(choice_to_branches)
                branch_rewards[branches] .+= values[c]
            end
        elseif entity == "branches"
            check_length(folder * "values.bin", values, nr_branches)
            branch_rewards .+= values
        else
            error("rewards on $entity are not supported")
        end
    end
    return branch_rewards
end

"""
    read_umb(path; discount=0.95, reward=nothing) -> Union{UMB_MDP,UMB_POMDP}

Read a UMB file (plain, gzip- or xz-compressed) with deterministic state observations, or without observations.
The result is a `UMB_MDP` if the model is fully observable (no observations, or a distinct observation for every
state; UMB also observes the initial state), and a `UMB_POMDP` otherwise.
UMB has no discount factor, so it is given by `discount`. If the file has
several rewards, `reward` selects one by identifier or alias. State and choice rewards are moved onto branches.
Labels, atomic propositions and valuations are ignored.
"""
function read_umb(path::AbstractString; discount::Real=DEFAULT_DISCOUNT, reward::Union{Nothing,AbstractString}=nothing)
    files = read_tar(decompress(read(path)))
    haskey(files, "index.json") || error("$path is not a UMB file: index.json is missing")
    index = JSON3.read(files["index.json"], Dict{String,Any})
    ts = index["transition-system"]
    check_supported(ts)
    nr_states, nr_choices, nr_branches = Int(ts["#states"]), Int(ts["#choices"]), Int(ts["#branches"])

    state_to_choices = read_ranges(files, "state-to-choices.bin", nr_states)
    choice_to_branches = read_ranges(files, "choice-to-branches.bin", nr_choices)
    last(last(state_to_choices)) == nr_choices || error("state-to-choices.bin does not cover $nr_choices choices")
    last(last(choice_to_branches)) == nr_branches || error("choice-to-branches.bin does not cover $nr_branches branches")

    branch_to_state = Int.(read_values(UInt64, files["branch-to-target.bin"])) .+ 1
    check_length("branch-to-target.bin", branch_to_state, nr_branches)
    all(in(1:nr_states), branch_to_state) || error("branch-to-target.bin contains states outside 0:$(nr_states - 1)")
    branch_to_probability = read_numeric(files["branch-to-probability.bin"], ts["branch-probability-type"])
    check_length("branch-to-probability.bin", branch_to_probability, nr_branches)

    initial_states = haskey(files, "state-is-initial.bin") ? findall(read_bitset(files["state-is-initial.bin"], nr_states)) : Int64[]
    choice_to_action, nr_actions = read_choice_actions(files, ts, state_to_choices, nr_choices)
    observations = read_observations(files, ts, nr_states)
    branch_to_reward = read_branch_rewards(files, index, reward, state_to_choices, choice_to_branches, nr_branches)

    fields = (; nr_states, nr_actions, initial_states, choice_to_branches, branch_to_state, branch_to_probability,
        state_to_choices, choice_to_action, branch_to_reward, discount=Float64(discount))
    if observations === nothing || allunique(first(observations))
        return UMB_MDP(; fields...)
    end
    state_to_observations, nr_observations = observations
    return UMB_POMDP(; fields..., nr_observations, state_to_observations)
end

## Export -----------------------------------------------------------------------------------------

const DOUBLE_TYPE = Dict("type" => "double", "size" => 64)

"Observation entries of the `transition-system` index, and the observation files."
observation_index(m::UMB_POMDP) = Dict("#observations" => m.nr_observations, "observations-apply-to" => "states")
observation_index(::UMB_MDP) = Dict("#observations" => 0)
observation_files(m::UMB_POMDP) = ["observations/states/values.bin" => write_values(UInt64.(m.state_to_observations .- 1))]
observation_files(::UMB_MDP) = Pair{String,Vector{UInt8}}[]

"The `index.json` content for `m`."
function umb_index(m::UMB_Model)
    return Dict(
        "format-version" => 1,
        "format-revision" => 0,
        "file-data" => Dict(
            "tool" => "POMDPs_UMB.jl",
            "tool-version" => string(pkgversion(@__MODULE__)),
            "creation-date" => round(Int, time())),
        "transition-system" => Dict(
            "time" => "discrete",
            "#players" => 1,
            "#states" => m.nr_states,
            "#initial-states" => length(unique(m.initial_states)),
            "#choices" => length(m.choice_to_branches),
            "#choice-actions" => m.nr_actions,
            "#branches" => length(m.branch_to_state),
            "#branch-actions" => 0,
            "branch-probability-type" => DOUBLE_TYPE,
            observation_index(m)...),
        "annotations" => Dict("rewards" => Dict(REWARD_ID => Dict("applies-to" => ["branches"], "type" => DOUBLE_TYPE))))
end

"The files of the UMB archive for `m`, starting with `index.json`."
function umb_files(m::UMB_Model)
    initial = falses(m.nr_states)
    initial[m.initial_states] .= true
    return [
        "index.json" => Vector{UInt8}(JSON3.write(umb_index(m))),
        "state-to-choices.bin" => write_values(ranges_to_csr(m.state_to_choices)),
        "state-is-initial.bin" => write_bitset(initial),
        "choice-to-branches.bin" => write_values(ranges_to_csr(m.choice_to_branches)),
        "branch-to-target.bin" => write_values(UInt64.(m.branch_to_state .- 1)),
        "branch-to-probability.bin" => write_values(m.branch_to_probability),
        "actions/choices/values.bin" => write_values(UInt32.(m.choice_to_action .- 1)),
        "annotations/rewards/$REWARD_ID/branches/values.bin" => write_values(m.branch_to_reward),
        observation_files(m)...,
    ]
end

"""
    write_umb(path, m::Union{UMB_MDP,UMB_POMDP}; compression=:gzip)

Write `m` as a UMB file, with rewards on branches (an MDP is written without observations). `compression` is `:gzip` (as Storm), `:xz` or `:none`.
The discount factor is not written, since UMB has no place for it.
"""
function write_umb(path::AbstractString, m::UMB_Model; compression::Symbol=:gzip)
    io = IOBuffer()
    write_tar(io, umb_files(m))
    write(path, compress(take!(io), compression))
    return path
end
