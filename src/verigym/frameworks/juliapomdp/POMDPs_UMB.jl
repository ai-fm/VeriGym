module POMDPs_UMB

using POMDPs
using POMDPTools: SparseCat, Deterministic, ordered_states, ordered_actions, weighted_iterator
using JSON3
using CodecZlib: GzipCompressor, GzipDecompressor
using CodecXz: XzCompressor, XzDecompressor
using Tar

# Conventions this package uses to encode POMDPs.jl concepts that UMB lacks (see coding/findings.md).
const REWARD_ID = "reward"              # identifier of the exported reward annotation
const DEFAULT_DISCOUNT = 0.95           # UMB carries no discount; used unless given

# Explicit MDP and POMDP mirroring the UMB arrays, with the POMDPs.jl interface
include("UMB_POMDP.jl")
# Reading and writing UMB files as UMB_MDPs and UMB_POMDPs
include("io_umb.jl")
# Converting POMDPs.jl models into UMB_MDPs
include("from_pomdps.jl")
# Deterministic state-based policies as tables
include("explicit_policy.jl")
# Writing and reading them in PRISM's strategy format
include("export_policy.jl")

export UMB_MDP, UMB_POMDP, read_umb, write_umb, Explicit_policy, write_policy, read_policy

end
