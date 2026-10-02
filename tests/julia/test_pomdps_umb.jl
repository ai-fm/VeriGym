# Tests for the Julia POMDPs_UMB module in src/verigym/external/julia/.
# Does not run automatically, since it requires a Julia installation.
# Instead, can be run manually as: julia tests/julia/test_pomdps_umb.jl

using Test
using POMDPs
using POMDPTools: SparseCat, Deterministic, FunctionPolicy

include("../../src/verigym/external/julia/POMDPs_UMB.jl")
using .POMDPs_UMB
const PU = POMDPs_UMB

"""
Hand-built 4-state MDP (1-based):
- s1: action 1 -> {s2: 0.5 (reward 1), s3: 0.5 (reward 3)}; action 2 -> s2 via two branches (0.3 reward 2, 0.7 reward 4)
- s2: action 2 -> s3 (reward 5)
- s3: zero-reward self-loop (terminal)
- s4: no choices
"""
toy_mdp(; action_labels=["a", "b"]) = UMB_MDP(;
    nr_states=4,
    nr_actions=2,
    initial_states=[1, 2],
    choice_to_branches=[1:2, 3:4, 5:5, 6:6],
    branch_to_state=[2, 3, 2, 2, 3, 3],
    branch_to_probability=[0.5, 0.5, 0.3, 0.7, 1.0, 1.0],
    state_to_choices=[1:2, 3:3, 4:4, 5:4],
    choice_to_action=[1, 2, 2, 1],
    action_labels,
    branch_to_reward=[1.0, 3.0, 2.0, 4.0, 5.0, 0.0],
)

"POMDP version of `toy_mdp()` in which s1 and s2 share an observation."
toy_pomdp() = UMB_POMDP(; (f => getfield(toy_mdp(), f) for f in fieldnames(UMB_MDP))...,
    nr_observations=3, state_to_observations=[1, 1, 2, 3])

"Test that `a` and `b` agree on every field except those in `skip`."
function test_same_fields(a, b; skip=(:discount,))
    @test typeof(a) == typeof(b)
    for f in fieldnames(typeof(a))
        f in skip && continue
        @test getfield(a, f) == getfield(b, f)
    end
end

# A small POMDPs.jl MDP to convert: going from :x reaches :y via two outcomes (to be merged) or :goal (terminal).
struct ToyMDP <: MDP{Symbol,Symbol}
    initial::SparseCat
end
const TOY_STATES = [:x, :y, :goal]
POMDPs.states(::ToyMDP) = TOY_STATES
POMDPs.stateindex(::ToyMDP, s::Symbol) = findfirst(==(s), TOY_STATES)
POMDPs.actions(::ToyMDP) = [:go, :stay]
POMDPs.actionindex(::ToyMDP, a::Symbol) = a == :go ? 1 : 2
function POMDPs.transition(::ToyMDP, s::Symbol, a::Symbol)
    (s == :goal || a == :stay) && return Deterministic(s)
    s == :x && return SparseCat([:y, :y, :goal], [0.25, 0.25, 0.5])
    return Deterministic(:goal)
end
POMDPs.reward(::ToyMDP, s::Symbol, a::Symbol, sp::Symbol) = sp == :goal && s != :goal ? 10.0 : (a == :go ? -1.0 : 0.0)
POMDPs.isterminal(::ToyMDP, s::Symbol) = s == :goal
POMDPs.initialstate(m::ToyMDP) = m.initial
POMDPs.discount(::ToyMDP) = 0.9

@testset "POMDPs_UMB" begin

@testset "Model semantics" begin
    m = toy_mdp()
    @test collect(actions(m, 1)) == [1, 2]
    @test collect(actions(m, 2)) == [2]
    @test isempty(actions(m, 4))

    d = transition(m, 1, 1)
    @test pdf(d, 2) == 0.5 && pdf(d, 3) == 0.5
    merged = transition(m, 1, 2)               # duplicate targets are merged
    @test collect(support(merged)) == [2]
    @test pdf(merged, 2) ≈ 1.0

    @test reward(m, 1, 1) ≈ 2.0
    @test reward(m, 1, 2) ≈ 3.4
    @test reward(m, 1, 1, 3) ≈ 3.0
    @test reward(m, 1, 2, 2) ≈ 3.4             # probability-weighted over both branches to s2
    @test reward(m, 1, 2, 3) == 0.0            # no branch to s3

    # An unavailable action behaves like the state's first choice
    @test pdf(transition(m, 2, 1), 3) == 1.0
    @test reward(m, 2, 1) == reward(m, 2, 2) == 5.0
    # A state without choices is a zero-reward self-loop
    @test pdf(transition(m, 4, 1), 4) == 1.0
    @test reward(m, 4, 1) == 0.0

    @test isterminal(m, 3) && isterminal(m, 4)
    @test !isterminal(m, 1) && !isterminal(m, 2)
    init = initialstate(m)
    @test pdf(init, 1) == pdf(init, 2) == 0.5
    @test discount(m) == 0.95

    p = UMB_POMDP(m)
    @test p.nr_observations == 4 && p.state_to_observations == 1:4
    @test pdf(observation(p, 1, 3), 3) == 1.0
    test_same_fields(UMB_MDP(p), m; skip=())
end

@testset "Binary helpers" begin
    bits = BitVector([isodd(i ÷ 3) for i in 1:70])
    bytes = PU.write_bitset(bits)
    @test length(bytes) == 16                  # padded to a multiple of 64 bits
    @test PU.read_bitset(bytes, 70) == bits

    ranges = [1:2, 3:2, 3:5]                   # includes an empty range
    csr = PU.ranges_to_csr(ranges)
    @test csr == UInt64[0, 2, 2, 5]
    @test PU.csr_to_ranges(csr) == ranges
    @test_throws ErrorException PU.ranges_to_csr([1:2, 4:5])

    @test PU.read_numeric(PU.write_values([0.25, -1.5]), Dict("type" => "double")) == [0.25, -1.5]
    rational128 = PU.write_values(UInt64[reinterpret(UInt64, -3), 4])
    @test PU.read_numeric(rational128, Dict("type" => "rational", "size" => 128)) == [-0.75]
    # 256 bits: two-word signed numerator, then two-word unsigned denominator (-3/4 and 2^64/2^62)
    rational256 = PU.write_values(UInt64[reinterpret(UInt64, -3), typemax(UInt64), 4, 0, 0, 1, UInt64(1) << 62, 0])
    @test PU.read_numeric(rational256, Dict("type" => "rational", "size" => 256)) == [-0.75, 4.0]
end

@testset "UMB round trip" begin
    mktempdir() do dir
        m = toy_mdp()
        for compression in (:gzip, :xz, :none)
            path = write_umb(joinpath(dir, "toy_$compression.umb"), m; compression)
            r = read_umb(path; discount=0.9)
            test_same_fields(r, m)
            @test r.discount == 0.9
        end

        p = toy_pomdp()
        r = read_umb(write_umb(joinpath(dir, "pomdp.umb"), p))
        @test r isa UMB_POMDP
        test_same_fields(r, p)

        # Distinct observations make the model fully observable
        @test read_umb(write_umb(joinpath(dir, "observable.umb"), UMB_POMDP(m))) isa UMB_MDP

        not_umb = joinpath(dir, "not_umb.umb")
        open(io -> PU.write_tar(io, ["foo.bin" => UInt8[1, 2, 3]]), not_umb, "w")
        @test_throws ErrorException read_umb(not_umb)
    end
end

@testset "Conversion from POMDPs.jl" begin
    targets, probabilities, rewards = PU.merge_branches([1, 2, 1, 3], [0.2, 0.3, 0.5, 0.0], [1.0, 2.0, 3.0, 9.0])
    @test targets == [1, 2]                    # zero-probability branch to 3 is dropped
    @test probabilities ≈ [0.7, 0.3]
    @test rewards ≈ [(0.2 * 1.0 + 0.5 * 3.0) / 0.7, 2.0]

    u = UMB_MDP(ToyMDP(SparseCat([:x, :y], [0.5, 0.5])))
    @test u.nr_states == 3 && u.initial_states == [1, 2]
    @test u.action_labels == ["go", "stay"]
    @test discount(u) == 0.9
    d = transition(u, 1, 1)                    # :x, :go
    @test length(support(d)) == 2              # the two outcomes to :y are merged
    @test pdf(d, 2) ≈ 0.5 && pdf(d, 3) ≈ 0.5
    @test reward(u, 1, 1, 2) == -1.0 && reward(u, 1, 1, 3) == 10.0
    @test pdf(transition(u, 2, 2), 2) == 1.0   # :y, :stay
    @test isterminal(u, 3) && reward(u, 3, 1) == 0.0

    # A non-uniform initial distribution is reached from a fresh initial state
    u = UMB_MDP(ToyMDP(SparseCat([:x, :y], [0.25, 0.75])))
    @test u.nr_states == 4 && u.initial_states == [4]
    @test collect(actions(u, 4)) == [1]
    @test pdf(transition(u, 4, 1), 1) ≈ 0.25 && pdf(transition(u, 4, 1), 2) ≈ 0.75
    @test reward(u, 4, 1) == 0.0
end

@testset "Explicit policies" begin
    m = toy_mdp()
    p = Explicit_policy(FunctionPolicy(s -> 2), m)
    @test p.state_to_action == [2, 2, 1, 0]    # terminal s3 gets its first action, s4 has none
    @test p.action_labels == ["a", "b"]
    @test action(p, 1) == 2

    # s2 has no action 1, so it gets its first choice instead
    p = @test_logs (:warn, r"unavailable") Explicit_policy(FunctionPolicy(s -> 1), m)
    @test p.state_to_action == [1, 2, 1, 0]

    toy = ToyMDP(SparseCat([:x, :y], [0.25, 0.75]))
    p = Explicit_policy(FunctionPolicy(s -> s == :x ? :stay : :go), toy)
    @test p.state_to_action == [2, 1, 1, 1]    # fresh initial state 4 gets its only action
    @test p.action_labels == ["go", "stay"]
end

@testset "Policy I/O" begin
    m = toy_mdp()
    p = Explicit_policy([2, 2, 1, 0], m.action_labels)
    io = IOBuffer()
    write_policy(io, p)
    text = String(take!(io))
    @test text == "0=b\n1=b\n2=a\n"            # 0-based states, s4 without choices omitted
    @test read_policy(IOBuffer(text), m).state_to_action == p.state_to_action
    mktempdir() do dir
        path = write_policy(joinpath(dir, "policy.txt"), p)
        @test read_policy(path, m).state_to_action == p.state_to_action
    end

    # Without labels, 0-based action indices are written and read
    unlabeled = toy_mdp(action_labels=String[])
    io = IOBuffer()
    @test_logs (:warn, r"no action labels") write_policy(io, Explicit_policy([2, 2, 1, 0]))
    text = String(take!(io))
    @test text == "0=1\n1=1\n2=0\n"
    @test read_policy(IOBuffer(text), unlabeled).state_to_action == [2, 2, 1, 0]

    # Actions with an empty label (Storm's terminal self-loops) are omitted
    io = IOBuffer()
    write_policy(io, Explicit_policy([2, 2, 1, 0], ["", "b"]))
    @test String(take!(io)) == "0=b\n1=b\n"

    @test_throws ErrorException read_policy(IOBuffer("0=c\n"), m)   # unknown label
    @test_throws ErrorException read_policy(IOBuffer("9=a\n"), m)   # state out of range
    @test_throws ErrorException read_policy(IOBuffer("0:a\n"), m)   # malformed line
end

end
