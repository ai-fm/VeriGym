using POMDPs
using MCTS
using Random
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--seed"
        arg_type = Int64
    "--n_iterations"
        arg_type = Int64
end
parsed_args = parse_args(ARGS, s)
seed = get!(parsed_args, "seed", 1)
n_iterations = get!(parsed_args, "n_iterations", 10000)

include("../../../src/verigym/frameworks/juliapomdp/POMDPs_UMB.jl")
import .POMDPs_UMB   # Ours!

t0 = time()
dir = "./output/workflows/ext_tools_files/"
input, output = dir * "model.umb", dir * "policy.txt"
model = POMDPs_UMB.read_umb(input; discount=0.95)
initial_state = support(POMDPs.initialstate(model))[1]
t_reading = time()

solver = MCTSSolver(n_iterations=n_iterations, depth=50, reuse_tree=true, rng=MersenneTwister(seed))
planner = solve(solver, model)

for s in support(POMDPs.initialstate(model))
    _action = action(planner, initial_state)
end
t_solving = time()

policy = POMDPs_UMB.Explicit_policy(planner, model)
POMDPs_UMB.write_policy(output, policy)
t_writing = time()

println("Policy computation complete! (In $(t_writing - t0)s, achieving value $(POMDPs.value(planner, initial_state))")