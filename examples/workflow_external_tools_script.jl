using POMDPs
using MCTS

include("../src/verigym/external/julia/POMDPs_UMB.jl")
import .POMDPs_UMB   # Ours!

dir = "./working_directory/"
input, output = dir * "model.umb", dir * "policy.txt"
model = POMDPs_UMB.read_umb(input; discount=0.95)

solver = MCTSSolver()
planner = solve(solver, model)

policy = POMDPs_UMB.Explicit_policy(planner, model)
POMDPs_UMB.write_policy(output, policy)
println("Policy computation complete!")