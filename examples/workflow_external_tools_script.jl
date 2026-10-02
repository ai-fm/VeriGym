using POMDPs
using MCTS

include("../src/verigym/external/julia/POMDPs_UMB.jl")
import .POMDPs_UMB   # Ours!

t0 = time()
dir = "./working_directory/"
input, output = dir * "model.umb", dir * "policy.txt"
model = POMDPs_UMB.read_umb(input; discount=0.95)
initial_state = support(POMDPs.initialstate(model))[1]
t_reading = time()

solver = MCTSSolver(n_iterations=5_000, depth=50, reuse_tree=true)
planner = solve(solver, model)
for s in support(POMDPs.initialstate(model))
    _action = action(planner, initial_state)
end
t_solving = time()

policy = POMDPs_UMB.Explicit_policy(planner, model)
POMDPs_UMB.write_policy(output, policy)
t_writing = time()

println("Policy computation complete! (In $(t_writing - t0)s, achieving value $(POMDPs.value(planner, initial_state))")