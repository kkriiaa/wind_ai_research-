import numpy as np
import pygad
import pyswarms as ps
from multi_agent import MultiAgentWindFarm

# 1. Initialize the Environment
env = MultiAgentWindFarm()

def get_reward(angles):
    """
    Bridge: Takes an array of 3 angles and returns 
    the total farm power from your MultiAgent code.
    """
    env.reset()
    # Format the angles for the 3 turbine agents
    actions = {f"turbine_{i}": np.array([angles[i]], dtype=np.float32) for i in range(3)}
    
    # Step the environment
    _, rewards, _, _, _ = env.step(actions)
    
    # In your multi_agent.py, turbine_0 reward is the total_farm_power
    return rewards["turbine_0"]

# ---------------------------------------------------------
# STAGE 1: Genetic Algorithm (Global Exploration)
# ---------------------------------------------------------
def fitness_func(ga_instance, solution, solution_idx):
    return get_reward(solution)

print("--- STAGE 1: Starting Genetic Algorithm ---")
ga_instance = pygad.GA(
    num_generations=30,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=3,
    init_range_low=-30,
    init_range_high=30,
    mutation_percent_genes=10, # Corrected parameter name
    parent_selection_type="sss", # Steady State Selection
    crossover_type="single_point",
    mutation_type="random"
)

ga_instance.run()

# Extract the elite group (top performers) to seed the PSO
best_ga_solution, best_ga_fitness, _ = ga_instance.best_solution()
print(f"GA Phase Complete. Best GA Power: {best_ga_fitness:.4f} MW")

# We take the best 10 individuals from the final GA population
# to use as the starting positions for the 10 PSO particles.
initial_pso_positions = ga_instance.population[:10]

# ---------------------------------------------------------
# STAGE 2: Particle Swarm Optimization (Fine Exploitation)
# ---------------------------------------------------------
def pso_objective(x):
    """PSO expects to minimize a cost, so we return negative power."""
    costs = []
    for i in range(x.shape[0]):
        power = get_reward(x[i])
        costs.append(-power)
    return np.array(costs)

# PSO Hyperparameters:
# c1=Cognitive (personal best), c2=Social (swarm best), w=Inertia
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Initialize the optimizer with the GA's results
optimizer = ps.single.GlobalBestPSO(
    n_particles=10,
    dimensions=3,
    options=options,
    bounds=([-30.0, -30.0, -30.0], [30.0, 30.0, 30.0]),
    init_pos=initial_pso_positions # The Hybrid Link
)

print("\n--- STAGE 2: Starting PSO Refinement ---")
cost, optimized_angles = optimizer.optimize(pso_objective, iters=40)

# ---------------------------------------------------------
# FINAL RESULTS
# ---------------------------------------------------------
print("\n" + "="*30)
print("HYBRID OPTIMIZATION RESULTS")
print("="*30)
print(f"Optimal Yaw Turbine 0: {optimized_angles[0]:.2f}°")
print(f"Optimal Yaw Turbine 1: {optimized_angles[1]:.2f}°")
print(f"Optimal Yaw Turbine 2: {optimized_angles[2]:.2f}°")
print(f"Final Total Farm Power: {-cost:.4f} MW")
print("="*30)
