import numpy as np
import functools
from gymnasium import spaces
from pettingzoo import ParallelEnv
from floris import FlorisModel

class MultiAgentWindFarm(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "windfarm_v1"}

    def __init__(self):
        print("Initializing PettingZoo Multi-Agent WindFarm...")
        self.fmodel = FlorisModel("gch.yaml")
        self.fmodel.set(layout_x=[0.0, 500.0, 1000.0], layout_y=[0.0, 0.0, 0.0])
        
        # Name our 3 independent agents
        self.possible_agents = ["turbine_0", "turbine_1", "turbine_2"]
        self.agents = self.possible_agents[:]

    # Define what EACH agent can do (Yaw angle -30 to 30 degrees)
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-30.0, high=30.0, shape=(1,), dtype=np.float32)

    # Define what EACH agent can see (Wind Speed and Direction)
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=np.array([0.0, 0.0]), high=np.array([25.0, 360.0]), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        
        # Randomize wind for the new episode
        self.current_wind_speed = np.random.uniform(7.0, 10.0)
        self.current_wind_direction = np.random.uniform(260.0, 280.0)
        
        self.fmodel.set(
            wind_speeds=[self.current_wind_speed],
            wind_directions=[self.current_wind_direction],
            yaw_angles=np.zeros((1, 3))
        )
        
        # Give every agent their initial observation
        obs = np.array([self.current_wind_speed, self.current_wind_direction], dtype=np.float32)
        observations = {agent: obs for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        # 1. Unpack the dictionary of actions into an array for FLORIS
        yaw_list = [actions["turbine_0"][0], actions["turbine_1"][0], actions["turbine_2"][0]]
        yaw_angles = np.array(yaw_list).reshape(1, 3)
        
        # 2. Run the Physics Engine
        self.fmodel.set(yaw_angles=yaw_angles)
        self.fmodel.run()
        
        # 3. Get Power Outputs
        turbine_powers = self.fmodel.get_turbine_powers() / 1_000_000 # MW
        total_farm_power = np.sum(turbine_powers[0, :])
        
        # 4. COOPERATIVE REWARD SYSTEM
        # To make them work as a team, we give every agent the exact same reward:
        # the TOTAL power of the farm. If turbine_0 acts selfishly, the whole team loses points.
        rewards = {agent: total_farm_power for agent in self.agents}
        
        # 5. Package observations and status
        obs = np.array([self.current_wind_speed, self.current_wind_direction], dtype=np.float32)
        observations = {agent: obs for agent in self.agents}
        
        # We run infinitely for this test
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        # Attach individual power stats so we can monitor them
        infos = {
            "turbine_0": {"power": turbine_powers[0, 0]},
            "turbine_1": {"power": turbine_powers[0, 1]},
            "turbine_2": {"power": turbine_powers[0, 2]}
        }

        # If anyone dies/terminates, remove them from the active list
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

# ==========================================
# TEST: 3 Agents taking random actions
# ==========================================
if __name__ == "__main__":
    env = MultiAgentWindFarm()
    observations, infos = env.reset()
    
    print(f"\n--- Episode Start ---")
    
    for step in range(3): # Run 3 time steps
        # Each agent independently decides an action
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Pass the dictionary of actions to the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print(f"\n[Step {step+1}]")
        for agent in env.agents:
            print(f"{agent} | Action (Yaw): {actions[agent][0]:.1f}° | Power: {infos[agent]['power']:.2f} MW | Reward: {rewards[agent]:.2f}")
