import gymnasium as gym
import torch as th
from environment import CircularTicTacToe
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gymnasium.utils.env_checker import check_env


# Create the agent
env = CircularTicTacToe()
def masking(env):
    return env.action_mask()
# try:
#     #gym.register(id="CircularTicTacToe-v0", entry_point="environment:CircularTicTacToe")
#     check_env(env)
# except Exception as e:
#     print(e)
# quit()
# model = MaskablePPO("MlpPolicy", ActionMasker(env, masking), verbose=1, learning_rate=0.001)
model = MaskablePPO.load("models/ppo_circular_tic_tac_toe_maskable_new_reward_2_1500000", env=ActionMasker(env, masking), verbose=1)

# Train the agent
total_timesteps = 6000000
save_every = 100000
iteration = 1600000
while iteration <= total_timesteps:
    try:
        model.learn(total_timesteps=save_every)
        model.save(f"models/ppo_circular_tic_tac_toe_maskable_new_reward_2_{iteration}")
        iteration += save_every
    except KeyboardInterrupt:
        break

