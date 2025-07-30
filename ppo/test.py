import gymnasium as gym
import numpy as np
from environment import CircularTicTacToe
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gymnasium.utils.env_checker import check_env


def masking(env):
    return env.action_mask()
# Create the agent
env = CircularTicTacToe()
env = ActionMasker(env, masking)

model = MaskablePPO.load("models/ppo_circular_tic_tac_toe_maskable_new_reward_2_1500000", env=env)

obs, _ = env.reset()
num_games = 100
debug = False
win = 0
lose = 0
draw = 0
for j in range(num_games):
    for i in range(32):
        if i%2 == 0:
            #Player 1:
            action_1, _ = model.predict(obs, action_masks=env.action_masks())
            obs, reward, done, truncated, _ = env.step(action_1)
            if debug:
                print("action 1:", action_1)
                print("reward 1:", reward)
                env.render()
        else:
            # Player 2:
            action_2 = np.random.choice([i for i in range(32) if env.action_masks()[i]==1])
            # action_2 = int(input("Enter your index:"))
            obs, reward, done, truncated, _ = env.step(action_2)
            if debug:
                print("action 2:", action_2)
                print("reward 2:", reward)
                env.render()

        if done or truncated:
            if debug:
                print("done in {} moves".format(i+1))
            obs, _ = env.reset()
            if (i == 31):
                draw += 1
            elif i%2==0:
                win += 1
            else:
                lose += 1
            break
            # quit()

print("win:", win)
print("lose:", lose)
print("draw:", draw)