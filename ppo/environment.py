from typing import Optional
import numpy as np
import gymnasium as gym


WINNING_CASES_COUNT = 56
RINGS = 4
LINES = 8

class CircularTicTacToe(gym.Env):
    def __init__(self):
        super().__init__()
        self.rings = RINGS
        self.lines = LINES

        self.current_player = 1
        self.action_space = gym.spaces.Discrete(self.rings*self.lines)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.rings*self.lines,), dtype=np.int8)
        self.board = np.zeros(self.rings*self.lines, dtype=np.int8)
        self.winning_cases = np.zeros((self.rings*self.lines, WINNING_CASES_COUNT), dtype=np.int8)
        self.game_matrix = np.zeros(WINNING_CASES_COUNT, dtype=np.int8)
        self.reward_win = 100
        self.reward_draw = 0
        self.reward_lose = -100
        self.initialize_winning_cases()
        self.reset()

    def action_mask(self) -> list[np.int8]:
        mask = [(entry==0)*1 for entry in self.board]
        return mask

    def initialize_winning_cases(self):
        for i in range(self.rings*self.lines):
            for j in range(4):
                v = i % 8
                v = (v - j + 8) % 8
                self.winning_cases[i][i // 8 * 8 + v] = 1
            self.winning_cases[i][32 + i % 8] = 1
            self.winning_cases[i][40 + (i + i // 8) % 8] = 1
            self.winning_cases[i][48 + (i - i // 8 + 8) % 8] = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.board[:] = 0
        self.game_matrix[:] = 0
        self.current_player = 1
        info = {}
        info["action_mask"] = self.action_mask()
        return self._get_obs(), info

    def _get_obs(self):
        return self.board

    def _is_valid_action(self, action):
        return self.board[action] == 0
    
    def calculate_intermediate_reward(self, action):
        player_perspective = self.game_matrix * self.current_player
        sorted_indices = np.argsort(player_perspective)
        sorted_game_matrix = player_perspective[sorted_indices]
        n_moves = np.sum(np.absolute(self.board))
        time_penalty = (n_moves)/(self.rings*self.lines)
        if sorted_game_matrix[0] == -3:
            return -5 - time_penalty
        for cell in range(self.rings*self.lines):
            if self.board[cell] == 0:
                next_state = self.game_matrix.copy()
                for i in range(WINNING_CASES_COUNT):
                    next_state[i] += -1 * self.current_player * self.winning_cases[cell][i]
                next_state = np.sort(next_state*self.current_player)
                if next_state[0] == -3 and next_state[1] == -3:
                    return -3 - time_penalty
        return (sorted_game_matrix[-1]-1)*self.winning_cases[action][sorted_indices[-1]] + 0.5*(sorted_game_matrix[-2]-1)*self.winning_cases[action][sorted_indices[-2]] - time_penalty

    
    def _is_winning_move(self):
        for i in range(WINNING_CASES_COUNT):
            if self.game_matrix[i] == 4 * self.current_player:
                return True
        return False

    def _is_draw(self):
        for i in range(self.rings*self.lines):
            if self.board[i] == 0:
                return False
        return True

    def _is_losing_move(self):
        for i in range(WINNING_CASES_COUNT):
            if self.game_matrix[i] == -4 * self.current_player:
                return True
        return False

    def step(self, action):
        if not self._is_valid_action(action):
            raise ValueError(f"Invalid action: {action}")

        self.board[action] = self.current_player
        for i in range(WINNING_CASES_COUNT):
            self.game_matrix[i] += self.current_player * self.winning_cases[action][i]
        
        info = {}
        info["action_mask"] = self.action_mask()
        res = None
        if self._is_winning_move():
            res = self._get_obs(), self.reward_win, True, False, info
        elif self._is_losing_move():
            res = self._get_obs(), self.reward_lose, True, False, info
        elif self._is_draw():
            res = self._get_obs(), self.reward_draw, True, False, info
        else:
            reward_normal = self.calculate_intermediate_reward(action)
            res = self._get_obs(), reward_normal, False, False, info
        # print(self.board)
        # print(self.game_matrix)
        # print(res[1])
        self.current_player = -self.current_player
        return res
    
    def render(self, mode="human"):
        print("Board:", self.board)
        print("Game Matrix:", self.game_matrix)


if __name__ == "__main__":
    env = CircularTicTacToe()
    action = env.action_space.sample()
    print(env.step(action))
    action = env.action_space.sample()
    print(env.step(action))
