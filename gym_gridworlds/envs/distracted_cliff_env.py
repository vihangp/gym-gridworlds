import gym
from gym import spaces
import numpy as np


class DistractedCliffEnv(gym.Env):
    def __init__(self):
        self.height = 4
        self.width = 12
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.height),
                spaces.Discrete(self.width)
                ))
        self.moves = {
                0: (-1, 0),   # up
                1: (0, 1),   # right
                2: (1, 0),  # down
                3: (0, -1),  # left
                }
        self.rf = np.ones([4, 12])
        self.rf[3, 1:11] = 10
        self.rf[3,0] = 0

        self.timestep = 0

        # begin in start state
        self.reset()

    def step(self, action):
        self.timestep += 1
        x, y = self.moves[action]
        self.S = self.S[0] + x, self.S[1] + y

        # if on the edge stay there itself
        self.S = max(0, self.S[0]), max(0, self.S[1])
        # if on the edge stay there itself
        self.S = (min(self.S[0], self.height - 1),
                  min(self.S[1], self.width - 1))
        # reward function
        if self.S == (self.height - 1, self.width - 1):
            r = self.rf[self.S]
            return self.S, r, True, {}
        elif self.S[1] != 0 and self.S[0] == self.height - 1:
            # the cliff - return the reward and end the episode
            r = self.rf[self.S]
            return self.S, r, True, {}
        elif self.timestep == 100
            r = self.rf[self.S]
            self.rf[self.S] = 0
            return self.S, r, True, {}

        r = self.rf[self.S]
        self.rf[self.S] = 0
        return self.S, r, False, {}

    def reset(self):
        # reset the state
        self.S = (3, 0)

        # reset reward function
        self.rf = np.ones([4, 12])
        self.rf[3, 1:11] = 10
        self.rf[3, 0] = 0

        # reset timesteps
        self.timestep = 0

        return self.S
