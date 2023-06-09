import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 render_mode=None,
                 size=5,
                 change_point=None,
                 censor_time=None):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.steps = 0  # The number of steps taken
        self.change_point = change_point
        self.censor_time = censor_time

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict({
            "agent":
            spaces.Box(0, size - 1, shape=(2, ), dtype=int),
            "target":
            spaces.Box(0, size - 1, shape=(2, ), dtype=int),
        })

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        self.reward_matrix = np.zeros([self.size, self.size])
        for i in range(1, self.size):
            self.reward_matrix[i:, i:] += np.ones([size - i, size - i])

        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.np_random = np.random.default_rng(seed=42)

    def _get_obs(self):
        # return {"agent": self._agent_location, "target": self._target_location}
        return {"agent": self._agent_location, "target": np.array([0, 0])}

    def _location_to_state(self, loc):
        return loc[0] * self.size + loc[1]

    def _get_info(self):
        return {
            "steps": self.steps,
            "state": self._location_to_state(self._agent_location)
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._agent_location = np.array([0, 0], dtype=int)
        # self._target_location = np.array([self.size - 1, self.size - 1],
        #                                  dtype=int)
        self.steps = 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_real_action(self, instructed_action, t):
        if self.change_point == None:
            return instructed_action
        _noised_action_to_real_action = {0: 2, 2: 0, 1: 3, 3: 1}
        if t < self.change_point:
            if self.np_random.random() <= 0.8:
                return _noised_action_to_real_action[instructed_action]
            else:
                return instructed_action
        else:
            if self.np_random.random() <= 0.2:
                return _noised_action_to_real_action[instructed_action]
            else:
                return instructed_action

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        real_action = self._get_real_action(action, self.steps)
        direction = self._action_to_direction[real_action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, 0,
                                       self.size - 1)
        self.steps += 1
        # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location,
        #                             self._target_location)
        terminated = False
        truncated = (self.steps == self.censor_time)

        # reward = 1 if terminated else 0  # Binary sparse rewards
        reward = self.reward_matrix[self._agent_location[0],
                                    self._agent_location[1]]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.size
                           )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)),
                                axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
