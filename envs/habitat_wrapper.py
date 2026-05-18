import gymnasium as gym
import numpy as np
import torch
import cv2

class HabitatLeWMWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that formats observations/actions for LeWM's data loader.
    
    Maintains a 3-frame history buffer matching LeWM's history_len=3.
    Resizes to 224x224, normalizes to [0,1], matches LeWM's expected input range.
    Action space is continuous [-1, 1]^2 for CEM compatibility.
    """
    
    def __init__(self, env, img_size=224, frame_skip=5, history_len=3):
        super().__init__(env)
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.history_len = history_len
        self.obs_buffer = []
        self.action_dim = 2  # (linear, angular)

        self.observation_space = gym.spaces.Box(
            0, 1, shape=(history_len, 3, img_size, img_size), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(self.action_dim,), dtype=np.float32
        )

    def _process_obs(self, obs):
        rgb = obs["rgb"]
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        return rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.obs_buffer = [self._process_obs(obs)] * self.history_len
        return np.stack(self.obs_buffer), info

    def step(self, action):
        v_lin, v_ang = np.clip(action, -1.0, 1.0)
        total_reward, done, truncated, info = 0.0, False, False, {}

        for _ in range(self.frame_skip):
            if abs(v_ang) > 0.3:
                discrete = "TURN_LEFT" if v_ang > 0 else "TURN_RIGHT"
            else:
                discrete = "MOVE_FORWARD"
            obs, reward, terminated, truncated, info = self.env.step(discrete)
            total_reward += reward
            done = terminated or truncated
            if done:
                break

        proc_obs = self._process_obs(obs)
        self.obs_buffer.pop(0)
        self.obs_buffer.append(proc_obs)
        return np.stack(self.obs_buffer), total_reward, done, truncated, info
