# import glob
import os
# from tqdm import trange
import numpy as np

from stable_baselines3 import PPO
# from env.delivery_env_with_obstacle import DeliveryEnvironmentWithObstacle
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from env.uav_env import UAVTrainingEnvironmentWithObstacle
from base import CustomFeatureExtractor

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder which contains the file created by the ``Monitor`` wrapper.
    :param model_dir: Path to the folder where the model will be saved.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, model_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(model_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True
    

def train_uav_policy(total_timesteps=20_000, progress_bar=False):
    monitor_log_dir = os.path.join("training", "logs", "Monitor")
    log_path = os.path.join("training", "logs")
    model_path = os.path.join("training", "models", "model_PPO")
    env = UAVTrainingEnvironmentWithObstacle(num_uav_obstacle=1, num_no_fly_zone=1, render_mode="human")
    env = Monitor(env, monitor_log_dir)
    callback = SaveOnBestTrainingRewardCallback(check_freq=10_000, log_dir=monitor_log_dir, model_dir=os.path.join("training", "models"))
    
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor, 
        # features_extractor_kwargs=dict()
    )
    if os.path.exists(model_path + ".zip"):
        print("load model from last training.")
        model = PPO.load(model_path)
        model.set_env(env)
    else:
        print("training new model from scratch.")
        model = PPO("MultiInputPolicy", env, verbose=1, batch_size=256, tensorboard_log=log_path, policy_kwargs=policy_kwargs)
        
    print(f"Starting training on {str(env.metadata['name'])}.")
    model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar, callback=callback)
    model.save(model_path)
    print("Last model has been saved.")
    # print(model.policy)
    

if __name__ == "__main__":
    env = UAVTrainingEnvironmentWithObstacle(num_uav_obstacle=1, num_no_fly_zone=1, render_mode="human")
    env = Monitor(env, os.path.join("training", "logs", "Monitor"))
    
    # print(env.observation_space.sample()["vecs"])
    # check_env(env)
    # # print(type(env.observation_space.spaces))
    # obs, info = env.reset()
    # print(obs["vecs"])


    # log_path = os.path.join("training", "logs")
    # model_path = os.path.join("training", "models", "model_PPO")
    # if os.path.exists(model_path + ".zip"):
    #     print("load model from last training.")
    #     model = PPO.load(model_path)
    #     model.set_env(env)
    # else:
    #     print("training new model from scratch.")
    #     model = PPO("MultiInputPolicy", env, verbose=0, batch_size=256, tensorboard_log=log_path)
        
    # print(f"Starting training on {str(env.metadata['name'])}.")
    # model.learn(total_timesteps=1_000, progress_bar=True)
    # model.save(model_path)
    
    # train_uav_policy(5_000_000, True)
    
    model_path = os.path.join("training", "models", "best_model_0")
    model = PPO.load(model_path)
    obs, info = env.reset(options=1)
    rewards = 0
    env.render()
    for _ in range(20):
    # while True:
        action, _ = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        # print(action)
        obs, reward, termination, truncation, info = env.step(action)
        # print(reward)
        rewards += reward
        env.render()
        if termination or truncation:
            # print(env.time_step, termination, truncation)
            break
            
    env.close()
    print(rewards)
    
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 8, 2)
    