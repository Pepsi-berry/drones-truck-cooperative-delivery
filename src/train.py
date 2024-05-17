# import glob
import os
# from tqdm import trange
import numpy as np

from stable_baselines3 import PPO, SAC, TD3
# from sb3_contrib import RecurrentPPO
# from env.delivery_env_with_obstacle import DeliveryEnvironmentWithObstacle
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from env.uav_env import UAVTrainingEnvironmentWithObstacle
from env.uav_env_eval import UAVEvalEnvironmentWithObstacle
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
    model_path = os.path.join("training", "models", "model_RNN_PPO")
    env = UAVTrainingEnvironmentWithObstacle(num_uav_obstacle=5, num_no_fly_zone=2, truck_velocity=5)
    # eval_env = UAVTrainingEnvironmentWithObstacle(num_uav_obstacle=5, num_no_fly_zone=2, truck_velocity=4)
    callback = SaveOnBestTrainingRewardCallback(check_freq=5_000, log_dir=monitor_log_dir, model_dir=os.path.join("training", "models"))
    # eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join("training", "models", "best_model"), 
    #                              eval_freq=1_000, n_eval_episodes=10, deterministic=True, callback_after_eval=callback)
    env = Monitor(env, monitor_log_dir)
    
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor, 
        # features_extractor_kwargs=dict()
    )
    if os.path.exists(model_path + ".zip"):
        print("load model from last training.")
        custom_objects = {
                "learning_rate": 0.0002
            }
        model = SAC.load(model_path, custom_objects=custom_objects)
        model.set_env(env)
    else:
        print("training new model from scratch.")
        # model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=0.0003, batch_size=1024, ent_coef=0.01, n_steps=1024, tensorboard_log=log_path, policy_kwargs=policy_kwargs)
        # model = TD3('MultiInputPolicy', env, verbose=1)
        model = SAC('MultiInputPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs=policy_kwargs)
        
    print(f"Starting training on {str(env.metadata['name'])}.")
    model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar, callback=callback)
    model.save(model_path)
    print("Last model has been saved.")
    

if __name__ == "__main__":
    env = UAVTrainingEnvironmentWithObstacle(num_uav_obstacle=25, num_no_fly_zone=4, truck_velocity=4, MAX_STEP=800, step_len=2, render_mode="human")
    # env = UAVEvalEnvironmentWithObstacle(num_uav_obstacle=10, num_no_fly_zone=4, truck_velocity=4, MAX_STEP=800, step_len=2, render_mode="human")
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
    
    # train_uav_policy(1_000, True)
    
    model_path = os.path.join("training", "models", "best_model_SAC_1M_1")
    model = SAC.load(model_path)
    # print("eval result: ", evaluate_policy(model, env, n_eval_episodes=10, deterministic=True))
    obs, info = env.reset(options=1)
    # print(obs)
    # print(env.uav_position, env.truck_position, env.truck_target_position, obs["vecs"])
    rewards = 0
    env.render()
    for _ in range(50):
    # while True:
        action, _ = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        # print(action)
        obs, reward, termination, truncation, info = env.step(action)
        # print(env.uav_position, env.truck_position, env.truck_target_position, obs["vecs"])
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
    