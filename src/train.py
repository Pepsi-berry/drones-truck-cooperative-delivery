import os
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
from env.multi_uav_env import MultiUAVsTrainingEnvironmentWithObstacle
# from env.uav_env_eval import UAVEvalEnvironmentWithObstacle
# from env.delivery_env_with_obstacle import DeliveryEnvironmentWithObstacle
from base import CustomFeatureExtractor, CustomPPOPolicyModel, CustomCallbacks
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

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
    

def env_creator(env_config):
    env = MultiUAVsTrainingEnvironmentWithObstacle(
        MAX_STEP=env_config.get("MAX_STEP", 2_500), 
        step_len=env_config.get("step_len", 2), 
        num_uav_obstacle=env_config.get("num_uav_obstacle", 20), 
        num_no_fly_zone=env_config.get("num_no_fly_zone", 4), 
        uav_velocity=env_config.get("uav_velocity", np.array([12, 29])), 
        truck_velocity=env_config.get("truck_velocity", 4), 
        render_mode=env_config.get("render_mode", None), 
        num_truck=env_config.get("num_truck", 1), 
        num_uavs_0=env_config.get("num_uavs_0", 2), 
        num_uavs_1=env_config.get("num_uavs_1", 4), 
        num_uavs=env_config.get("num_uavs_0", 2) + env_config.get("num_uavs_1", 4), 
        )
    
    return env


if __name__ == "__main__":
    env_config = {
        'MAX_STEP': 2_500, 
        'step_len': 2, 
        # uav parameters
        # unit here is m/s
        'truck_velocity': 4, 
        'uav_velocity': np.array([12, 29]), 
        # unit here is kg
        'uav_capacity': np.array([10, 3.6]), 
        # unit here is m
        'uav_range': np.array([10_000, 15_000]), 
        'uav_obs_range': 150, 
        
        'num_truck': 1, 
        'num_uavs_0': 2, 
        'num_uavs_1': 4, 
        'num_uavs': 6, 
        
        # parcels parameters
        'num_parcels': 40, 
        'num_parcels_truck': 10, 
        'num_parcels_uav': 15, 
        'num_customer_truck': 25, 
        'num_customer_uav': 30, 
        'num_customer_both': 15, 
        'weight_probabilities': [0.8, 0.1, 0.1], 
        
        # map parameters
        'map_size': 10_000,  # m as unit here
        'grid_edge': 125,  # m as unit here
        
        # obstacle parameters
        'num_uav_obstacle': 1, 
        'num_no_fly_zone': 1, 
    }
    
    ray.init()
    
    register_env("ma_training_env", lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("Custom_PPO_Model", CustomPPOPolicyModel)
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # print("worker: ", worker)
        # print("episode: ", episode)
        if agent_id.startswith("uav"):
            return "mappo_policy"
        else:
            raise ValueError("Unknown agent type: ", agent_id)
    
    config = (
        PPOConfig()
        .environment('ma_training_env')
        .env_runners(num_env_runners=4, rollout_fragment_length=128)
        # .rollouts(num_rollout_workers=4, rollout_fragment_length=128) # deprecated, use env_runners instead、
        .experimental(
            _disable_preprocessor_api=True
        )
        .framework('torch')
        .checkpointing(export_native_model_files=True)
        .training(
            model={
                'custom_model': "Custom_PPO_Model",
                'vf_share_layers': True,
                "_disable_preprocessor_api": True
                },
            gamma=0.99, 
            lr=0.0003,
            use_gae=True, 
            use_critic=True, 
            train_batch_size=256, 
            # entropy_coeff=entropy_coeff, 
            # clip_param=clip_param,
            # vf_loss_coeff=vf_loss_coeff,
            # vf_clip_param=vf_clip_param,
            # train_batch_size=train_batch_size,
            # sgd_minibatch_size=sgd_minibatch_size,
            # num_sgd_iter=num_sgd_iter,
            # lambda_=lambda_,
            )
        # .rl_module()
        .multi_agent(
            policies={
                "mappo_policy": PolicySpec(
                policy_class=None,  # infer automatically from Algorithm
                observation_space=None,  # infer automatically from env
                action_space=None,  # infer automatically from env
                config={},  # use main config plus <- this override here
                ),
            },
            policy_mapping_fn=policy_mapping_fn
        )
        .resources(num_gpus=0)
        .debugging(log_level='INFO') # INFO, DEBUG, ERROR, WARN
        .callbacks(callbacks_class=CustomCallbacks)
        )
    
    # algo_ppo = config.build()
    # # print("Model Info", algo_ppo.get_policy())
    # algo_ppo.train()
    
    analysis = tune.run(
        "PPO", 
        config=config, 
        stop={
            # "training_iteration": 50, 
            "timesteps_total": 1_000_000, 
        }, 
        # storage_path="./training/logs", 
        checkpoint_config={
            'checkpoint_at_end': True, 
            'checkpoint_frequncy': 20, 
        }
    )
    
    ##########
    
    # env = MultiUAVsTrainingEnvironmentWithObstacle()
    # parallel_api_test(env, 10_000)
    
    # env = UAVTrainingEnvironmentWithObstacle(num_uav_obstacle=20, num_no_fly_zone=4, truck_velocity=4, MAX_STEP=800, step_len=2, render_mode="human")
    # # env = UAVEvalEnvironmentWithObstacle(num_uav_obstacle=10, num_no_fly_zone=4, truck_velocity=4, MAX_STEP=800, step_len=2, render_mode="human")
     
    # env = Monitor(env, os.path.join("training", "logs", "Monitor"))
    
    # model_path = os.path.join("training", "models", "best_model_SAC_20K")
    # model = SAC.load(model_path)
    # # print(model.policy)
    # # print("eval result: ", evaluate_policy(model, env, n_eval_episodes=10, deterministic=True))
    # obs, info = env.reset(options=0)
    # # print(obs)
    # # print(env.uav_position, env.truck_position, env.truck_target_position, obs["vecs"])
    # rewards = 0
    # env.render()
    # for i in range(50):
    # # while True:
    #     action, _ = model.predict(obs, deterministic=True)
    #     # action = env.action_space.sample()
    #     # print(action)
    #     obs, reward, termination, truncation, info = env.step(action)
    #     # print(env.uav_position, env.truck_position, env.truck_target_position, obs["vecs"])
    #     # print(reward)
    #     rewards += reward
    #     if i % 3 == 0:
    #         env.render()
    #     if termination or truncation:
    #         # print(env.time_step, termination, truncation)
    #         break
            
    # env.close()
    # print(rewards)
    
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 8, 2)
    