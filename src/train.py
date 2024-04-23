import glob
import os
import time
from tqdm import trange

# import supersuit as ss
from stable_baselines3 import PPO
# from env.delivery_env_with_obstacle import DeliveryEnvironmentWithObstacle
# from stable_baselines3.common.env_checker import check_env
from env.uav_env import UAVTrainingEnvironmentWithObstacle

def train_uav_policy():
    pass

if __name__ == "__main__":
    env = UAVTrainingEnvironmentWithObstacle(num_uav_obstacle=1, num_no_fly_zone=1, render_mode="human")
    # print(env.observation_space.sample())
    # check_env(env)
    # print(env.observation_space.spaces.shape)

    log_path = os.path.join("training", "logs")
    model_path = os.path.join("training", "models", "model_PPO")
    if os.path.exists(model_path + ".zip"):
        print("load model from last training.")
        model = PPO.load(model_path)
    else:
        print("training new model from scratch.")
        model = PPO("MultiInputPolicy", env, verbose=1, batch_size=256, tensorboard_log=log_path)
        
    print(f"Starting training on {str(env.metadata['name'])}.")
    model.learn(total_timesteps=500_000, progress_bar=True)
    model.save(model_path)
    
    obs, info = env.reset(options=1)
    rewards = 0
    env.render()
    for _ in range(100):
        action, _ = model.predict(obs)
        # action = env.action_space.sample()
        # print(action)
        obs, reward, termination, truncation, info = env.step(action)
        rewards += reward
        env.render()
        if termination or truncation:
            break
            
    env.close()
    print(rewards)
    
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 8, 2)
    