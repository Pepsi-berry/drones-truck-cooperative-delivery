import os
import numpy as np
from re import match, findall
# from copy import copy
import random
# import glob
# import time
# import supersuit as ss
# from env.delivery_env import DeliveryEnvironment
from env.delivery_env_with_obstacle import DeliveryEnvironmentWithObstacle
# from stable_baselines3 import PPO
# from pettingzoo.test import parallel_api_test
# from pettingzoo.utils import parallel_to_aec
# from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv

MAX_INT = 100

def sample_action(
    env: ParallelEnv[AgentID, ObsType, ActionType],
    obs: dict[AgentID, ObsType],
    agent: AgentID,
) -> ActionType:
    agent_obs = obs[agent]
    if isinstance(agent_obs, dict) and "action_mask" in agent_obs:
        legal_actions = np.flatnonzero(agent_obs["action_mask"])
        if len(legal_actions) == 0:
            return 0
        return random.choice(legal_actions)
    return env.action_space(agent).sample()

if __name__ == "__main__":

    
    # env = DeliveryEnvironment(render_mode="human")
    env = DeliveryEnvironmentWithObstacle(render_mode="human")
    
    # env.reset()
    
    # parallel_api_test(env, num_cycles=1_000)
    
    observations, infos = env.reset()
    # print(env.uav_battery_remaining)
    env.render()
    total_rewards = None
    time_step = 0
    # for _ in range(20):
    #     print(type(env.action_space("returning_uav_1_1").sample()))
    # print(env.parcels_weight)
    # for agent_obs in observations:
    #     print(observations[agent_obs]["action_mask"])
    
    for _ in range(50):
        # this is where you would insert your policy
        actions = {
            # here is situated the policy
            agent: (sample_action(env, observations, agent) if infos[agent]["IsReady"] == True and not match("return", agent)
                    else env.action_space(agent).sample() if infos[agent]["IsReady"] == True and match("return", agent)
                    else None)
            # agent: sample_action(env, observations, agent)
            for agent in env.agents
        }
        
        # print(env.uav_battery_remaining)
        # print(env.parcels_weight)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # if rewards["Global"] != -1:
        #     print("rewards: " + str(rewards["Global"]))
        
        if time_step <= 100:
            env.render()
        
        time_step += 1
    
    
        # total_rewards["prisoner"] = rewards["prisoner"]
        # total_rewards["guard"] = rewards["guard"]
    
    env.close()
    # for agent_obs in observations:
    #     print(observations[agent_obs]["action_mask"])
    print("pass")
    # print(total_rewards)
    # env.close()

    # # Pre-process using SuperSuit
    # print(f"Starting training on {str(env.metadata['name'])}.")
    
    # # env = parallel_to_aec(env)
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")
    
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path, batch_size=256)
    
    # model.learn(total_timesteps=100_000)
    
    # model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    # print("Model has been saved.")

    # print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    # env.close()
