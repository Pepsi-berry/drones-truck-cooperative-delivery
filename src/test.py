import os
import numpy as np
from re import match
from copy import copy
# import glob
# import time
# import supersuit as ss
from env.delivery_env import DeliveryEnvironment
from stable_baselines3 import PPO
from pettingzoo.test import parallel_api_test
from pettingzoo.utils import parallel_to_aec
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.spaces import Box, Discrete, MultiDiscrete


if __name__ == "__main__":
    # temp = Box(low=-1, high=1, shape=(2, ))
    
    # print(temp.sample())
    
    # ListA = [1, 3, 4, 2, 7, 6, 5, 5, 11, 14]
    # result = [(x ** 2 if x % 2 != 0 else x // 2) for x in ListA]
    # print(result)
    
    # ListA = [1, 3, 4, 2, 7, 6, 5, 5, 11, 14, 16]
    # new_list = [(x**2 if x % 3 == 0 else x//2 if x % 4 == 0 else x) for x in ListA]
    # print(new_list)
    possible_agents = ["truck", 
                        "carried_uav_1_1", "carried_uav_1_2", 
                        "carried_uav_2_1", "carried_uav_2_2", 
                        "carried_uav_2_3", "carried_uav_2_4", 
                        "returning_uav_1_1", "returning_uav_1_2", 
                        "returning_uav_2_1", "returning_uav_2_2", 
                        "returning_uav_2_3", "returning_uav_2_4", 
                        ]
    # action_spaces = {
    # agent: (
    #         Discrete(17) if match("truck", agent) 
    #         else Discrete(4) if match("carried", agent)
    #         else Box(low=0, high=np.pi, shape=(1, ))
    #     ) 
    #     for agent in possible_agents
    # }   
    # observation_spaces = {
    #     agent: (
    #         MultiDiscrete(np.full([12, 2], 1500)) if match("truck", agent) 
    #         else MultiDiscrete([15_000, 15_000] * (1 + 1 + 1 + 6))
    #     ) 
    #     for agent in possible_agents
    # }
    # print(action_spaces)
    # print(observation_spaces["truck"].sample()[0])
    
    # for i in range(len(possible_agents)):
    #     print(possible_agents[i]) 
    
    # warehouse_position = np.array([50, 50])
    # truck_position = copy(warehouse_position)
    # uav_position = np.array([copy(truck_position) for _ in range(5)])
    # observations = {
    #     possible_agents[i]: (
    #         np.concatenate([[warehouse_position, truck_position], uav_position]) if i == 0
    #         else [warehouse_position, truck_position, uav_position[i%5]]
    #     ) 
    #     for i in range(len(possible_agents))
    # }
    
    # print(observations)
    
    # env = DeliveryEnvironment()
    
    # parallel_api_test(env, num_cycles=1_000_000)
    
    # log_path = os.path.join("training", "logs")
    
    # observations, infos = env.reset()
    # total_rewards = {agent : 0 for agent in env.agents}

    # while env.agents:
    #     # this is where you would insert your policy
    #     actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     total_rewards["prisoner"] = rewards["prisoner"]
    #     total_rewards["guard"] = rewards["guard"]
    
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


# env = DeliveryEnvironment()

# env.reset()

# env.render()

