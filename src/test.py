import numpy as np
from re import match, findall
import random
from env.delivery_env_with_obstacle import DeliveryEnvironmentWithObstacle
from pettingzoo.test import parallel_api_test # , render_test
from gymnasium.spaces import MultiDiscrete, Dict, MultiBinary, Box
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv
# import os
# from copy import copy
# import glob
# import time
# import supersuit as ss
# from env.delivery_env import DeliveryEnvironment
# from stable_baselines3 import PPO
# from pettingzoo.utils import parallel_to_aec
# from stable_baselines3.common.evaluation import evaluate_policy


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

def get_uav_info(uav):
    # get the uav info: kind and no.
    uav_info = findall(r'\d+', uav)
    uav_info = [int(num) for num in uav_info]
    uav_no = uav_info[0] * 2 + uav_info[1]
    return uav_info + [uav_no]

def zones_intersection(zone, xlo , xhi, ylo, yhi):
    lower_left = zone[0]
    upper_right = zone[0] + zone[1]
    print(lower_left, upper_right, xlo, xhi, ylo, yhi)
    if xlo > upper_right[0] or xhi < lower_left[0] or ylo > upper_right[1] or yhi < lower_left[1]:
        return None
    else:
        return np.array([
            max(xlo, lower_left[0]), 
            min(xhi, upper_right[0]), 
            max(ylo, lower_left[1]), 
            min(yhi, upper_right[1])])
        
class upper_solver():
    def __init__(self, pos_obs) -> None:
        self.warehouse_pos = pos_obs[0]
        self.customer_pos_truck = pos_obs[2 + 6 :]
        self.customer_pos_uav = pos_obs[2 + 6 + 4 :]

        
    # idea 1: launch a uav to a customer point when there is a uav available, 
    # and a customer point which is closer than the suitable distance limitation from truck 
    # and start to getting further
    def solve(self, global_obs, agent_infos, num_uavs, uav_range):
        pos_obs = global_obs["pos_obs"]
        truck_action_masks = global_obs["truck_action_masks"]
        uav_action_masks = global_obs["uav_action_masks"]
        truck_pos = pos_obs[1]
        uav_pos = pos_obs[2 : 2 + num_uavs]
        
        task_truck_queue = [ 0 ]
        task_uav_0_queue = [ -1 ]
        task_uav_1_queue = [ -1 ]
        truck_queue = []
        uav_0_avail_queue = []
        uav_1_avail_queue = []
        uav_NA_queue = []
        
        for agent_info in agent_infos:
            if agent_infos[agent_info]:
                if match("truck", agent_info):
                    truck_queue.append(agent_info)
                elif match("uav_0", agent_info):
                    uav_0_avail_queue.append(
                        agent_info
                        # {
                        #     get_uav_info(agent_info)[2] : agent_info
                        # }
                    )
                else:
                    uav_1_avail_queue.append(agent_info)
                    
            elif not agent_infos[agent_info] and match("uav", agent_info):
                uav_NA_queue.append(
                    agent_info
                    # {
                    #     get_uav_info(agent_info)[2] : agent_info
                    # }
                )
                
        # get task queue for truck
        for i in range(self.customer_pos_truck.shape[0]): 
            if truck_action_masks[i]:
                task_truck_queue.append(i + 1)
        
        # get task queue for uav
        for i in range(self.customer_pos_uav.shape[0]): 
            if np.sqrt(np.sum(np.square(truck_pos - self.customer_pos_uav[i]))) < uav_range[0] * 0.3 and uav_action_masks[0][i]:
                task_uav_0_queue.append(i)
            if np.sqrt(np.sum(np.square(truck_pos - self.customer_pos_uav[i]))) < uav_range[1] * 0.3 and uav_action_masks[1][i]:
                task_uav_1_queue.append(i)
        
        # print("wh:", self.warehouse_pos, "truck:", self.customer_pos_truck, "uav:", self.customer_pos_uav)
        # print(truck_queue, uav_0_avail_queue, uav_1_avail_queue, uav_NA_queue, task_truck_queue)
        # print(task_uav_0_queue, task_uav_1_queue)
        
        actions = {}
        # if truck_queue:
        #     actions.update(dict(zip([truck_queue], random.sample(task_truck_queue, len(truck_queue)))))
        for uav in uav_0_avail_queue:
            if task_uav_0_queue:
                task = random.choice(task_uav_0_queue)
                actions[uav] = task
                if task in task_truck_queue:
                    task_truck_queue.remove(task)
                task_uav_0_queue.remove(task)
                if task in task_uav_1_queue:
                    task_uav_1_queue.remove(task)
        for uav in uav_1_avail_queue:
            if task_uav_1_queue:
                task = random.choice(task_uav_1_queue)
                actions[uav] = task
                if task in task_truck_queue:
                    task_truck_queue.remove(task)
                if task in task_uav_0_queue:
                    task_uav_0_queue.remove(task)
                task_uav_1_queue.remove(task)
        for truck in truck_queue:
            if task_truck_queue:
                task = random.choice(task_truck_queue)
                actions[truck] = task
                task_truck_queue.remove(task)
                if task in task_uav_0_queue:
                    task_uav_0_queue.remove(task)
                if task in task_uav_1_queue:
                    task_uav_1_queue.remove(task)
                    
        return actions


if __name__ == "__main__":

    
    # env = DeliveryEnvironment(render_mode="human")
    
    # print(np.row_stack([[0, 0], np.ones(2), np.zeros([5, 2])]))
    # print([[0, 0], np.ones(2), np.zeros([5, 2])])
    
    # b = [1, 2]
    # a = 3
    # print(b + [a])
    
    # possible_agents = ["truck", 
    #                         "uav_0_0", "uav_0_1", 
    #                         "uav_1_0", "uav_1_1", 
    #                         "uav_1_2", "uav_1_3", 
    #                         ]
    # uav_velocity = np.array([12, 29])
    # uav_name_mapping = dict(zip([agent for agent in possible_agents if not match("truck", agent) ],
    #                                     list(range(6))))
    # print(uav_name_mapping)
    # action_spaces = {
    #     agent: (
    #         Discrete(10 + 1) if match("truck", agent) 
    #         else Box(low=np.array([0, 0]), high=np.array([2*np.pi, uav_velocity[get_uav_info(agent)[0]]]), dtype=np.float32)
    #     ) 
    #     for agent in possible_agents
    # }
    
    # space = MultiDiscrete(np.array([[4, 4, 4], [4, 4, 4]]))
    # space = Dict(
    #     {
    #         "surroundings": MultiBinary([3, 5, 5]), 
    #         "coordi_info": MultiDiscrete(np.full([2, 2], 15_000 + 1))
    #     }
    # )
    # space = Box(low=np.array([1, 1]), high=np.array([10, 40]), dtype=float)
    
    # print(space.sample())
    # print(zones_intersection(np.array([[100, 100], [100, 100]]), 150, 250, 50, 150))
    # a = np.zeros([2, 2, 2])
    # a[0] = 1
    # print(a)
    
    # for a in action_spaces:
    #     print(action_spaces[a])
    
    env = DeliveryEnvironmentWithObstacle(render_mode="human")
    parallel_api_test(env, num_cycles=1000)
    # render_test(DeliveryEnvironmentWithObstacle)
    
    # print(
    #     np.row_stack([env.truck_position, env.truck_position, 
    #                 np.array([[env.get_uav_info(agent)[2], env.uav_velocity[env.get_uav_info(agent)[0]]], 
    #                             [20, 75]]), 
    #                 env.uav_position, np.array([center for center in env.uav_obstacles]), 
    #                 env.no_fly_zones[:, :1], env.no_fly_zones[:, 1:]] for agent in env.possible_agents)
    # )
    # uav_obs_space = np.row_stack([[env.num_uavs, np.max(env.uav_velocity)], 
    #                               [20, 75], 
    #                               np.full([(1 + 1 + env.num_uavs + env.num_uav_obstacle + env.num_no_fly_zone * 2), 2], 
    #                                       env.map_size + 1)])
    # print(uav_obs_space)
    # print(len(uav_obs_space))
    # env.reset()
    
    # # parallel_api_test(env, num_cycles=1_000)
    
    # arr = np.ones([2, 3, 4])
    # print(arr.shape)
    
    observations, infos = env.reset()
    delivery_upper_solver = upper_solver(observations["truck"]["observation"]["pos_obs"])
    
    # print(observations["uav_0_1"]["observation"]["coordi_info"])
    # print(env.parcels_weight)
    # # print(env.uav_battery_remaining)
    # env.render()
    # total_rewards = None
    # time_step = 0
    # # for _ in range(20):
    # #     print(type(env.action_space("returning_uav_1_1").sample()))
    # # print(env.parcels_weight)
    # # for agent_obs in observations:
    # #     print(observations[agent_obs]["action_mask"])
    # for agent_obs in observations:
    #     if match("truck", agent_obs):
    #         print(observations[agent_obs]) 
    # TA_Scheduling_action = delivery_upper_solver.solve(observations["truck"]["observation"], infos, env.num_uavs, env.uav_range)
    # print(observations["uav_0_1"])
    # np.savetxt("obs.txt", np.row_stack([observations[agent_obs]["observation"]["surroundings"][0] for agent_obs in observations if match("uav", agent_obs)]))
    
    for i in range(150):
        # this is where you would insert your policy
        if infos["truck"] or (i % 6 == 0):
            TA_Scheduling_action = delivery_upper_solver.solve(observations["truck"]["observation"], infos, env.num_uavs, env.uav_range)
            # print(TA_Scheduling_action)
            env.TA_Scheduling(TA_Scheduling_action)
        
        actions = {
            # here is situated the policy
            # agent: (sample_action(env, observations, agent) if infos[agent]["IsReady"] == True and not match("return", agent)
            #         else env.action_space(agent).sample() if infos[agent]["IsReady"] == True and match("return", agent)
            #         else None)
            # agent: (sample_action(env, observations, agent) if infos[agent]["IsReady"] == True and not match("uav", agent)
            #         else env.action_space(agent).sample() if infos[agent]["IsReady"] == True and match("uav", agent)
            #         else None)
            agent: sample_action(env, observations, agent)
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # env.render()
        # print(observations["uav_0_1"]["observation"]["coordi_info"])

        
        
    #     # if rewards["Global"] != -1:
    #     #     print("rewards: " + str(rewards["Global"]))
        
    #     if time_step <= 100:
    #         env.render()
        
    #     time_step += 1
    
    # env.close()
    # # for agent_obs in observations:
    # #     print(observations[agent_obs]["action_mask"])
    print("pass")
    # print(observations["uav_0_1"]["observation"]["coordi_info"])
    # np.savetxt("obs.txt", np.row_stack([observations[agent_obs]["observation"]["surroundings"][0] for agent_obs in observations if match("uav", agent_obs)]))
    # time.sleep(5)
    # print(total_rewards)
    env.close()

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
