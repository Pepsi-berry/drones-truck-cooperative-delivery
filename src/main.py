# import os
import numpy as np
from re import match, findall
import random
import time
# import supersuit as ss
# from env.delivery_env import DeliveryEnvironment
from env.delivery_env_with_obstacle import DeliveryEnvironmentWithObstacle
# from stable_baselines3 import PPO
# from pettingzoo.test import parallel_api_test
# from pettingzoo.utils import parallel_to_aec
# from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.spaces import MultiDiscrete, Dict, MultiBinary, Box
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
    def __init__(self, pos_obs, 
                 num_both_customer=10, 
                 num_truck_customer=4, 
                 num_uav_customer=6, 
                 num_uavs=6) -> None:
        
        self.num_both_customer = num_both_customer
        self.num_truck_customer = num_truck_customer
        self.num_uav_customer = num_uav_customer
        self.warehouse_pos = pos_obs[0]
        self.num_uavs = num_uavs
        self.customer_pos_truck = pos_obs[2 + self.num_uavs :]
        self.customer_pos_uav = pos_obs[2 + self.num_uavs + self.num_truck_customer :]
        
    # idea 1: launch a uav to a customer point when there is a uav available, 
    # and a customer point which is closer than the suitable distance limitation from truck 
    # and start to getting further
    def solve(self, global_obs, agent_infos, uav_range):
        pos_obs = global_obs["pos_obs"]
        truck_action_masks = global_obs["truck_action_masks"]
        uav_action_masks = global_obs["uav_action_masks"]
        truck_pos = pos_obs[1]
        uav_pos = pos_obs[2 : 2 + self.num_uavs]
        # num_truck_customer = self.customer_pos_truck.shape[0] - self.customer_pos_uav.shape[0]
        
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
                    )
                else:
                    uav_1_avail_queue.append(agent_info)
                    
            elif not agent_infos[agent_info] and match("uav", agent_info):
                uav_NA_queue.append(
                    agent_info
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
        
        # randomly assigned task to uav and truck in loop, and update the queue 
        # to avoid the same task being repeatedly assigned among different agents
        for uav in uav_0_avail_queue:
            if task_uav_0_queue:
                task = random.choice(task_uav_0_queue)
                actions[uav] = task
                # empty action should always be kept in queue
                if task != -1:
                    # if task in task_truck_queue:
                    task_truck_queue.remove(task + 1 + self.num_truck_customer)
                    task_uav_0_queue.remove(task)
                    if task in task_uav_1_queue:
                        task_uav_1_queue.remove(task)
        for uav in uav_1_avail_queue:
            if task_uav_1_queue:
                task = random.choice(task_uav_1_queue)
                actions[uav] = task
                if task != -1:
                    # if task in task_truck_queue:
                    task_truck_queue.remove(task + 1 + self.num_truck_customer)
                    if task in task_uav_0_queue:
                        task_uav_0_queue.remove(task)
                    task_uav_1_queue.remove(task)
        for truck in truck_queue:
            if task_truck_queue:
                task = random.choice(task_truck_queue)
                actions[truck] = task
                task_truck_queue.remove(task)
                if (task - 1 - self.num_truck_customer) in task_uav_0_queue:
                    task_uav_0_queue.remove((task - 1 - self.num_truck_customer))
                if (task - 1 - self.num_truck_customer) in task_uav_1_queue:
                    task_uav_1_queue.remove((task - 1 - self.num_truck_customer))
                    
        return actions

    # no action should not be removed from queue, so does solve()
    def solve_greedy(self, global_obs, agent_infos, uav_range):
        pos_obs = global_obs["pos_obs"]
        truck_action_masks = global_obs["truck_action_masks"]
        uav_action_masks = global_obs["uav_action_masks"]
        truck_pos = pos_obs[1]
        uav_pos = pos_obs[2 : 2 + self.num_uavs]
        # num_truck_customer = self.customer_pos_truck.shape[0] - self.customer_pos_uav.shape[0]
        
        task_truck_queue = [ 0 ]
        # no action as the last action(could be remove)
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
                    )
                else:
                    uav_1_avail_queue.append(agent_info)
                    
            elif not agent_infos[agent_info] and match("uav", agent_info):
                uav_NA_queue.append(
                    agent_info
                )
                
        # get the indice of ordered task distance queue from truck
        task_dists = np.zeros(self.customer_pos_truck.shape[0])
        for i in range(self.customer_pos_truck.shape[0]): 
            task_dists[i] = np.sqrt(np.sum(np.square(truck_pos - self.customer_pos_truck[i])))
        task_dists_order = np.argsort(task_dists)
        # uav_task_dists = task_dists[ -self.customer_pos_uav.shape[0] : ]
        
        truck_task_dists = task_dists[ : self.num_truck_customer]
        both_task_dists = task_dists[self.num_truck_customer : self.num_truck_customer + self.num_both_customer]
        uav_task_dists = task_dists[self.num_truck_customer + self.num_both_customer : ]
        
        truck_task_dists_order = np.argsort(truck_task_dists)
        both_task_dists_order = np.argsort(both_task_dists)
        uav_task_dists_order = np.argsort(uav_task_dists)
        
        truck_task_dists_order = np.concatenate([truck_task_dists_order, 
                                                 both_task_dists_order + self.num_truck_customer, 
                                                 uav_task_dists_order + self.num_truck_customer + self.num_both_customer])
        uav_task_dists_order = np.concatenate([uav_task_dists_order + self.num_both_customer, 
                                               both_task_dists_order])
        
        uav_task_dists = task_dists[self.num_truck_customer : ]
        # uav_task_dists_order = np.argsort(uav_task_dists)
        # print("uav TD: ", task_dists[ -self.customer_pos_uav.shape[0] : ])
        # print("TD: ", task_dists)
        # print("order: ", task_dists_order)
        # print("truck order: ", truck_task_dists_order, "uav order; ", uav_task_dists_order)
        # print("TC: ", truck_c_dists, "BC: ", both_c_dists, "UC: ", uav_c_dists)
          
        # get task queue for truck
        for idx in np.flip(truck_task_dists_order): # truck_task_dists_order[::-1]
            if truck_action_masks[idx]:
                task_truck_queue.append(idx + 1)
        
        # get task queue for uav
        for idx in np.flip(uav_task_dists_order): 
            if uav_task_dists[idx] < uav_range[0] * 0.3 and uav_action_masks[0][idx]:
                task_uav_0_queue.append(idx)
            if uav_task_dists[idx] < uav_range[1] * 0.3 and uav_action_masks[1][idx]:
                task_uav_1_queue.append(idx)
        
        # task_uav_0_queue.append(-1)
        # task_uav_1_queue.append(-1)
        
        # print("wh:", self.warehouse_pos, "truck:", self.customer_pos_truck, "uav:", self.customer_pos_uav)
        # print(truck_queue, uav_0_avail_queue, uav_1_avail_queue, uav_NA_queue, task_truck_queue)
        # print(task_uav_0_queue, task_uav_1_queue)
        
        actions = {}
        
        # assigned the shortest customer point in available task queue to available agent
        for uav in uav_0_avail_queue:
            if task_uav_0_queue:
                task = task_uav_0_queue.pop()
                actions[uav] = task
                if task != -1:
                    # if task in task_truck_queue:
                    task_truck_queue.remove(task + 1 + self.num_truck_customer)
                    if task in task_uav_1_queue:
                        task_uav_1_queue.remove(task)
                else:
                    task_uav_0_queue.append(-1)
        for uav in uav_1_avail_queue:
            if task_uav_1_queue:
                task = task_uav_1_queue.pop()
                actions[uav] = task
                if task != -1:
                    # if task in task_truck_queue:
                    task_truck_queue.remove(task + 1 + self.num_truck_customer)
                    if task in task_uav_0_queue:
                        task_uav_0_queue.remove(task)
                else:
                    task_uav_1_queue.append(-1)
        for truck in truck_queue:
            if task_truck_queue:
                task = task_truck_queue.pop()
                actions[truck] = task
                if (task - 1 - self.num_truck_customer) in task_uav_0_queue:
                    task_uav_0_queue.remove((task - 1 - self.num_truck_customer))
                if (task - 1 - self.num_truck_customer) in task_uav_1_queue:
                    task_uav_1_queue.remove((task - 1 - self.num_truck_customer))
        
        # print(actions)
        return actions

# probably we need to implement the marl algorithm by ourselves :(
class PPO():
    def __init__(self) -> None:
        pass

if __name__ == "__main__":
    
    possible_agents = ["truck", 
                            "uav_0_0", "uav_0_1", 
                            "uav_1_0", "uav_1_1", 
                            "uav_1_2", "uav_1_3", 
                            ]
    # uav parameters
    # unit here is m/s
    truck_velocity = 7
    uav_velocity = np.array([12, 29])
    # unit here is kg
    uav_capacity = np.array([10, 3.6])
    # unit here is m
    uav_range = np.array([10_000, 15_000])
    uav_obs_range = 151
    
    num_truck = 1
    num_uavs = 6
    num_uavs_0 = 2
    num_uavs_1 = 4
    
    # parcels parameters
    num_parcels = 20
    num_parcels_truck = 4
    num_parcels_uav = 6
    num_customer_truck = num_parcels - num_parcels_uav
    num_customer_uav = num_parcels - num_parcels_truck
    num_customer_both = num_parcels - num_parcels_truck - num_parcels_uav
    weight_probabilities = [0.8, 0.1, 0.1]
    
    # map parameters
    map_size = 10_000 # m as unit here
    grid_edge = 250 # m as unit here
    
    # obstacle parameters
    num_uav_obstacle = 20
    num_no_fly_zone = 8
    
    env = DeliveryEnvironmentWithObstacle(render_mode="human")
    
    observations, infos = env.reset()
    # print(env.parcels_weight)
    delivery_upper_solver = upper_solver(observations["truck"]["observation"]["pos_obs"], num_customer_both, num_parcels_truck, num_parcels_uav, num_uavs)

    env.render()
    
    # TA_Scheduling_action_random = delivery_upper_solver.solve(observations["truck"]["observation"], infos, env.num_uavs, env.uav_range)
    # TA_Scheduling_action_greedy = delivery_upper_solver.solve_greedy(observations["truck"]["observation"], infos, env.num_uavs, env.uav_range)
    
    for i in range(50):
        # this is where you would insert your policy
        if infos["truck"] or (i % 6 == 0):
            TA_Scheduling_action = delivery_upper_solver.solve_greedy(observations["truck"]["observation"], infos, uav_range)
            env.TA_Scheduling(TA_Scheduling_action)
        
        actions = {
            # here is situated the policy
            agent: sample_action(env, observations, agent)
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        env.render()

    # print("pass")
    
    env.close()
    
