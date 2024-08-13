import os
import numpy as np
from re import match, findall
import random
from env.delivery_env_with_obstacle import DeliveryEnvironmentWithObstacle

from ray.rllib.algorithms.sac import SACConfig, SAC
from ray.tune import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

# from tsp_solver import solve_tsp
# from customer_clustering_solver import solve_drones_truck_with_parking
from train import env_creator
from base import CustomSACPolicyModel, CustomSACQModel
from CTDRSP_FL.assignment_routing_SA import assignment_routing_SA
from CTDRSP_FL.assignment_routing_VNS import assignment_routing_VNS
from CTDRSP_FL.drone_assignment_routing_heuristic import drone_assignment_routing

MAX_INT = 100
        

def reshape_tsp_route(route):
    if 0 not in route:
        return route  # return original list if there is no 0 there (or throw an error?)

    zero_index = route.index(0)  # find the index of 0 in list
    return route[zero_index + 1:] + route[:zero_index + 1]  # reshape


class upper_solver():
    def __init__(self, pos_obs, 
                 num_both_customer=10, 
                 num_truck_customer=4, 
                 num_uav_customer=6, 
                 num_uavs=6) -> None:
        
        self.num_both_customer = num_both_customer
        self.num_truck_customer = num_truck_customer
        self.num_uav_customer = num_uav_customer
        self.num_customer = self.num_both_customer + self.num_truck_customer + self.num_uav_customer
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
        
        task_truck_queue = [ np.int64(0) ]
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
        
        task_truck_queue = [ np.int64(0) ]
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
        
        truck_task_dists = task_dists[ : self.num_truck_customer + self.num_both_customer]
        both_task_dists = task_dists[self.num_truck_customer : self.num_truck_customer + self.num_both_customer]
        uav_task_dists = task_dists[self.num_truck_customer + self.num_both_customer : ]
        
        truck_task_dists_order = np.argsort(truck_task_dists)
        both_task_dists_order = np.argsort(both_task_dists)
        uav_task_dists_order = np.argsort(uav_task_dists)
        
        truck_task_dists_order = np.concatenate([truck_task_dists_order, 
                                                #  both_task_dists_order + self.num_truck_customer, 
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
            if uav_task_dists[idx] < uav_range[0] * 0.1 and uav_action_masks[0][idx]:
                task_uav_0_queue.append(idx)
            if uav_task_dists[idx] < uav_range[1] * 0.1 and uav_action_masks[1][idx]:
                task_uav_1_queue.append(idx)
        
        # task_uav_0_queue.append(-1)
        # task_uav_1_queue.append(-1)
        
        # print("wh:", self.warehouse_pos, "truck:", self.customer_pos_truck, "uav:", self.customer_pos_uav)
        # print(truck_queue, uav_0_avail_queue, uav_1_avail_queue, uav_NA_queue, task_truck_queue)
        # print(task_uav_0_queue, task_uav_1_queue)
        
        actions = {}
        
        # assigned the shortest customer point in available task queue to available agent
        # make that only one uav is assigned a task at a time step
        # to reducing conflicts between uavs
        if uav_0_avail_queue: 
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
                    break
        else: 
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
                    break
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
    
    
    def build_CTDRSP(self):
        num_flexible_location = self.num_customer * 1
        nodes = [
            0, 
            list(range(1, self.num_truck_customer + 1)), 
            list(range(self.num_truck_customer + 1, self.num_customer + 1)), 
            list(range(self.num_customer + 1, self.num_customer + num_flexible_location + 1)), 
            self.num_customer + num_flexible_location + 1
        ]
        # setting seed is not supported currently.
        locations = [tuple(self.warehouse_pos)] + [ tuple(pos) for pos in self.customer_pos_truck ] + [(random.randint(1_000, 4_000), random.randint(1_000, 4_000)) for _ in range(num_flexible_location)] + [tuple(self.warehouse_pos)]
        
        # distance matrix calculation
        # respectively manhattan distance and euclidean distance
        distances_truck = [
            [sum(abs(loc[k] - lo[k]) for k in range(2)) for lo in locations]
            for loc in locations
        ]
        distances_uav = {
            i_loc: [int(np.sqrt(sum((locations[i_loc][k] - lo[k])**2 for k in range(2)))) for lo in locations]
            for i_loc in nodes[2]
        }
        
        return nodes, locations, distances_truck, distances_uav
    
    
    def solve_CTDRSP(self, num_uav_0, num_uav_1, uav_velocity, uav_range, truck_velocity):
        uavs = [ 0 ] * num_uav_0 + [ 1 ] * num_uav_1
        nodes, locations, distances_truck, distances_uav = self.build_CTDRSP()
        # print(nodes, locations)
        S_hat = int(self.num_customer * 0.5)
        T_max = 660
        T_min = 1e-2
        K = 0.94
        iter_max = 22
        
        iter_max_inner = 30
        iter_max_outer = 15
        
        solution_sa, value_sa, _, _ = assignment_routing_SA(
            nodes, 
            locations, 
            distances_truck, 
            distances_uav, 
            uavs, 
            uav_velocity, 
            uav_range, 
            truck_velocity, 
            S_hat, T_max, T_min, K, iter_max
        )
        print(solution_sa, value_sa)
        
        solution_vns, value_vns, _, _ = assignment_routing_VNS(
            solution_sa, 
            value_sa, 
            nodes, 
            distances_truck, 
            truck_velocity, 
            distances_uav, 
            uavs, 
            uav_velocity, 
            uav_range, 
            iter_max_inner, iter_max_outer
        )
        print(solution_vns, value_vns)
        print(drone_assignment_routing(solution_vns, nodes, distances_truck, truck_velocity, distances_uav, uavs, uav_velocity, uav_range)[0])
        
        # visual CTDRSP solution
        # import matplotlib.pyplot as plt
        # route = [locations[i] for i in solution_vns]
        # x_values, y_values = zip(*locations[:self.num_customer + 1])
        # x_r, y_r = zip(*route)

        # plt.figure(figsize=(8, 6))

        # plt.scatter(x_values, y_values, color='blue')
        # plt.plot(x_r, y_r, color='red', marker='o', label='B Path')

        # for i, (x, y) in enumerate(locations[:self.num_customer + 1]):
        #     plt.text(x, y, f'{i}', fontsize=12, ha='right')

        # plt.xlabel('X')
        # plt.ylabel('Y')

        # plt.grid(True)

        # plt.show()


# probably we need to implement the marl algorithm by ourselves :(
# class PPO():
#     def __init__(self) -> None:
#         pass


if __name__ == "__main__":    
    step_len = 2
    # uav parameters
    # unit here is m/s
    truck_velocity = 6
    uav_velocity = np.array([12, 12])
    # unit here is kg
    uav_capacity = np.array([10, 3.6])
    # unit here is m
    uav_range = np.array([10_000, 15_000])
    uav_obs_range = 150
    
    num_truck = 1
    num_uavs_0 = 4
    num_uavs_1 = 4
    num_uavs = num_uavs_0 + num_uavs_1
    
    # parcels parameters
    num_parcels = 30
    num_parcels_truck = 6
    num_parcels_uav = 4
    num_customer_both = num_parcels - num_parcels_truck - num_parcels_uav
    
    # obstacle parameters
    num_uav_obstacle = 1
    num_no_fly_zone = 1
    
    customer_params_set = [
        [20, 4, 6], 
        # [20, 6, 6], 
        # [10, 2, 2], 
        # [30, 6, 9], 
        # [30, 6, 12], 
        # [30, 5, 5], 
        # [40, 8, 12], 
        # [40, 10, 15], 
        # [40, 10, 10]
    ]
    uav_params_set = [
        # [1, 1], 
        # [1, 2], 
        [2, 2], 
        # [3, 2], 
        # [4, 2]
    ]
    obstacle_params_set = [
        [15, 2],
        # [1, 1], 
        # [5, 1], 
        # [10, 2]
    ]
    
    # env = env_creator({
    #     'render_mode': 'human', 
        
    #     'uav_velocity': np.array([12, 20]), 
    #     'num_uavs_0': 6, 
    #     'num_uavs_1': 6, 
    #     'num_uavs': 12, 

    #     'num_uav_obstacle': 8, 
    #     'num_no_fly_zone': 0, 
    # })

    # ray rllib style code
    register_env("ma_training_env", lambda config: ParallelPettingZooEnv(env_creator(config)))
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id.startswith("uav"):
            return "masac_policy"
        else:
            raise ValueError("Unknown agent type: ", agent_id)
    config = (
        SACConfig()
        .environment('ma_training_env')
        .env_runners(
            num_env_runners=4, 
            num_cpus_per_env_runner=2, 
            rollout_fragment_length='auto'
        )
        # .rollouts(num_rollout_workers=4, rollout_fragment_length=128) # deprecated, use env_runners instead、
        .experimental(
            _disable_preprocessor_api=True
        )
        .framework('torch')
        .checkpointing(export_native_model_files=True)
        .training(
            policy_model_config = {
                "custom_model": CustomSACPolicyModel,  # Use this to define custom Q-model(s).
            },
            q_model_config = {
                "custom_model": CustomSACQModel,  # Use this to define custom Q-model(s).
            },
            gamma=0.99, 
            lr=0.0003,
            train_batch_size=256, 
            num_steps_sampled_before_learning_starts=1_000, 
            )
        .multi_agent(
            policies={
                "masac_policy": PolicySpec(
                policy_class=None,  # infer automatically from Algorithm
                observation_space=None,  # infer automatically from env
                action_space=None,  # infer automatically from env
                config={},  # use main config plus <- this override here
                ),
            },
            policy_mapping_fn=policy_mapping_fn
        )
        .resources(num_gpus=0)
        )
    masac_agent = SAC(config=config)
    masac_agent.restore('training/models/SAC_best_checkpoint_12_1000_137')
        
    env = DeliveryEnvironmentWithObstacle(
        step_len=step_len, 
        truck_velocity=truck_velocity, 
        uav_velocity=uav_velocity, 
        uav_capacity=uav_capacity, 
        uav_range=uav_range, 
        uav_obs_range=uav_obs_range, 
        num_truck=num_truck, 
        num_uavs=num_uavs, 
        num_uavs_0=num_uavs_0, 
        num_uavs_1=num_uavs_1, 
        num_parcels=num_parcels, 
        num_parcels_truck=num_parcels_truck, 
        num_parcels_uav=num_parcels_uav, 
        num_uav_obstacle=num_uav_obstacle, 
        num_no_fly_zone=num_no_fly_zone, 
        render_mode="human"
    )
    
    # **CTDRSP**
    # observations, infos = env.reset()
    # delivery_upper_solver = upper_solver(observations["truck"]["pos_obs"], num_customer_both, num_parcels_truck, num_parcels_uav, num_uavs)
    
    # delivery_upper_solver.solve_CTDRSP(num_uavs_0, num_uavs_1, uav_velocity, uav_range * 0.4, truck_velocity)
    
    # **OURS**
    num_collisions = [0, 0]
    for _ in range(20):
        seed = random.randint(1, 20_021_122)

        observations, infos = env.reset(seed=seed)
        delivery_upper_solver = upper_solver(observations["truck"]["pos_obs"], num_customer_both, num_parcels_truck, num_parcels_uav, num_uavs)

        TA_Scheduling_action = delivery_upper_solver.solve_greedy(observations["truck"], infos['is_ready'], uav_range)
        env.TA_Scheduling(TA_Scheduling_action)
        observations, rewards, terminations, truncations, infos = env.step({})
        
        for i in range(2_000):
            # this is where you would insert your policy
            if infos['is_ready']["truck"] or (i % 5 == 4):
                TA_Scheduling_action = delivery_upper_solver.solve_greedy(observations["truck"], infos['is_ready'], uav_range)
                env.TA_Scheduling(TA_Scheduling_action)

            actions = {
                # here is situated the policy
                agent: masac_agent.compute_single_action(
                    observation=observations[agent], 
                    policy_id='masac_policy'
                )
                for agent in env.agents if match("uav", agent) #  and not infos[agent]
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            num_collisions[0] += infos['collisions_with_obstacle']
            num_collisions[1] += infos['collisions_with_uav']
            
            if not env.agents:
                print("finish in : ", i)
                # print("conflict with obstacle ", num_collisions[0], " times.")
                # print("conflict with uav ", num_collisions[1], " times.")
                break
            # if i % 4 == 0:
            #     env.render()
    print("conflict with obstacle ", num_collisions[0], " times.")
    print("conflict with uav ", num_collisions[1], " times.")

    env.close()
