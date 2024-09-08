import os
import numpy as np
import pandas as pd
from re import match, findall, search
from copy import copy, deepcopy
import random
from env.delivery_env_with_obstacle import DeliveryEnvironmentWithObstacle

from ray.rllib.algorithms.sac import SACConfig, SAC
from ray.tune import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

from train import env_creator
from base import CustomSACPolicyModel, CustomSACQModel
from CTDRSP_FL.assignment_routing_SA import assignment_routing_SA
from CTDRSP_FL.assignment_routing_VNS import assignment_routing_VNS
from CTDRSP_FL.drone_assignment_routing_heuristic import drone_assignment_routing
from MFSTSP.mfstsp import MFSTSPSolver
from JOCR.unrestricted_JOCR_solver import solve_JOCR_U

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
        uav_assignments_best, lower_solution_best, _, _, _ = drone_assignment_routing(solution_vns, nodes, distances_truck, truck_velocity, distances_uav, uavs, uav_velocity, uav_range)
        lower_solution_best[-1] = lower_solution_best[0]
        route_edge_dict = { lower_solution_best[i]:lower_solution_best[i + 1] for i in range(len(lower_solution_best) - 1) }
        assign_best_dict = { (uav, start): (cust - num_parcels_truck - 1, route_edge_dict[start]) for start, cust, uav in uav_assignments_best }

        return lower_solution_best, assign_best_dict


    def solve_MFSTSP(self, num_uav_0, num_uav_1, uav_velocity, uav_capacity, truck_velocity):
        # prepare parameters
        uavs = [0] * num_uav_0 + [1] * num_uav_1
        uav_range_mfstsp = ["low", "high"]
        uav_infos = [
			{
				'velocity': uav_velocity[0], 
				'range': uav_range_mfstsp[0], 
				'capacity': uav_capacity[0], 
			}, 
			{
				'velocity': uav_velocity[1], 
				'range': uav_range_mfstsp[1], 
				'capacity': uav_capacity[1], 
			}
        ]
        locations = np.concatenate(([self.warehouse_pos], self.customer_pos_truck))
        manhattan_dist_matrix = np.abs(locations[:, np.newaxis] - locations).sum(axis=2)
        masks = np.zeros([len(uav_velocity), self.num_customer + 1])
        masks[:, self.num_truck_customer + 1:] = 1
        
        solver = MFSTSPSolver(self.num_uavs, uavs, uav_infos, locations, manhattan_dist_matrix, truck_velocity, masks)
        # assign: [(uav, start, cust, end)]
        tour, assign = solver.get_solution()
        
        tour_dict = { start: end for start, end in tour }
        node = 0
        route = [ 0 ]
        for _ in range(len(tour) - 1):
            node = tour_dict[node]
            route.append(node)
        route.append( 0 )
        
        return route, assign


    def solve_JOCR(self, uav_velocity, uav_range, truck_velocity, MAP_SIZE, GRID_SIZE):
        # parameters initialization        
        nodes = [
            0, 
            list(range(1, self.num_truck_customer + 1)), 
            list(range(self.num_truck_customer + 1, self.num_customer + 1)), 
        ]
        locations = np.concatenate(([self.warehouse_pos], self.customer_pos_truck))

        G_max = self.num_uavs
        coef_k = int(np.ceil(self.num_customer / G_max) * 2)
        K_min = int(np.ceil(self.num_customer / G_max))
        K_max = min(K_min + coef_k, self.num_customer + 1)
        max_iter = 200
        
        _, f_p, assignments = solve_JOCR_U(nodes, locations, uav_range[0], uav_velocity[0], truck_velocity, MAP_SIZE, K_min, K_max, G_max, max_iter)
        
        # transfer focal points to its corresponding parking point on the road
        focal_points = []
        for focal in f_p:
            bias_x = focal[0] % GRID_SIZE
            bias_y = focal[1] % GRID_SIZE
            
            if bias_x == 0 or bias_y == 0:
                focal_points.append((int(focal[0]), int(focal[1])))
                continue
            
            bias_x = min(bias_x, GRID_SIZE - bias_x)
            bias_y = min(bias_y, GRID_SIZE - bias_y)
            
            if bias_x < bias_y:
                focal_points.append((round(focal[0] / GRID_SIZE) * GRID_SIZE, int(focal[1])))
            else:
                focal_points.append((int(focal[0]), round(focal[1] / GRID_SIZE) * GRID_SIZE))
        
        clusters = [
            [] for _ in range(len(focal_points))
        ]
        for idx, label in enumerate(assignments):
            clusters[label].append(idx)
        
        # print(focal_points, '\n', clusters)
        
        # eliminate the empty cluster
        focal_points_organized = []
        clusters_organized = []
        for i in range(len(focal_points)):
            if clusters[i]:
                clusters_organized.append(clusters[i])
                focal_points_organized.append(focal_points[i])
        
        focal_points_organized.append(focal_points_organized[0])
        clusters_organized.append([])
        # print(focal_points_organized, '\n', clusters_organized)
        return focal_points_organized, clusters_organized


def heuristic_lower_policy(obs, agent):
    angle = np.arctan2(obs[1], obs[0])
    angle = angle if angle >= 0 else angle + 2 * np.pi
    
    norm_angle = (angle / np.pi) - 1
    
    uav_type = int(findall(r'\d+', agent)[0])
    
    v_max = uav_velocity[uav_type]
    v = (2 * min(np.linalg.norm(obs) / step_len, v_max)) / v_max - 1
    
    return np.array([norm_angle, v])


def run_JOCR_exp(env, seed, max_iter=3_000, render=None, video_record=False):
    observations, infos = env.reset(seed=seed)
    num_collisions = [0, 0]
    infos = infos['is_ready']
    solver = upper_solver(observations["truck"]["pos_obs"], num_customer_both, num_parcels_truck, num_parcels_uav, num_uavs)
    focal_route, clusters = solver.solve_JOCR(uav_velocity, uav_range * 0.2, truck_velocity, map_size, grid_size)
    clusters_copy = deepcopy(clusters)
    
    cluster_idx = 0 # index of cluster
    num_clusters = len(focal_route)
    
    if infos["truck"] and clusters[cluster_idx]:
        schedule_actions = {
            'truck': np.array(focal_route[cluster_idx])
        }
        
    env.TA_Scheduling(schedule_actions)
    for i in range(max_iter):
        # this is where you would insert your upper policy
        schedule_actions.clear()
        if infos["truck"] and not clusters[cluster_idx]:
            increase = True
            for info in infos:
                if not infos[info] and match('uav', info):
                    increase = False
                    break
            if increase:
                cluster_idx += 1
                schedule_actions = {
                    'truck': np.array(focal_route[cluster_idx])
                }
                env.TA_Scheduling(schedule_actions)
                schedule_actions.clear()
                if clusters[cluster_idx] and clusters[cluster_idx][0] < num_parcels_truck:
                    clusters[cluster_idx].pop(0)
        
        if infos['truck']:
            for info in infos:
                if infos[info] and match('uav', info) and clusters[cluster_idx]:
                    schedule_actions[info] = clusters[cluster_idx].pop() - num_parcels_truck
        
        env.TA_Scheduling(schedule_actions)
        
        direction_vecs = {agent: observations[agent]['vecs'][:2] for agent in env.agents if match("uav", agent)}
        actions = {
            # here is situated the lower policy
            agent: heuristic_lower_policy(direction_vecs[agent], agent)
            for agent in env.agents if match("uav", agent) #  and not infos[agent]
        }

        observations, _, _, _, infos = env.step(actions)
        num_collisions[0] += infos['collisions_with_obstacle']
        num_collisions[1] += infos['collisions_with_uav']
        infos = infos['is_ready']
        
        if not env.agents:
            print("parking finish in : ", i)

            return i, num_collisions, np.concatenate(([solver.warehouse_pos], solver.customer_pos_truck)), focal_route, clusters_copy
        if render and (i % 5 == 0):
            env.render()
            
    return 0, [0, 0], solver.customer_pos_truck, focal_route, clusters_copy


def run_MFSTSP_exp(env, seed, max_iter=3_000, render=None, video_record=False):
    observations, infos = env.reset(seed=seed, options=2)
    num_collisions = [0, 0]
    infos = infos['is_ready']
    solver = upper_solver(observations["truck"]["pos_obs"], num_customer_both, num_parcels_truck, num_parcels_uav, num_uavs)
    # assignments: [(uav, start, customer, end)]
    route, assignments = solver.solve_MFSTSP(num_uavs_0, num_uavs_1, uav_velocity, uav_capacity, truck_velocity)
    assign_dict = {(uav, start): (cust - num_parcels_truck - 1, end % (num_parcels + 1)) for uav, start, cust, end in assignments}
    assign_dict_copy = copy(assign_dict)
    
    schedule_actions = {}
    launchs = { stop: [] for stop in route[1:] }
    current = 0
    for i in range(max_iter):
        # this is where you would insert your upper policy
        schedule_actions.clear()
        
        if infos['truck']:
            for info in infos:
                if infos[info] and match('uav', info):
                    if info in launchs[route[current]]:
                        launchs[route[current]].remove(info)
                    uav_info = findall(r'\d+', info)
                    uav_info = [int(num) for num in uav_info]
                    uav_no = uav_info[0] * num_uavs_0 + uav_info[1]
                    if (uav_no, route[current]) in assign_dict:
                        cust, end = assign_dict.pop((uav_no, route[current]))
                        schedule_actions[info] = (cust, end)
                        launchs[end].append(info)

            if not launchs[route[current]] and current < len(route) - 1:
                current += 1
                schedule_actions['truck'] = route[current]
        
        env.TA_Scheduling(schedule_actions)
        
        direction_vecs = {agent: observations[agent]['vecs'][:2] for agent in env.agents if match("uav", agent)}
        actions = {
            # here is situated the lower policy
            agent: heuristic_lower_policy(direction_vecs[agent], agent)
            for agent in env.agents if match("uav", agent) #  and not infos[agent]
        }

        observations, _, _, _, infos = env.step(actions)
        num_collisions[0] += infos['collisions_with_obstacle']
        num_collisions[1] += infos['collisions_with_uav']
        infos = infos['is_ready']
        
        if not env.agents:
            print("parking finish in : ", i)

            return i, num_collisions, np.concatenate(([solver.warehouse_pos], solver.customer_pos_truck)), route, assign_dict_copy
        if render and (i % 5 == 0):
            env.render()
            
    return 0, [0, 0], np.concatenate(([solver.warehouse_pos], solver.customer_pos_truck)), route, assign_dict_copy


def run_CTDRSP_exp(env, seed, max_iter=3_000, render=None, video_record=False):
    observations, infos = env.reset(seed=seed, options=2)
    num_collisions = [0, 0]
    infos = infos['is_ready']
    solver = upper_solver(observations["truck"]["pos_obs"], num_customer_both, num_parcels_truck, num_parcels_uav, num_uavs)
    route, assign_dict = solver.solve_CTDRSP(num_uavs_0, num_uavs_1, uav_velocity, uav_range, truck_velocity)
    assign_dict_copy = copy(assign_dict)
    
    # simplicity available
    schedule_actions = {}
    launchs = { stop: [] for stop in route[1:] }
    current = 0
    for i in range(max_iter):
        # this is where you would insert your upper policy
        schedule_actions.clear()
        
        if infos['truck']:
            for info in infos:
                if infos[info] and match('uav', info):
                    if info in launchs[route[current]]:
                        launchs[route[current]].remove(info)
                    uav_info = findall(r'\d+', info)
                    uav_info = [int(num) for num in uav_info]
                    uav_no = uav_info[0] * num_uavs_0 + uav_info[1]
                    if (uav_no, route[current]) in assign_dict:
                        cust, end = assign_dict.pop((uav_no, route[current]))
                        schedule_actions[info] = (cust, end)
                        launchs[end].append(info)

            if not launchs[route[current]] and current < len(route) - 1:
                current += 1
                schedule_actions['truck'] = route[current]
        
        env.TA_Scheduling(schedule_actions)
        
        direction_vecs = {agent: observations[agent]['vecs'][:2] for agent in env.agents if match("uav", agent)}
        actions = {
            # here is situated the lower policy
            agent: heuristic_lower_policy(direction_vecs[agent], agent)
            for agent in env.agents if match("uav", agent) #  and not infos[agent]
        }

        observations, _, _, _, infos = env.step(actions)
        num_collisions[0] += infos['collisions_with_obstacle']
        num_collisions[1] += infos['collisions_with_uav']
        infos = infos['is_ready']
        
        if not env.agents:
            print("parking finish in : ", i)

            return i, num_collisions, np.concatenate(([solver.warehouse_pos], solver.customer_pos_truck)), route, assign_dict_copy
        if render and (i % 10 == 0):
            env.render()
            
    return 0, [0, 0], np.concatenate(([solver.warehouse_pos], solver.customer_pos_truck)), route, assign_dict_copy


def run_hierarchical_exp(env, model, seed, max_iter=2_000, render=None, video_record=False):
    num_collisions = [0, 0]

    observations, infos = env.reset(seed=seed)
    delivery_upper_solver = upper_solver(observations["truck"]["pos_obs"], num_customer_both, num_parcels_truck, num_parcels_uav, num_uavs)

    TA_Scheduling_action = delivery_upper_solver.solve_greedy(observations["truck"], infos['is_ready'], uav_range)

    env.TA_Scheduling(TA_Scheduling_action)
    observations, _, _, _, infos = env.step({})
    
    for i in range(max_iter):
        # this is where you would insert your policy
        if infos['is_ready']["truck"] or (i % 5 == 4):
            TA_Scheduling_action = delivery_upper_solver.solve_greedy(observations["truck"], infos['is_ready'], uav_range)
            env.TA_Scheduling(TA_Scheduling_action)

        actions = {
            # here is situated the policy
            agent: model.compute_single_action(
                observation=observations[agent], 
                policy_id='masac_policy'
            )
            for agent in env.agents if match("uav", agent) #  and not infos[agent]
        }
        observations, _, _, _, infos = env.step(actions)
        
        num_collisions[0] += infos['collisions_with_obstacle']
        num_collisions[1] += infos['collisions_with_uav']
        
        if not env.agents:
            print("finish in : ", i)

            return i, num_collisions, np.concatenate(([delivery_upper_solver.warehouse_pos], delivery_upper_solver.customer_pos_truck))
        if render and i % 8 == 0:
            env.render()
    
    return 0, [0, 0], np.concatenate(([delivery_upper_solver.warehouse_pos], delivery_upper_solver.customer_pos_truck))


if __name__ == "__main__":    
    step_len = 2
    map_size = 5_000
    grid_size = 125
    # uav parameters
    # unit here is m/s
    truck_velocity = 6
    uav_velocity = np.array([8, 12])
    # unit here is kg
    uav_capacity = np.array([10, 3.6])
    # unit here is m
    uav_range = np.array([10_000, 15_000])
    uav_obs_range = 150
    
    num_truck = 1
    num_uavs_0 = 1
    num_uavs_1 = 1
    num_uavs = num_uavs_0 + num_uavs_1
    
    # parcels parameters
    num_parcels = 30
    num_parcels_truck = 6
    num_parcels_uav = 4
    num_customer_both = num_parcels - num_parcels_truck - num_parcels_uav
    
    # obstacle parameters
    num_uav_obstacle = 0
    num_no_fly_zone = 0
    
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
    
    base_dir = os.path.join('experiments')
    existing_folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    data_folders = [name for name in existing_folders if match(r'^data_\d+$', name)]
    
    if data_folders:
        max_num = max(int(search(r'\d+', folder).group()) for folder in data_folders)
    else:
        max_num = 0
    
    exp_folder_name = f"data_{max_num + 1}"
    exp_folder_path = os.path.join(base_dir, exp_folder_name)
    
    os.makedirs(exp_folder_path)
    
    # column_params = ['map_size', 'grid_size', 'truck_velocity']
    # column_uav_params = ['uav_no', 'velocity', 'range', 'capacity']

    columns_CTDRSP = ['exp_name', 'seed', 'locations', 'num_uavs', 'num_customers', 'num_obstacles', 'route', 'assignments', 'makespan', 'collisions_obstacle', 'collision_uavs']
    df_CTDRSP = pd.DataFrame(columns=columns_CTDRSP)
    columns_MFSTSP = ['exp_name', 'seed', 'locations', 'num_uavs', 'num_customers', 'num_obstacles', 'route', 'assignments', 'makespan', 'collisions_obstacle', 'collision_uavs']
    df_MFSTSP = pd.DataFrame(columns=columns_MFSTSP)
    columns_JOCR = ['exp_name', 'seed', 'locations', 'num_uavs', 'num_customers', 'num_obstacles', 'focal_route', 'cluster_assignments', 'makespan', 'collisions_obstacle', 'collision_uavs']
    df_JOCR = pd.DataFrame(columns=columns_JOCR)
    
    num_experiments = 2
    seed_range = 20_021_122
    seed_seq = np.random.randint(1, seed_range, size=num_experiments)
    print(seed_seq)

    for exp_no in range(num_experiments):
        seed = int(seed_seq[exp_no])
        
        # **CTDRSP**
        makespan_CTDRSP, collisions_CTDRSP, locations_CTDRSP, route_CTDRSP, assignments_CTDRSP = run_CTDRSP_exp(env, seed, render=False)
        CTDRSP_exp_row = pd.DataFrame([{
            'exp_name': f"CTDRSP_{exp_no}", 
            'seed': seed, 
            'locations': locations_CTDRSP, # { 'x': locations_CTDRSP[:, 0], 'y': locations_CTDRSP[:, 1] }, 
            'num_uavs': { 'num_uavs_0': num_uavs_0, 'num_uavs_1': num_uavs_1 }, 
            'num_customers': { 'truck': num_parcels_truck, 'both': num_customer_both, 'uav': num_parcels_uav }, 
            'num_obstacles': { 'buildings': num_uav_obstacle, 'no_fly_zone': num_no_fly_zone }, 
            'route': route_CTDRSP, 
            'assignments': assignments_CTDRSP, 
            'makespan': makespan_CTDRSP, 
            'collisions_obstacle': collisions_CTDRSP[0], 
            'collision_uavs': collisions_CTDRSP[1],
        }], columns=columns_CTDRSP)
        df_CTDRSP = pd.concat([df_CTDRSP, CTDRSP_exp_row], ignore_index=True)

        # **MFSTSP**
        makespan_MFSTSP, collisions_MFSTSP, locations_MFSTSP, route_MFSTSP, assignments_MFSTSP = run_MFSTSP_exp(env, seed, render=False)
        MFSTSP_exp_row = pd.DataFrame([{
            'exp_name': f"MFSTSP_{exp_no}", 
            'seed': seed, 
            'locations': locations_MFSTSP, 
            'num_uavs': { 'num_uavs_0': num_uavs_0, 'num_uavs_1': num_uavs_1 }, 
            'num_customers': { 'truck': num_parcels_truck, 'both': num_customer_both, 'uav': num_parcels_uav }, 
            'num_obstacles': { 'buildings': num_uav_obstacle, 'no_fly_zone': num_no_fly_zone }, 
            'route': route_MFSTSP, 
            'assignments': assignments_MFSTSP, 
            'makespan': makespan_MFSTSP, 
            'collisions_obstacle': collisions_MFSTSP[0], 
            'collision_uavs': collisions_MFSTSP[1],
        }])
        df_MFSTSP = pd.concat([df_MFSTSP, MFSTSP_exp_row], ignore_index=True)

        # **JOCR**
        makespan_JOCR, collisions_JOCR, locations_JOCR, focal_route, clusters = run_JOCR_exp(env, seed, render=False)
        JOCR_exp_row = pd.DataFrame([{
            'exp_name': f"JOCR_{exp_no}", 
            'seed': seed, 
            'locations': locations_JOCR, 
            'num_uavs': { 'num_uavs_0': num_uavs_0, 'num_uavs_1': num_uavs_1 }, 
            'num_customers': { 'truck': num_parcels_truck, 'both': num_customer_both, 'uav': num_parcels_uav }, 
            'num_obstacles': { 'buildings': num_uav_obstacle, 'no_fly_zone': num_no_fly_zone }, 
            'focal_route': focal_route, 
            'cluster_assignments': clusters, 
            'makespan': makespan_JOCR, 
            'collisions_obstacle': collisions_JOCR[0], 
            'collision_uavs': collisions_JOCR[1],
        }])
        df_JOCR = pd.concat([df_JOCR, JOCR_exp_row], ignore_index=True)
    
    df_CTDRSP.to_csv(os.path.join(exp_folder_path, "CTDRSP_experiments.csv"))
    df_MFSTSP.to_csv(os.path.join(exp_folder_path, "MFSTSP_experiments.csv"))
    df_JOCR.to_csv(os.path.join(exp_folder_path, "JOCR_experiments.csv"))
    
    # **OURS**
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
    
    columns_OURS = ['exp_name', 'seed', 'locations', 'num_uavs', 'num_customers', 'num_obstacles', 'makespan', 'collisions_obstacle', 'collision_uavs']
    df_OURS = pd.DataFrame(columns=columns_OURS)
    for exp_no in range(num_experiments):
        makespan_OURS, collisions_OURS, locations_OURS = run_hierarchical_exp(env, masac_agent, seed, render=False)
        exp_row = pd.DataFrame([{
            'exp_name': f"OURS_{exp_no}", 
            'seed': seed, 
            'locations': locations_OURS, 
            'num_uavs': { 'num_uavs_0': num_uavs_0, 'num_uavs_1': num_uavs_1 }, 
            'num_customers': { 'truck': num_parcels_truck, 'both': num_customer_both, 'uav': num_parcels_uav }, 
            'num_obstacles': { 'buildings': num_uav_obstacle, 'no_fly_zone': num_no_fly_zone }, 
            'makespan': makespan_OURS, 
            'collisions_obstacle': collisions_OURS[0], 
            'collision_uavs': collisions_OURS[1],
        }])
        df_OURS = pd.concat([df_OURS, exp_row], ignore_index=True)
    df_OURS.to_csv(os.path.join(exp_folder_path, "OURS_experiments.csv"))

    env.close()
