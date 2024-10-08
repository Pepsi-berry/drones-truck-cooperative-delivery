import os
import numpy as np
from re import match, findall
import random
from itertools import combinations
from env.delivery_env_with_obstacle import DeliveryEnvironmentWithObstacle
# from env.uav_env import UAVTrainingEnvironmentWithObstacle
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecVideoRecorder
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv
# from tsp_solver import solve_tsp
# from customer_clustering_solver import solve_drones_truck_with_parking
import pandas as pd

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.fx.all import resize
from PIL import Image as im

MAX_INT = 100

def save_video(
    frames: list,
    video_folder: str,
    i,
    name,
    fps=60,
):
    video_folder = os.path.abspath(video_folder)
    os.makedirs(video_folder, exist_ok=True)
    video_path = f"{video_folder}/{name}_{i}fps.mp4"
    clip = ImageSequenceClip(frames, fps=fps)
    # making sure even dimensions
    width = clip.w if (clip.w % 2 == 0) else clip.w - 1
    height = clip.h if (clip.h % 2 == 0) else clip.h - 1
    clip = resize(clip, newsize=(width, height))
    clip.write_videofile(video_path)


def save_last_frame(frame, image_folder, i, name):
    image_folder = os.path.abspath(image_folder)
    os.makedirs(image_folder, exist_ok=True)
    image_path = f"{image_folder}/{name}{i}.png"
    # Saving as a PNG file
    img = im.fromarray(frame)
    img.save(image_path)


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
# class PPO():
#     def __init__(self) -> None:
#         pass

def run_hierarchical_policy_solution(env, model, seed, max_iter=2_000, render=None, video_record=False):
    video_frame = []
    observations, infos = env.reset(seed=seed)
    solver = upper_solver(observations["truck"]["pos_obs"], num_customer_both, num_parcels_truck, num_parcels_uav, num_uavs)
    # env.render()

    TA_Scheduling_action = solver.solve_greedy(observations["truck"], infos, uav_range)
    env.TA_Scheduling(TA_Scheduling_action)
    observations, rewards, terminations, truncations, infos = env.step({})
    
    for i in range(max_iter):
        # this is where you would insert your policy
        if infos["truck"] or (i % 6 == 5):
            TA_Scheduling_action = solver.solve_greedy(observations["truck"], infos, uav_range)
            # print(TA_Scheduling_action)
            env.TA_Scheduling(TA_Scheduling_action)
        # print([
        #     observations[a]
        #     for a in observations if match('uav', a)
        # ])
        actions = {
            # here is situated the policy
            # agent: sample_action(env, observations, agent)
            agent: model.predict(observations[agent], deterministic=True)[0]
            for agent in env.agents if match("uav", agent) #  and not infos[agent]
        }
        # print(actions)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        if not env.agents:
            # print("finish in : ", i)
            if video_record:
                video_name = save_video(video_frame, "hierarchical_video", 15, 'delivery_env', fps=15)
                last_frame_location = save_last_frame(video_frame[-1], "last_frame", i, 'delivery_env')
                print('video_name: ', video_name)
                print('last_frame_location: ', last_frame_location)
            return i
        if render and (i % 4 == 0):
            env.render()
        elif video_record:
            video_frame.append(env.render())
            
    return 0


def run_tsp_solution(env, seed, max_iter=3_000, render=None, video_record=False):
    video_frame = []
    observations, infos = env.reset(seed=seed, options=1)
    solver = upper_solver(observations["truck"]["pos_obs"], num_customer_both, num_parcels_truck, num_parcels_uav, num_uavs)
    route, _ = solver.solve_TSP()
    route = reshape_tsp_route(route)
    
    route.reverse()
    for i in range(max_iter):
        # this is where you would insert your upper policy
        if infos["truck"]:
            schedule_actions = {
                'truck': route.pop()
            }
            
        env.TA_Scheduling(schedule_actions)
        # actions = {
        #     # here is situated the lower policy
        #     agent: model.predict(observations[agent], deterministic=True)[0]
        #     for agent in env.agents if match("uav", agent) #  and not infos[agent]
        # }
        observations, rewards, terminations, truncations, infos = env.step({'truck':None})
        
        if not env.agents:
            # print("tsp finish in : ", i)
            if video_record:
                video_name = save_video(video_frame, "tsp_video", 15, 'delivery_env', fps=15)
                last_frame_location = save_last_frame(video_frame[-1], "last_frame", i, 'delivery_env')
                print('video_name: ', video_name)
                print('last_frame_location: ', last_frame_location)
            return i
        if render and (i % 4 == 0):
            env.render()
        elif video_record:
            video_frame.append(env.render())
    
    return 0
            

def run_carried_vehicle_supporting_drone_solution(env, model, seed, max_iter=3_000, render=None, video_record=False):
    video_frame = []
    observations, infos = env.reset(seed=seed)
    solver = upper_solver(observations["truck"]["pos_obs"], num_customer_both, num_parcels_truck, num_parcels_uav, num_uavs)
    route_truck, clusters, centroids = solver.solve_parking(grid_edge)
    # print("clusters: ", clusters)
    # print("centroids: ", centroids)
    # print("route: ", route_truck)
    
    this_cluster = route_truck.pop()
    if infos["truck"] and clusters[this_cluster]:
        schedule_actions = {
            'truck': centroids[this_cluster]
        }
    
    # for info in infos:
    #     # consider all the uav with the same range of 15_000 m
    #     if infos[info] and match('uav', info) and clusters[this_cluster]:
    #         schedule_actions[info] = clusters[this_cluster].pop()
        
    env.TA_Scheduling(schedule_actions)
    for i in range(max_iter):
        # this is where you would insert your upper policy
        schedule_actions.clear()
        if infos["truck"] and not clusters[this_cluster]:
            increase = True
            # print(infos)
            for info in infos:
                if not infos[info] and match('uav', info):
                    increase = False
                    break
            if increase:
                this_cluster = route_truck.pop()
                schedule_actions = {
                    'truck': centroids[this_cluster]
                }
                env.TA_Scheduling(schedule_actions)
                schedule_actions.clear()
                if this_cluster <= num_parcels_truck and this_cluster:
                    clusters[this_cluster].pop(0)
        
        if infos['truck']:
            for info in infos:
                # consider all the uav with the same range of 15_000 m
                if infos[info] and match('uav', info) and clusters[this_cluster]:
                    schedule_actions[info] = clusters[this_cluster].pop() - num_parcels_truck - 1
                # if not infos[info] and match('uav', info):
                #     schedule_actions.pop('truck')
        
        # print(schedule_actions)
        env.TA_Scheduling(schedule_actions)
        
        actions = {
            # here is situated the lower policy
            agent: model.predict(observations[agent], deterministic=True)[0]
            for agent in env.agents if match("uav", agent) #  and not infos[agent]
        }

        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # print(infos)
        
        if not env.agents:
            # print("parking finish in : ", i)
            if video_record:
                video_name = save_video(video_frame, "parking_video", 15, 'delivery_env', fps=15)
                last_frame_location = save_last_frame(video_frame[-1], "last_frame", i, 'delivery_env')
                print('video_name: ', video_name)
                print('last_frame_location: ', last_frame_location)
            return i
        if render and (i % 4 == 0):
            env.render()
        elif video_record:
            video_frame.append(env.render())
            
    return 0


if __name__ == "__main__":
    
    seed_range = 20_021_122
    
    possible_agents = [
        "truck", 
        "uav_0_0", "uav_0_1", 
        "uav_1_0", "uav_1_1", 
        "uav_1_2", "uav_1_3", 
    ]
    
    step_len = 2
    # uav parameters
    # unit here is m/s
    truck_velocity = 4
    uav_velocity = np.array([12, 29])
    # unit here is kg
    uav_capacity = np.array([10, 3.6])
    # unit here is m
    uav_range = np.array([10_000, 15_000])
    uav_obs_range = 150
    
    num_truck = 1
    num_uavs_0 = 2
    num_uavs_1 = 2
    num_uavs = num_uavs_0 + num_uavs_1
    
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
    grid_edge = 125 # m as unit here
    
    # obstacle parameters
    num_uav_obstacle = 1
    num_no_fly_zone = 1
    
    model_path = os.path.join("training", "models", "best_model_SAC_20K")
    model = SAC.load(model_path)
    
    # randList = [99112, 39566, 26912, 97613, 100615, 91316, 91792, 50701, 83019, 112200, 47254, 78875, 38088, 21103, 44819]
    ep_num = 1
    seed_seq = np.random.randint(1, seed_range, size=ep_num)
    seed_seq = [597215]
    # seed_seq = [# 12831503, 8783665, 13697239, 17765942, 5782998, 
    #             8324678, # 5859629, 3410441, 6293047, 17758899, 
    #             # 17855103, 9974297, 19783562, 15393630, 11538025, 
    #             # 14505477, 12396531,  7001911, 13342663, 12587394, 
    #             # 2925388, 8120709, 11921896, 17496592, 10244225, 
    #             # 5900518, 18898807,  4809629,  7023294, 16910173, 15270425, 19131162,
    #             # 17327394, 12642687,  7028938, 19562665, 16276660, 13756282,  3555379,  3015676,
    #             # 13257574, 19678909,  6546734,  5069702,  9966370, 13302556, 13478283,  3893352,
    #             # 4994314, 1590666
    #             ]
    
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
    
    # define the structure of result frame
    results = {
        'Solution': [],
        'customer_params': [],
        'uav_num': [], 
        'obstacle_params': [],
        'time_len_mean': [],
        'time_len_std': [],
        # 'success rate': [],
    }

    # convert to DataFrame
    results_df = pd.DataFrame(results)
    
    for num_parcels, num_parcels_truck, num_parcels_uav in customer_params_set:
        num_customer_truck = num_parcels - num_parcels_uav
        num_customer_uav = num_parcels - num_parcels_truck
        num_customer_both = num_parcels - num_parcels_truck - num_parcels_uav
        for num_uavs_0, num_uavs_1 in uav_params_set:
            num_uavs = num_uavs_0 + num_uavs_1
            for num_uav_obstacle, num_no_fly_zone in obstacle_params_set:
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
                    render_mode="rgb_array"
                )
                
                if env.render_mode == "rgb_array":
                    video_record = True
                else:
                    video_record = False
                    
                print(video_record)
                
                time_len_list = []
                time_len_parking_list = []
                time_len_tsp_list = []

                for i in range(ep_num):
                # 47899, 108221, 103327, 12512, 65758
                    seed = int(seed_seq[i])
                    print(seed)
                    
                    time_len_tsp_list = np.append(time_len_tsp_list, [run_tsp_solution(env, seed, render=None, video_record=video_record)])

                    time_len_list = np.append(time_len_list, [run_hierarchical_policy_solution(env, model, seed, render=None, video_record=video_record)])
                        
                    result = run_carried_vehicle_supporting_drone_solution(env, model, seed, render=None, video_record=video_record)
                    if result:
                        time_len_parking_list = np.append(time_len_parking_list, [result])
                
                # print(num_parcels, num_parcels_truck, num_parcels_uav, num_uavs_0, num_uavs_1, num_uav_obstacle, num_no_fly_zone)
                
                print("time len mean: ", np.mean(time_len_list))
                print("parking time len mean: ", np.mean(time_len_parking_list))
                # print("parking time len std: ", np.std(time_len_list))
                print("tsp time len mean: ", np.mean(time_len_tsp_list))
                
                new_row = pd.DataFrame([
                    {
                        'Solution': 'hierarchical_policy_solution',
                        'customer_params': {'num_parcels_truck': num_parcels_truck, 
                                            'num_parcels_uav': num_parcels_uav},
                        'uav_num': num_uavs, 
                        'obstacle_params': {'num_uav_obstacle': num_uav_obstacle, 
                                            'num_no_fly_zone': num_no_fly_zone},
                        'time_len_mean': np.mean(time_len_list),
                        'time_len_std': np.std(time_len_list),
                    },
                    # {
                    #     'Solution': 'carried_vehicle_supporting_drone_solution',
                    #     'customer_params': {'num_parcels_truck': num_parcels_truck, 
                    #                         'num_parcels_uav': num_parcels_uav},
                    #     'uav_num': num_uavs, 
                    #     'obstacle_params': {'num_uav_obstacle': num_uav_obstacle, 
                    #                         'num_no_fly_zone': num_no_fly_zone},
                    #     'time_len_mean': np.mean(time_len_list),
                    #     'time_len_std': np.std(time_len_list),
                    # },
                #     {
                #         'Solution': 'tsp_solution',
                #         'customer_params': {'num_parcels_truck': num_parcels_truck, 
                #                             'num_parcels_uav': num_parcels_uav},
                #         'uav_num': num_uavs, 
                #         'obstacle_params': {'num_uav_obstacle': num_uav_obstacle, 
                #                             'num_no_fly_zone': num_no_fly_zone},
                #         'time_len_mean': time_len_tsp / ep_num,
                #     },
                ])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                env.close()
                
    # save as .CSV file
    results_df.to_csv('experiment_results.csv', index=False)

