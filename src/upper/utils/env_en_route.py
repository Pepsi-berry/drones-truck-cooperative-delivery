import os
import functools
import random
from copy import copy
from re import findall
# from itertools import combinations

import numpy as np
from gymnasium.utils import seeding

# from pettingzoo import ParallelEnv
import pygame

MAX_INT = 2**20
INVALID_ANGLE = 10
# When the distance between the returning uav and the truck is less than this threshold, 
# the return is considered complete.
DIST_RESTRICT_UAV = 20
DIST_RESTRICT_OBSTACLE = 75
# reward in various situations
REWARD_DELIVERY = 20
REWARD_VICTORY = 100
REWARD_UAV_WRECK = -2
REWARD_UAV_VIOLATE = -2
REWARD_UAV_ARRIVAL = 20
REWARD_URGENCY = -1
REWARD_APPROUCHING = 0.02 # get REWARD_APPROUCHING when get closer to target
REWARD_UAVS_DANGER = float(-400) # coefficient of the penalty for being to close with other uavs
PENALTY_TRUNCATION = 4_000
# REWARD_SLOW = -0.02

TYPE_TRUCK = 0
TYPE_UAV = 1

# it seems that wrapping truck and uav into classes would make programming significantly less difficult...
# but when I realize this, it had gone much much too far...

# In once decision, there may be more than one uav selected the same customer as the target point
# special treatment of this needs to be refined(maybe AEV is more suitable? :(

class UpperSolverTrainingEnvironment():
    """The metadata holds environment constants.
    
    The "name" metadata allows the environment to be pretty printed.
    
    """

    metadata = {
        "render_modes": [None, "human", "rgb_array"],
        "name": "upper_training_environment_v2",
    }

    def __init__(
        self, 
        MAX_STEP=2_000, 
        step_len=10, 
        truck_velocity=7, 
        uav_velocity=np.array([12, 29]), 
        uav_capacity=np.array([10, 3.6]), 
        uav_range=np.array([10_000, 15_000]), 
        num_truck=1, 
        num_uavs_0=2, 
        num_uavs_1=4, 
        num_parcels=20, 
        num_parcels_truck=4, 
        render_mode=None
        ):
        """The init method takes in environment arguments.


        These attributes should not be changed after initialization.
        """
        ##########################################################################################
        # These attributes should not be changed after initialization except time_step.
        self.render_mode = render_mode
        self.screen = None
        self.RNG, _ = seeding.np_random()
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()
                
        self.MAX_STEP = MAX_STEP
        self.step_len = step_len
        self.dist_threshold = 10
        self.mask_range = True
        
        # uav parameters
        # unit here is m/s
        self.truck_velocity = truck_velocity
        self.uav_velocity = copy(uav_velocity)
        # unit here is kg
        self.uav_capacity = copy(uav_capacity)

        # unit here is m
        self.uav_range = copy(uav_range)
        
        self.num_truck = num_truck
        self.num_uavs_0 = num_uavs_0
        self.num_uavs_1 = num_uavs_1
        self.num_uavs = num_uavs_0 + num_uavs_1
        
        self.num_parcels = num_parcels
        self.num_customer_truck = num_parcels_truck
        self.num_customer_both = num_parcels - num_parcels_truck
        
        # map parameters
        self.map_size = 10_000 # m as unit here
        self.grid_edge = 250 # m as unit here
        
        self.batch_size = None
        
        self.uav_names = ["uav_0_" + str(i) for i in range(self.num_uavs_0)] + ["uav_1_" + str(i) for i in range(self.num_uavs_1)]
        
        self.uav_name_mapping = dict(zip([agent for agent in self.uav_names], list(range(self.num_uavs))))
        
        self.truck_position = None
        self.uav_positions = None
        self.uav_target_positions = None
        self.truck_target_position = None
        self.truck_path = None
        # 0 : truck, 1-num_uavs : uav
        # mask unavailable as -1
        self.current_carrier = None
        # True means available, False means unavailable
        self.available_carrier = None
        # 1 means available, 0 means not
        # customer_truck -> customer_both
        self.mask = None
        self.mask_uav_0 = None
        self.mask_uav_1 = None

        self.uav_stages = None

        self.warehouse_position = None
        self.customer_position_truck = None
        self.customer_position_both = None
        
        self.last_tr = None
        self.last_dr = None


    def get_uav_info(self, uav):
        # get the uav info: kind and no.
        uav_info = findall(r'\d+', uav)
        uav_info = [int(num) for num in uav_info]
        uav_no = uav_info[0] * self.num_uavs_0 + uav_info[1]
        return uav_info + [uav_no]
        

    def generate_weight(self, lower, partition1, partition2, upper):
        probability = self.RNG.random()  # generate a random number between [0, 1)
        if probability < 0.8:
            # 0.8：<= 3.6kg
            return self.RNG.uniform(lower, partition1)
        elif probability < 0.9:
            # 0.1：3.6kg - 10kg
            return self.RNG.uniform(partition1, partition2)
        else:
            # 0.1：>= 10kg
            return self.RNG.uniform(partition2, upper)


    def reset(self, seed=None, batch_size=1):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - timestamp
        - observation
        - info

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        # re-seed the RNG
        # if seed is None: 
        #     seed = random.randint(0, 1_000)
        if seed is not None:
            self.RNG, _ = seeding.np_random(seed)
        # set the time step to 0 initially
        self.batch_size = batch_size
        self.time_step = np.zeros(self.batch_size) 
                
        # Initially, all UAVs are in state of carried
        self.uav_stages = np.ones((self.batch_size, self.num_uavs)) * -1 
        # carrier status
        self.available_carrier = np.ones((self.batch_size, self.num_uavs + 1)) 
        self.current_carrier = np.argmax(self.available_carrier, axis=1) 
        # customer status
        self.mask = np.ones((self.batch_size, self.num_parcels)) 
        # uav capacity load mask
        self.mask_uav_0 = np.ones((self.batch_size, self.num_customer_both)) 
        self.mask_uav_1 = np.ones((self.batch_size, self.num_customer_both)) 
        
        self.warehouse_position = np.full((self.batch_size, 2), [self.map_size / 2, self.map_size / 2], dtype=np.int32) 
        # All customer points are distributed at the edge of the road grid
        grid_num = self.map_size / self.grid_edge

        # randomly distribute
        self.customer_position_truck = np.array([[
            [self.RNG.integers(0.2 * self.map_size, 0.8 * self.map_size), self.RNG.integers(0.2 * grid_num, 0.8 * grid_num)*self.grid_edge] if i % 2 
            else [self.RNG.integers(0.2 * grid_num, 0.8 * grid_num)*self.grid_edge, self.RNG.integers(0.2 * self.map_size, 0.8 * self.map_size)] 
            for i in range(self.num_customer_truck)
        ] for _ in range(self.batch_size)])
        
        self.customer_position_both = np.array([[
            [self.RNG.integers(0.2 * self.map_size, 0.8 * self.map_size), self.RNG.integers(0.2 * grid_num, 0.8 * grid_num)*self.grid_edge] if i % 2 
            else [self.RNG.integers(0.2 * grid_num, 0.8 * grid_num)*self.grid_edge, self.RNG.integers(0.2 * self.map_size, 0.8 * self.map_size)] 
            for i in range(self.num_customer_both)
        ] for _ in range(self.batch_size)])

        self.truck_path = [[] for _ in range(self.batch_size)] 
        self.truck_target_position = np.ones((self.batch_size, 2)) * -1 
        self.uav_target_positions = np.ones((self.batch_size, self.num_uavs, 2), dtype=np.int32) * (-1) 
        
        self.truck_position = copy(self.warehouse_position) 
        self.uav_positions = np.repeat(self.warehouse_position[:, np.newaxis, :], self.num_uavs, axis=1)
        
        # Ignore capacity constraint temporarily
        # self.parcels_weight = [
        #     # (0, 3.6, 6, 10)
        #     self.generate_weight(0, 1, 2, 3.6) if idx < self.num_customer_both
        #     else self.generate_weight(0, 1, 2, 3.6) # to make sure all the uav parcel can be delivered
        #     for idx in range(self.num_customer_both)
        # ] 
        
        # for i in range(self.num_customer_both):
        #     if self.parcels_weight[i] > self.uav_capacity[0]:
        #         self.mask_uav_0[i] = 0
        #     if self.parcels_weight[i] > self.uav_capacity[1]:
        #         self.mask_uav_1[i] = 0
        
        self.last_tr = np.zeros(self.batch_size) 
        self.last_dr = np.zeros(self.batch_size) 
        
        obs = np.concatenate([self.warehouse_position[:, np.newaxis, :], self.customer_position_truck, self.customer_position_both], axis=1) - self.truck_position[:, np.newaxis, :] 
        # obs = np.concatenate((obs, np.zeros((self.num_parcels - obs.shape[0], 2))))
        observation = { 'obs': obs, 'mask': np.concatenate((np.zeros((self.batch_size, 1)), self.mask), axis=1), 'last': self.last_tr.astype(np.int64) } 
        
        # # Get dummy info. Necessary for proper parallel_to_aec conversion
        
        info = {}
        self.termination = np.zeros(self.batch_size)
        
        return observation, info
    
    
    # When the truck performs a new action, it first generates a refined path through genarate_truck_path(),
    # and then moves in truck_move() according to the generated path before reaching the target 
    # (that is, before generating a new action).
    def genarate_truck_path(self, truck_position, target):
        truck_path = []
        # get the id of the grid which truck and target located here...
        id_grid_truck_x = int(truck_position[0] / self.grid_edge)
        id_grid_target_x = int(target[0] / self.grid_edge)
        id_grid_truck_y = int(truck_position[1] / self.grid_edge)
        id_grid_target_y = int(target[1] / self.grid_edge)
    
        if target[0] == truck_position[0] or target[1] == truck_position[1]:
            # situation 1:
            # ##T##...###C#
            # truck and target at the same route line 
            truck_path.append(target)
        elif id_grid_target_x == id_grid_truck_x and id_grid_target_y != id_grid_truck_y:
            # situation 2:
            #    # # T # #
            #    #       #
            #    #  ...  #
            #    #       #
            #    # # # C #
            if (truck_position[0] % self.grid_edge) + (target[0] % self.grid_edge) <= self.grid_edge:
                truck_path.append(np.array([id_grid_truck_x * self.grid_edge, id_grid_truck_y * self.grid_edge]))
                truck_path.append(np.array([id_grid_target_x * self.grid_edge, id_grid_target_y * self.grid_edge]))
                truck_path.append(target)
            else:
                truck_path.append(np.array([(id_grid_truck_x + 1) * self.grid_edge, id_grid_truck_y * self.grid_edge]))
                truck_path.append(np.array([(id_grid_target_x + 1) * self.grid_edge, id_grid_target_y * self.grid_edge]))
                truck_path.append(target)
        elif id_grid_target_y == id_grid_truck_y and id_grid_target_x != id_grid_truck_x:
            # situation 3:
            #    # # # # #
            #    #   .   C
            #    T   .   #
            #    #   .   #
            #    # # # # #
            if (truck_position[1] % self.grid_edge) + (target[1] % self.grid_edge) <= self.grid_edge:
                truck_path.append(np.array([id_grid_truck_x * self.grid_edge, id_grid_truck_y * self.grid_edge]))
                truck_path.append(np.array([id_grid_target_x * self.grid_edge, id_grid_target_y * self.grid_edge]))
                truck_path.append(target)
            else:
                truck_path.append(np.array([id_grid_truck_x * self.grid_edge, (id_grid_truck_y + 1) * self.grid_edge]))
                truck_path.append(np.array([id_grid_target_x * self.grid_edge, (id_grid_target_y + 1) * self.grid_edge]))
                truck_path.append(target)
        elif truck_position[0] % self.grid_edge == 0:
            if target[1] % self.grid_edge == 0:
                # situation 4:
                #    # # # # # # # C #
                #    #       #       #
                #    T  ...  #       #
                #    #       #       #
                #    # # # # # # # # #
                truck_path.append(np.array([id_grid_truck_x * self.grid_edge, id_grid_target_y * self.grid_edge]))
                truck_path.append(target)
            else:
                # situation 5:
                #    # # T # # # # # #
                #    #       #       #
                #    #  ...  #       #
                #    #       #       #
                #    # # # # # C # # #
                truck_path.append(np.array([id_grid_truck_x * self.grid_edge, id_grid_target_y * self.grid_edge] if truck_position[1] < target[1]
                                                else [id_grid_truck_x * self.grid_edge, (id_grid_target_y + 1) * self.grid_edge]))
                # truck_path.append(np.array([id_grid_target_x * self.grid_edge, truck_path[-1][1]]))
                truck_path.append(np.array([target[0], truck_path[-1][1]]))
                truck_path.append(target)
        # so as situation 4 and 5.
        elif truck_position[1] % self.grid_edge == 0:
            if target[0] % self.grid_edge == 0:
                truck_path.append(np.array([id_grid_target_x * self.grid_edge, id_grid_truck_y * self.grid_edge]))
                truck_path.append(target)
            else:
                truck_path.append(np.array([id_grid_target_x * self.grid_edge, id_grid_truck_y * self.grid_edge] if truck_position[0] < target[0]
                                                else [(id_grid_target_x + 1) * self.grid_edge, id_grid_truck_y * self.grid_edge]))
                truck_path.append(np.array([truck_path[-1][0], target[1]]))
                truck_path.append(target)
        
        return truck_path

    
    def truck_move(self, truck_position, truck_path, available_carrier):
        # target point x, y coordinate
        time_left = self.step_len
        while truck_path:
            if time_left == 0:
                break
            if abs(truck_position[0] + truck_position[1] - truck_path[0][0] - truck_path[0][1]) <= self.truck_velocity * time_left:
                truck_position[0] = truck_path[0][0]
                truck_position[1] = truck_path[0][1]
                time_left -= abs(truck_position[0] + truck_position[1] - truck_path[0][0] - truck_path[0][1]) / float(self.truck_velocity)
                truck_path.pop(0)
            elif truck_position[0] == truck_path[0][0]:
                truck_position[1] += (int(time_left * self.truck_velocity) if truck_position[1] < truck_path[0][1] 
                                           else int(time_left * self.truck_velocity * (-1)))
                time_left = 0
            else:
                truck_position[0] += (int(time_left * self.truck_velocity) if truck_position[0] < truck_path[0][0] 
                                           else int(time_left * self.truck_velocity) * (-1))
                time_left = 0
        if not truck_path:
            available_carrier[0] = 1
            return True
        else:
            return False
    
    
    # launched uavs move for 1 timestep in straight line
    def uav_move(self, uav_stage, truck_position, uav_positions, uav_target_positions, available_carrier):
        # return the type of retrieved uav or None
        uav_retrieved_type = None
        for uav in self.uav_names:
            uav_no = self.uav_name_mapping[uav]
            uav_type = int(findall(r'\d+', uav)[0])
            
            if uav_stage[uav_no] >= 0:
                if uav_stage[uav_no] == 1:
                    uav_target = copy(uav_target_positions[uav_no])
                else:
                    uav_target = copy(truck_position)
                
                vec = uav_target - uav_positions[uav_no]
                dist = np.linalg.norm(vec)
                length = min(dist, self.uav_velocity[uav_type] * self.step_len)
                
                # when the distance between uav and target is less than a threshold
                # then consider the uav as arrival 
                if dist - length < self.dist_threshold:
                    uav_positions[uav_no] = copy(uav_target)
                    uav_stage[uav_no] -= 1
                    
                    if uav_stage[uav_no] == -1:
                        # if (uav_no < self.num_uavs_0 and available_uav_0) or (uav_no >= self.num_uavs_0 and available_uav_1):
                        uav_retrieved_type = uav_type
                        # mark this uav as available
                        available_carrier[uav_no + 1] = 1
                else:
                    normed_vec = vec / dist
                    
                    # angle = np.arctan2(vec[1], vec[0])
                    # angle = angle if angle >= 0 else angle + 2 * np.pi            
                    
                    # action = np.array([angle, v])
                    uav_positions[uav_no][0] += normed_vec[0] * length
                    uav_positions[uav_no][1] += normed_vec[1] * length
                
            else:
                uav_positions[uav_no] = copy(truck_position)
        
        return uav_retrieved_type


    def get_range_mask(self):
        dists = np.linalg.norm(self.customer_position_both - self.truck_position[:, np.newaxis, :], axis=2)
        range_mask = np.zeros((self.batch_size, self.num_customer_both))
        
        for i in range(self.batch_size): 
            if self.current_carrier[i] == 0:
                range_mask[i] = np.ones(self.num_customer_both)
            elif self.current_carrier[i] <= self.num_uavs_0:
                range_mask[i] = (dists[i] <= 0.5 * self.uav_range[0]).astype(int)
            else:
                range_mask[i] = (dists[i] <= 0.5 * self.uav_range[1]).astype(int)
        
        return range_mask


    def get_range_mask_by_uav_type(self, uav_type):
        dists = np.linalg.norm(self.customer_position_both - self.truck_position[:, np.newaxis, :], axis=2)
        
        return (dists <= 0.5 * self.uav_range[uav_type]).astype(int)

    
    def get_range_mask_by_id_type(self, idx, uav_type):
        dist = np.linalg.norm(self.customer_position_both[idx] - self.truck_position[idx], axis=1)
        
        return (dist <= 0.5 * self.uav_range[uav_type]).astype(int)
    
    
    # When all uav_service customer have been delivered, 
    # the uavs shouln't be considered as available carriers.
    def step(self, action):
        """Takes in an action for the agent.
        Assign Task action to current available agent. (similar to TA_scheduling)
        If there is no other available agents, then: 
        Transite to next decision occasion. 
        - truck move 
        - uav move 
        reward need to be carefully designed: time penalty, victory reward, weighted route length? 
        action mask
        """
        reward = np.zeros(self.batch_size)
        observation = { 
            "obs": np.zeros((self.batch_size, self.num_parcels + 1, 2)), 
            "mask": np.zeros((self.batch_size, self.num_parcels + 1)), 
            'last': np.zeros(self.batch_size) 
        }
        
        # TA part
        for i in range(self.batch_size):
            if self.termination[i]:
                continue
            # prepare for the non-batch data
            avai_carrier = self.available_carrier[i]
            cust_pos_truck = self.customer_position_truck[i]
            cust_pos_both = self.customer_position_both[i]
            mask = self.mask[i]
            act = action[i]
            truck_target_pos = self.truck_target_position[i]
            uav_target_pos = self.uav_target_positions[i]
            uav_stage = self.uav_stages[i]
            
            if self.current_carrier[i] != -1: # typically must not None
                if self.current_carrier[i] == 0:
                    if isinstance(act, np.int64) or isinstance(act, np.int32) or isinstance(act, int):
                        avai_carrier[0] = 0
                        self.current_carrier[i] = -1
                        self.last_tr[i] = np.int64(act)
                        
                        act -= 1
                        if act < self.num_customer_truck:
                            cust = cust_pos_truck[act]
                        else:
                            cust = cust_pos_both[act - self.num_customer_truck]
                        truck_target_pos[:] = copy(cust)
                        
                        mask[act] = 0

                    else: 
                        print(f'Wrong Action Form in position {i}: {act}')
                        
                else:
                    uav_no = self.current_carrier[i] - 1
                    if uav_stage[uav_no] == -1:
                        if isinstance(act, np.int64) or isinstance(act, np.int32) or isinstance(act, int): 
                            if act != 0: # empty uav action
                                self.last_dr[i] = np.int64(act)
                                act -= (1 + self.num_customer_truck)
                                uav_stage[uav_no] = 1
                                avai_carrier[self.current_carrier[i]] = 0
                                self.current_carrier[i] = -1
                                
                                cust = cust_pos_both[act]

                                uav_target_pos[uav_no] = copy(cust)
                                mask[act + self.num_customer_truck] = 0

                        else: 
                            print(f'Wrong Action Form in position {i}: {act}')
        
        # calculate uav mask based on the uav payload mask and the customer mask
        # if there is no customer available for uav, then set all uavs' state unavailable
        mask_uav_0 = self.mask_uav_0 * self.mask[:, self.num_customer_truck:]
        mask_uav_1 = self.mask_uav_1 * self.mask[:, self.num_customer_truck:]
        
        if self.mask_range:
            mask_uav_0 = mask_uav_0 * self.get_range_mask_by_uav_type(0)
            mask_uav_1 = mask_uav_1 * self.get_range_mask_by_uav_type(1)
        
        no_available_uav_0 = np.all(mask_uav_0 == 0, axis=1)
        no_available_uav_1 = np.all(mask_uav_1 == 0, axis=1)
        avail_carrier = self.available_carrier.copy()
        avail_carrier[no_available_uav_0, 1: self.num_uavs_0 + 1] = 0
        avail_carrier[no_available_uav_1, self.num_uavs_0 + 1 :] = 0

        has_avai_carrier = avail_carrier.any(axis=1)
        self.current_carrier = np.argmax(avail_carrier, axis=1) 
        self.current_carrier = np.where(has_avai_carrier, self.current_carrier, -1) # mask the env without available carrier as -1
        
        # agent move part (when there is no available agent)
        for i in range(self.batch_size):
            # prepare for the non-batch data
            if self.termination[i]:
                continue
            current_carrier = self.current_carrier[i]
            avai_carrier = self.available_carrier[i]
            
            truck_pos = self.truck_position[i]
            truck_target_pos = self.truck_target_position[i]
            
            uav_stage = self.uav_stages[i]
            uav_pos = self.uav_positions[i]
            uav_target_pos = self.uav_target_positions[i]
                        
            if current_carrier == -1:
                # break when move for more than 1000 step without available carrier appearance
                for _ in range(1_000):
                    completed = False
                    # truck moves
                    # in the first movement, a refined path needs to be generated.
                    if not self.truck_path[i]:
                        self.truck_path[i] = self.genarate_truck_path(truck_pos, truck_target_pos)
                    if self.truck_move(truck_pos, self.truck_path[i], avai_carrier):
                        # truck becomes available to be assigned task
                        completed = True
                    retrieved_uav = self.uav_move(uav_stage, truck_pos, uav_pos, uav_target_pos, avai_carrier)
                    if retrieved_uav is not None:
                        # at least one uav becomes available to be assigned task
                        if retrieved_uav == 0:
                            mask_uav_retrieved = self.mask_uav_0[i] * self.mask[i, self.num_customer_truck:]
                            if self.mask_range:
                                mask_uav_retrieved = mask_uav_retrieved * self.get_range_mask_by_id_type(i, 0)
                        else:
                            mask_uav_retrieved = self.mask_uav_1[i] * self.mask[i, self.num_customer_truck:]
                            if self.mask_range:
                                mask_uav_retrieved = mask_uav_retrieved * self.get_range_mask_by_id_type(i, 1)
                        
                        if np.any(mask_uav_retrieved):
                            completed = True
                    # elif np.any(uav_stage == -1):
                    #     current_mask_uav_0 = self.mask_uav_0[i] * self.mask[i, self.num_customer_truck:]
                    #     current_mask_uav_1 = self.mask_uav_1[i] * self.mask[i, self.num_customer_truck:]
                    #     if self.mask_range:
                    #         current_mask_uav_0 = current_mask_uav_0 * self.get_range_mask_by_id_type(i, 0)
                    #         current_mask_uav_1 = current_mask_uav_1 * self.get_range_mask_by_id_type(i, 1)
                        
                    #     no_available_uav_0 = np.all(current_mask_uav_0 == 0)
                    #     no_available_uav_1 = np.all(current_mask_uav_1 == 0)
                    #     current_avail_carrier = avai_carrier.copy()
                    #     current_avail_carrier[no_available_uav_0, 1: self.num_uavs_0 + 1] = 0
                    #     current_avail_carrier[no_available_uav_1, self.num_uavs_0 + 1 :] = 0

                    #     if current_avail_carrier.any():
                    #         completed = True
                    
                    self.time_step[i] += 1
                    if completed: 
                        break
                
            else:
                for _ in range(2):
                    # truck moves
                    # in the first movement, a refined path needs to be generated.
                    if not self.truck_path[i]:
                        self.truck_path[i] = self.genarate_truck_path(truck_pos, truck_target_pos)
                    self.truck_move(truck_pos, self.truck_path[i], avai_carrier)

                    self.uav_move(uav_stage, truck_pos, uav_pos, uav_target_pos, avai_carrier)
                    
                    self.time_step[i] += 1
        
        # update the next assigned carrier:current_carrier after moving
        mask_uav_0 = self.mask_uav_0 * self.mask[:, self.num_customer_truck:]
        mask_uav_1 = self.mask_uav_1 * self.mask[:, self.num_customer_truck:]
        if self.mask_range:
            mask_uav_0 = mask_uav_0 * self.get_range_mask_by_uav_type(0)
            mask_uav_1 = mask_uav_1 * self.get_range_mask_by_uav_type(1)
        
        no_available_uav_0 = np.all(mask_uav_0 == 0, axis=1)
        no_available_uav_1 = np.all(mask_uav_1 == 0, axis=1)
        
        avail_carrier = self.available_carrier.copy()
        avail_carrier[no_available_uav_0, 1: self.num_uavs_0 + 1] = 0
        avail_carrier[no_available_uav_1, self.num_uavs_0 + 1 :] = 0

        has_avai_carrier = avail_carrier.any(axis=1)
        self.current_carrier = np.argmax(avail_carrier, axis=1) 
        self.current_carrier = np.where(has_avai_carrier, self.current_carrier, -1) # mask the env without available carrier as -1

        info = {}
        
        # termination check part
        for i in range(self.batch_size):
            if self.termination[i]:
                continue
            mask = self.mask[i]
            
            if not np.any(mask):
                current_carrier = self.current_carrier[i]
                avai_carrier = self.available_carrier[i]
                
                truck_target_pos = self.truck_target_position[i]
                truck_pos = self.truck_position[i]
                wh_pos = self.warehouse_position[i]
                
                uav_stage = self.uav_stages[i]
                uav_pos = self.uav_positions[i]
                uav_target_pos = self.uav_target_positions[i]
                
                if avai_carrier[0] == 0:
                    arrived = False
                    while not arrived:
                        arrived = self.truck_move(truck_pos, self.truck_path[i], avai_carrier)
                        self.uav_move(uav_stage, truck_pos, uav_pos, uav_target_pos, avai_carrier)
                        self.time_step[i] += 1
                
                truck_target_pos = copy(wh_pos)
                self.truck_path[i] = self.genarate_truck_path(truck_pos, truck_target_pos)
                avai_carrier[0] = 0
                current_carrier = -1
                arrived = False
                while not arrived:
                    arrived = self.truck_move(truck_pos, self.truck_path[i], avai_carrier)
                    self.uav_move(uav_stage, truck_pos, uav_pos, uav_target_pos, avai_carrier)
                    self.time_step[i] += 1
                
                self.termination[i] = 1
                reward[i] += self.time_step[i] * REWARD_URGENCY

        # Check truncation conditions (overwrites termination conditions)
        truncation = [ self.time_step[idx] > self.MAX_STEP for idx in range(self.batch_size) ]

        # Get observation
        ####
        obs = np.concatenate([self.warehouse_position[:, np.newaxis, :], self.customer_position_truck, self.customer_position_both], axis=1) - self.truck_position[:, np.newaxis, :]
        
        action_mask = np.zeros((self.batch_size, self.num_parcels + 1))
        for i in range(self.batch_size): 
            if self.current_carrier[i] == 0:
                action_mask[i] = np.concatenate((np.zeros((1)), self.mask[i]))
            elif self.current_carrier[i] <= self.num_uavs_0:
                action_mask[i] = np.concatenate((np.zeros((1 + self.num_customer_truck)), mask_uav_0[i]))
            else:
                action_mask[i] = np.concatenate((np.zeros((1 + self.num_customer_truck)), mask_uav_1[i]))
        
        last = np.where((self.current_carrier == 0) & (self.uav_stages[:, 0] != -1), self.last_dr, self.last_tr)
        
        # if self.mask_range:
        #     range_mask = self.get_range_mask()
        #     action_mask[:, -self.num_customer_both:] = action_mask[:, -self.num_customer_both:] * range_mask

        # No big difference from the observation at reset()
        observation = { 'obs': obs, 'mask': action_mask, 'last': last.astype(np.int64) }
        
        return observation, reward, self.termination, truncation, info
    
    
    def time_steps(self):
        return np.where(self.termination, self.time_step, PENALTY_TRUNCATION)
    
    
    def get_battery_dynamic(self):
        return np.tile(np.where(self.current_carrier == 0, float(100_000), self.uav_range[1]*0.5)[:, np.newaxis, np.newaxis], (1, self.num_parcels + 1, 1))


    def render(self):
        """Renders the environment."""
        # currently, render used mainly in testing rather than visualization :)
        if self.render_mode == None:
            return
        
        screen_width = 1000
        screen_height = 1000
        grid_width = screen_width / 40
        grid_height = screen_height / 40
        scale = self.map_size / screen_width
        
        # As an offset to the coordinates of objects in the scene
        position_bias = np.array([grid_width, grid_height])
        
        if self.screen == None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Truck & UAVs")
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((screen_width, screen_height))
        
        # Load the texture used to render the scene
        map_image = get_image(os.path.join("img", "Map5.png"))
        map_image = pygame.transform.scale(map_image, (screen_width, screen_height))
        truck_image = get_image(os.path.join("img", "Truck.png"))
        truck_image = pygame.transform.scale(truck_image, (grid_width * 0.8, grid_height * 0.8))
        uav_image = get_image(os.path.join("img", "UAV.png"))
        uav_image = pygame.transform.scale(uav_image, (grid_width * 0.6, grid_height * 0.6))
        customer_truck_image = get_image(os.path.join("img", "CustomerTruck.png"))
        customer_truck_image = pygame.transform.scale(customer_truck_image, (grid_width * 0.8, grid_height * 0.8))
        customer_uav_image = get_image(os.path.join("img", "CustomerUAV.png"))
        customer_uav_image = pygame.transform.scale(customer_uav_image, (grid_width * 0.8, grid_height * 0.8))
        
        self.screen.blit(map_image, (0, 0))
        
        # There is a scale between simulation coordinates and rendering coordinates, here it is 10:1
        for customer in self.customer_position_truck:
            self.screen.blit(customer_truck_image, customer / scale - position_bias * 0.4)
        for customer in self.customer_position_both:
            self.screen.blit(customer_uav_image, customer / scale - position_bias * 0.4)
        
        self.screen.blit(truck_image, self.truck_position / scale - position_bias * 0.4)
        for uav in self.uav_positions:
            self.screen.blit(uav_image, uav / scale - position_bias * 0.3)
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(10)
            
        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )
        
                
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        return


def get_image(path):
    from os import path as os_path
    import pygame

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def env_creator(env_config={}):
    env = UpperSolverTrainingEnvironment(
        MAX_STEP=env_config.get("MAX_STEP", 2_000), 
        step_len=env_config.get("step_len", 5), 
        truck_velocity=env_config.get("truck_velocity", 6), 
        uav_velocity=env_config.get("uav_velocity", np.array([12, 29])), 
        uav_capacity=env_config.get("uav_capacity", np.array([10, 3.6])), 
        uav_range=env_config.get("uav_range", np.array([10_000, 15_000]) * 0.6), 
        num_truck=env_config.get("num_truck", 1), 
        num_uavs_0=env_config.get("num_uavs_0", 0), 
        num_uavs_1=env_config.get("num_uavs_1", 1), 
        num_parcels=env_config.get("num_parcels", 20), 
        num_parcels_truck=env_config.get("num_parcels_truck", 4), 
        render_mode=env_config.get("render_mode", None), 
    )
    
    return env


if __name__ == '__main__':
    env = env_creator()
    batch_size = 128
    random_seed = random.randint(0, 100_000)
    print("# Set random seed to %d" % random_seed)
    observation, _ = env.reset(seed=random_seed, batch_size=batch_size)
    np.random.seed(random_seed)

    mask = observation['mask']
    
    action = np.full(batch_size, -1) 
    
    for idx in range(30):
        for i in range(batch_size):
            nonzero_indices = np.flatnonzero(mask[i]) 
            if nonzero_indices.size > 0:
                action[i] = np.random.choice(nonzero_indices) 
        observation, reward, termination, truncation, _ = env.step(action)
        mask = observation['mask']
        # print(f"ts in {idx}: {env.time_step}")
    print(np.mean(env.time_steps()))
        # print(termination)