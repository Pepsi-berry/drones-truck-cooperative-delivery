import os
import functools
import random
from copy import copy
from re import match, findall
# from itertools import combinations

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, Sequence
from gymnasium.utils import seeding

from gymnasium import Env
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
REWARD_URGENCY = -0.2
REWARD_APPROUCHING = 0.02 # get REWARD_APPROUCHING when get closer to target
REWARD_UAVS_DANGER = float(-400) # coefficient of the penalty for being to close with other uavs
# REWARD_SLOW = -0.02

TYPE_TRUCK = 0
TYPE_UAV = 1

# it seems that wrapping truck and uav into classes would make programming significantly less difficult...
# but when I realize this, it had gone much much too far...

# In once decision, there may be more than one uav selected the same customer as the target point
# special treatment of this needs to be refined(maybe AEV is more suitable? :(

class UpperSolverTrainingEnvironment(Env):
    """The metadata holds environment constants.
    
    The "name" metadata allows the environment to be pretty printed.
    
    """

    metadata = {
        "render_modes": [None, "text", "human", "rgb_array"],
        "name": "upper_training_environment_v1",
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
        
        self.console_infos = None
        
        self.MAX_STEP = MAX_STEP
        self.step_len = step_len
        self.time_step = 0
        self.dist_threshold = 10
        self.generative_range = 1000 #  + int(self.RNG.integers(1, 4) / 4) * 1000
        
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
        self.map_size = 1_0000 # m as unit here
        self.grid_edge = 250 # m as unit here
        
        self.uav_names = ["uav_0_" + str(i) for i in range(self.num_uavs_0)] + ["uav_1_" + str(i) for i in range(self.num_uavs_1)]
        
        self.uav_name_mapping = dict(zip([agent for agent in self.uav_names], list(range(self.num_uavs))))
        
        self.truck_position = None
        self.uav_positions = None
        self.uav_target_positions = None
        self.truck_target_position = None
        self.truck_path = None
        # 0 means truck, 1-num_uavs means uav
        self.current_carrier = None
        # True means available, False means unavailable
        self.available_carrier = None

        self.uav_stages = None

        self.warehouse_position = None
        self.customer_position_truck = None
        self.customer_position_both = None
        
        ###########################################################################################
        # 1*pos_moving_target(if current agent_type is uav) + N*pos_cust
        # action mask or sequence? 
        self.observation_space = Dict(
            {
                'agent_type': Discrete(2), 
                'obs': Sequence(Box(low=0, high=self.map_size, shape=(2, ), dtype=np.float32))
            }
        )
        # the action space means task assignment of the current available carrier
        # uav could stay still
        self.action_space = Discrete(self.num_parcels)


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


    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        # re-seed the RNG
        if seed is not None:
            self.RNG, _ = seeding.np_random(seed)
        # set the time step to 0 initially
        self.time_step = 0
        
        self.console_infos = []
        
        # Initially, all UAVs are in state of delivery
        self.uav_stages = np.ones(self.num_uavs) * -1
        
        self.available_carrier = np.ones(self.num_uavs + 1)
        available_indices = np.flatnonzero(self.available_carrier)
        self.current_carrier = available_indices[0] if available_indices.size > 0 else None
        
        self.warehouse_position = np.array([self.map_size / 2, self.map_size / 2], dtype=np.int32)
        # All customer points are distributed at the edge of the road grid
        grid_num = self.map_size / self.grid_edge

        self.customer_position_truck = [[self.RNG.integers(0.4 * self.map_size, 0.6 * self.map_size), self.RNG.integers(0.4 * grid_num, 0.6 * grid_num)*self.grid_edge] if i % 2 
                                        else [self.RNG.integers(0.4 * grid_num, 0.6 * grid_num)*self.grid_edge, self.RNG.integers(0.4 * self.map_size, 0.6 * self.map_size)] 
                                        for i in range(self.num_customer_truck)]
            
        self.customer_position_both = [[self.RNG.integers(0.4 * self.map_size, 0.6 * self.map_size), self.RNG.integers(0.4 * grid_num, 0.6 * grid_num)*self.grid_edge] if i % 2 
                                       else [self.RNG.integers(0.4 * grid_num, 0.6 * grid_num)*self.grid_edge, self.RNG.integers(0.4 * self.map_size, 0.6 * self.map_size)] 
                                       for i in range(self.num_customer_both)]

        self.truck_path = []
        self.truck_target_position = np.ones(2) * -1
        self.uav_target_positions = np.ones([self.num_uavs, 2], dtype=np.int32) * (-1)
        
        self.truck_position = copy(self.warehouse_position)
        self.uav_positions = np.array([copy(self.truck_position) for _ in range(self.num_uavs)])
                
        self.parcels_weight = [
            # (0, 3.6, 6, 10)
            self.generate_weight(0, 1, 2, 3.6) if idx < self.num_customer_both
            else self.generate_weight(0, 1, 2, 3.6) # to make sure all the uav parcel can be delivered
            for idx in range(self.num_customer_both)
        ]
        
        self.customer_position_uav_0 = [ 
            self.customer_position_both[i] 
            for i in range(len(self.customer_position_both)) 
            if self.parcels_weight[i] <= self.uav_capacity[0] 
        ]
        self.customer_position_uav_1 = [ 
            self.customer_position_both[i] 
            for i in range(len(self.customer_position_both)) 
            if self.parcels_weight[i] <= self.uav_capacity[1] 
        ]
        
        observation = {
            "agent_type": TYPE_TRUCK, 
            "obs": np.array(self.customer_position_truck + self.customer_position_both) - self.truck_position
        }
        
        # # Get dummy infos. Necessary for proper parallel_to_aec conversion
        
        # infos contain if agent is available for TA
        # if there is at least one agent(or truck?) available, then run the upper solver
        infos = {
            "training_enabled": True
        }
        
        return observation, infos
    
    
    # When the truck performs a new action, it first generates a refined path through genarate_truck_path(),
    # and then moves in truck_move() according to the generated path before reaching the target 
    # (that is, before generating a new action).
    def genarate_truck_path(self, target):
        # get the id of the grid which truck and target located here...
        id_grid_truck_x = int(self.truck_position[0] / self.grid_edge)
        id_grid_target_x = int(target[0] / self.grid_edge)
        id_grid_truck_y = int(self.truck_position[1] / self.grid_edge)
        id_grid_target_y = int(target[1] / self.grid_edge)
    
        if target[0] == self.truck_position[0] or target[1] == self.truck_position[1]:
            # situation 1:
            # ##T##...###C#
            # truck and target at the same route line 
            self.truck_path.append(target)
        elif id_grid_target_x == id_grid_truck_x and id_grid_target_y != id_grid_truck_y:
            # situation 2:
            #    # # T # #
            #    #       #
            #    #  ...  #
            #    #       #
            #    # # # C #
            if (self.truck_position[0] % self.grid_edge) + (target[0] % self.grid_edge) <= self.grid_edge:
                self.truck_path.append(np.array([id_grid_truck_x * self.grid_edge, id_grid_truck_y * self.grid_edge]))
                self.truck_path.append(np.array([id_grid_target_x * self.grid_edge, id_grid_target_y * self.grid_edge]))
                self.truck_path.append(target)
            else:
                self.truck_path.append(np.array([(id_grid_truck_x + 1) * self.grid_edge, id_grid_truck_y * self.grid_edge]))
                self.truck_path.append(np.array([(id_grid_target_x + 1) * self.grid_edge, id_grid_target_y * self.grid_edge]))
                self.truck_path.append(target)
        elif id_grid_target_y == id_grid_truck_y and id_grid_target_x != id_grid_truck_x:
            # situation 3:
            #    # # # # #
            #    #   .   C
            #    T   .   #
            #    #   .   #
            #    # # # # #
            if (self.truck_position[1] % self.grid_edge) + (target[1] % self.grid_edge) <= self.grid_edge:
                self.truck_path.append(np.array([id_grid_truck_x * self.grid_edge, id_grid_truck_y * self.grid_edge]))
                self.truck_path.append(np.array([id_grid_target_x * self.grid_edge, id_grid_target_y * self.grid_edge]))
                self.truck_path.append(target)
            else:
                self.truck_path.append(np.array([id_grid_truck_x * self.grid_edge, (id_grid_truck_y + 1) * self.grid_edge]))
                self.truck_path.append(np.array([id_grid_target_x * self.grid_edge, (id_grid_target_y + 1) * self.grid_edge]))
                self.truck_path.append(target)
        elif self.truck_position[0] % self.grid_edge == 0:
            if target[1] % self.grid_edge == 0:
                # situation 4:
                #    # # # # # # # C #
                #    #       #       #
                #    T  ...  #       #
                #    #       #       #
                #    # # # # # # # # #
                self.truck_path.append(np.array([id_grid_truck_x * self.grid_edge, id_grid_target_y * self.grid_edge]))
                self.truck_path.append(target)
            else:
                # situation 5:
                #    # # T # # # # # #
                #    #       #       #
                #    #  ...  #       #
                #    #       #       #
                #    # # # # # C # # #
                self.truck_path.append(np.array([id_grid_truck_x * self.grid_edge, id_grid_target_y * self.grid_edge] if self.truck_position[1] < target[1]
                                                else [id_grid_truck_x * self.grid_edge, (id_grid_target_y + 1) * self.grid_edge]))
                # self.truck_path.append(np.array([id_grid_target_x * self.grid_edge, self.truck_path[-1][1]]))
                self.truck_path.append(np.array([target[0], self.truck_path[-1][1]]))
                self.truck_path.append(target)
        # so as situation 4 and 5.
        elif self.truck_position[1] % self.grid_edge == 0:
            if target[0] % self.grid_edge == 0:
                self.truck_path.append(np.array([id_grid_target_x * self.grid_edge, id_grid_truck_y * self.grid_edge]))
                self.truck_path.append(target)
            else:
                self.truck_path.append(np.array([id_grid_target_x * self.grid_edge, id_grid_truck_y * self.grid_edge] if self.truck_position[0] < target[0]
                                                else [(id_grid_target_x + 1) * self.grid_edge, id_grid_truck_y * self.grid_edge]))
                self.truck_path.append(np.array([self.truck_path[-1][0], target[1]]))
                self.truck_path.append(target)

    
    def truck_move(self):
        # target point x, y coordinate
        time_left = self.step_len
        while self.truck_path:
            if time_left == 0:
                break
            if abs(self.truck_position[0] + self.truck_position[1] - self.truck_path[0][0] - self.truck_path[0][1]) <= self.truck_velocity * time_left:
                self.truck_position[0] = self.truck_path[0][0]
                self.truck_position[1] = self.truck_path[0][1]
                time_left -= abs(self.truck_position[0] + self.truck_position[1] - self.truck_path[0][0] - self.truck_path[0][1]) / float(self.truck_velocity)
                self.truck_path.pop(0)
            elif self.truck_position[0] == self.truck_path[0][0]:
                self.truck_position[1] += (int(time_left * self.truck_velocity) if self.truck_position[1] < self.truck_path[0][1] 
                                           else int(time_left * self.truck_velocity * (-1)))
                time_left = 0
            else:
                self.truck_position[0] += (int(time_left * self.truck_velocity) if self.truck_position[0] < self.truck_path[0][0] 
                                           else int(time_left * self.truck_velocity) * (-1))
                time_left = 0
        if not self.truck_path:
            self.available_carrier[0] = 1
            self.console_infos.append(f"Truck arrived to {self.truck_target_position} at {self.time_step} step. ")
            return True
        else:
            return False
    
    
    # launched uavs move for 1 timestep in straight line
    def uav_move(self):
        # return uav_retrieved to indicate whether an available carrier appears, 
        # that is, whether this transition can be stopped.
        uav_retrieved = False
        for uav in self.uav_names:
            uav_no = self.uav_name_mapping[uav]
            uav_type = int(findall(r'\d+', uav)[0])
            
            if self.uav_stages[uav_no] >= 0:
                if self.uav_stages[uav_no] == 1:
                    uav_target = copy(self.uav_target_positions[uav_no])
                else:
                    uav_target = copy(self.truck_position)
                
                vec = uav_target - self.uav_positions[uav_no]
                dist = np.linalg.norm(vec)
                normed_vec = vec / dist
                
                # angle = np.arctan2(vec[1], vec[0])
                # angle = angle if angle >= 0 else angle + 2 * np.pi            
                
                length = min(dist, self.uav_velocity[uav_type] * self.step_len)
                
                # action = np.array([angle, v])
                self.uav_positions[uav_no][0] += normed_vec[0] * length
                self.uav_positions[uav_no][1] += normed_vec[1] * length
                
                # when the distance between uav and target is less than a threshold
                # then consider the uav as arrival 
                if np.sqrt(np.sum(np.square(self.uav_positions[uav_no] - uav_target))) < self.dist_threshold:
                    self.uav_positions[uav_no] = copy(uav_target)
                    self.console_infos.append(f"uav {uav_no} arrived to {uav_target} at {self.time_step} step. ")
                    self.uav_stages[uav_no] -= 1
                    
                    if self.uav_stages[uav_no] == -1:
                        # if there is at least one customer for this uav, 
                        # then considered this uav as an available carrier
                        if (uav_no < self.num_uavs_0 and self.customer_position_uav_0) or (uav_no >= self.num_uavs_0 and self.customer_position_uav_1):
                            uav_retrieved = True
                            # mark this uav as available
                            self.console_infos.append(f"uav {uav_no} arrived to truck at {self.time_step} step. ")
                            self.available_carrier[uav_no + 1] = 1
            else:
                self.uav_positions[uav_no] = copy(self.truck_position)
        
        return uav_retrieved


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
        """
        time_before = copy(self.time_step)
        reward = REWARD_URGENCY * self.step_len
        # get penalty every transitions to encourage faster delivery
        
        
        if self.current_carrier is not None: # typically must not None
            if self.current_carrier == 0:
                if isinstance(action, np.int64) or isinstance(action, int):
                    self.available_carrier[0] = 0
                    self.current_carrier = None
                    if action < self.num_customer_truck:
                        popped_cust = self.customer_position_truck.pop(action)
                    else: 
                        popped_cust = self.customer_position_both.pop(action - self.num_customer_truck)
                    
                    self.truck_target_position = copy(popped_cust)
                    self.console_infos.append(f"Assign Task {popped_cust} to truck at {self.time_step} step. ")
                    
                    if popped_cust in self.customer_position_uav_0: 
                        self.customer_position_uav_0.remove(popped_cust)
                    if popped_cust in self.customer_position_uav_1: 
                        self.customer_position_uav_1.remove(popped_cust)
                else: 
                    print('Wrong Action Form!')
                    
            else:
                uav_no = self.current_carrier - 1
                if self.uav_stages[uav_no] == -1:
                    if isinstance(action, np.int64) or isinstance(action, int): 
                        if action != 0: # empty uav action
                            action -= 1
                            self.uav_stages[uav_no] = 1
                            self.available_carrier[self.current_carrier] = 0
                            self.current_carrier = None
                            
                            if uav_no < self.num_uavs_0:
                                popped_cust = self.customer_position_uav_0.pop(action)
                                if popped_cust in self.customer_position_uav_1: 
                                    self.customer_position_uav_1.remove(popped_cust)
                            else:
                                popped_cust = self.customer_position_uav_1.pop(action)
                                if popped_cust in self.customer_position_uav_0: 
                                    self.customer_position_uav_0.remove(popped_cust)
                            self.customer_position_both.remove(popped_cust)
                            
                            self.uav_target_positions[uav_no] = copy(popped_cust)
                            self.console_infos.append(f"Assign Task {popped_cust} to uav {uav_no} at {self.time_step} step. ")
                        else:
                            self.console_infos.append(f"Hold off launching uav {uav_no} at {self.time_step} step. ")

                    else: 
                        print('Wrong Action Form!')
        
        available_indices = np.flatnonzero(self.available_carrier)
        self.current_carrier = available_indices[0] if available_indices.size > 0 else None # typically couldn't <= 0
        
        if self.current_carrier is None:
            # break when move for more than 1000 step without available carrier appearance
            for _ in range(1_000):
                completed = False
                # truck moves
                # in the first movement, a refined path needs to be generated.
                if not self.truck_path:
                    self.genarate_truck_path(self.truck_target_position)
                if self.truck_move():
                    # truck becomes available to be assigned task
                    completed = True
                if self.uav_move():
                    # at least one uav becomes available to be assigned task
                    # if self.customer_position_both: 
                    completed = True
                
                self.time_step += 1
                if completed: 
                    break
            
            available_indices = np.flatnonzero(self.available_carrier)
            self.current_carrier = available_indices[0] if available_indices.size > 0 else None # typically couldn't <= 0
        
        truncation = False
        termination = False
        infos = {
            "training_enabled": True
        }
        
        # Check termination conditions
        ####
        if not self.customer_position_both and not self.customer_position_truck:
            self.truck_target_position = copy(self.warehouse_position)
            self.genarate_truck_path(self.truck_target_position)
            self.available_carrier[0] = 0
            self.current_carrier = None
            arrived = False
            while not arrived:
                arrived = self.truck_move()
                self.uav_move()
                self.time_step += 1
            
            self.console_infos.append(f"truck return to warehouse at {self.time_step} step.")
            termination = True
        
        if np.sum(self.uav_stages) == (-1) * self.num_uavs and np.array_equal(self.truck_position, self.warehouse_position) and not self.customer_position_both and not self.customer_position_truck:
            termination = True

        # Check truncation conditions (overwrites termination conditions)
        ####
        if self.time_step >= self.MAX_STEP:
            truncation = True

        # Get observations
        ####
        if not self.customer_position_both and not self.customer_position_truck:
            observation = {
                "agent_type": TYPE_TRUCK, 
                "obs": np.zeros((1, 2))
            }
        else:
            if self.current_carrier == 0:
                obs = np.array(self.customer_position_truck + self.customer_position_both) - self.truck_position
            elif self.current_carrier <= self.num_uavs_0:
                obs = np.concatenate(([self.truck_target_position], self.customer_position_uav_0)) - self.truck_position
            else:
                obs = np.concatenate(([self.truck_target_position], self.customer_position_uav_1)) - self.truck_position    
            
            # No big difference from the observations at reset()
            observation = {
                "agent_type": TYPE_TRUCK if self.current_carrier == 0 else TYPE_UAV, 
                "obs": obs
            }
        
        return observation, reward, termination, truncation, infos

    def render(self):
        """Renders the environment."""
        # currently, render used mainly in testing rather than visualization :)
        if self.render_mode == None:
            return
        if self.render_mode == "text":
            for info in self.console_infos:
                print(info)
            self.console_infos.clear()
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
        return super().close()


def get_image(path):
    from os import path as os_path
    import pygame

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc

def cross_product_2d_array(v1, v2):
    return v1[0] * v2[1] - v2[0] * v1[1]


def generate_random_upper_left_corner(p1, p2):
    xlo, xhi, ylo, yhi = (
        min(p1[0], p2[0]), 
        max(p1[0], p2[0]), 
        min(p1[1], p2[1]), 
        max(p1[1], p2[1])
    )
    
    offset_x = xhi - xlo
    offset_y = yhi - ylo
    
    return [int(xlo + (random.randrange(5, 45) / 100) * offset_x), 
            int(ylo + (random.randrange(5, 45) / 100) * offset_y)]