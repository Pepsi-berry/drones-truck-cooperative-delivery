import os
import functools
import random
from copy import copy
from re import match, findall

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from pettingzoo import ParallelEnv
import pygame

MAX_INT = 2**20
INVALID_ANGLE = 10
# When the distance between the returning uav and the truck is less than this threshold, 
# the return is considered complete.
DIST_THRESHOLD = 100
# rewards in various situations
REWARD_DELIVERY = 20
REWARD_VICTORY = 100
REWARD_UAV_WRECK = -200
REWARD_UAV_RETURNING = 1

# it seems that wrapping truck and uav into classes would make programming significantly less difficult...
# but when I realize this, it had gone much much too far...

# In once decision, there may be more than one uav selected the same customer as the target point
# special treatment of this needs to be refined(maybe AEV is more suitable? :(

# The unimplemented parts in the scenario modeling are listed here:
# *the range model of uavs is to be continue...*
class DeliveryEnvironment(ParallelEnv):
    """The metadata holds environment constants.
    
    The "name" metadata allows the environment to be pretty printed.
    
    """

    metadata = {
        "render_mode": [None, "human"],
        "name": "delivery_environment_v1",
    }

    def __init__(self, render_mode=None):
        """The init method takes in environment arguments.


        These attributes should not be changed after initialization.
        """
        ##########################################################################################
        # These attributes should not be changed after initialization except time_step.
        self.render_mode = render_mode
        self.screen = None
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # self.MAX_STEP = 1_000_000
        self.MAX_STEP = 100_000
        self.step_len = 10
        self.time_step = None
        
        self.possible_agents = ["truck", 
                                "carried_uav_0_0", "carried_uav_0_1", 
                                "carried_uav_1_0", "carried_uav_1_1", 
                                "carried_uav_1_2", "carried_uav_1_3", 
                                "returning_uav_0_0", "returning_uav_0_1", 
                                "returning_uav_1_0", "returning_uav_1_1", 
                                "returning_uav_1_2", "returning_uav_1_3", 
                                ]
        # uav parameters
        # unit here is m/s
        self.truck_velocity = 7
        self.uav_velocity = np.array([12, 29])
        # unit here is kg
        self.uav_capacity = np.array([10, 3.6])
        # unit here is m
        # self.uav_range = 15_000, 20_000
        self.uav_range = np.array([10_000, 15_000])
        
        self.num_truck = 1
        self.num_uavs = 6
        self.num_uavs_0 = 2
        self.num_uavs_1 = 4
        # self.max_num_agents = self.num_truck + self.num_uavs # wrong operation
        
        # parcels parameters
        self.num_parcels = 20
        self.num_parcels_truck = 4
        self.num_parcels_uav = 6
        self.num_customer_truck = self.num_parcels - self.num_parcels_uav
        self.num_customer_uav = self.num_parcels - self.num_parcels_truck
        self.num_customer_both = self.num_parcels - self.num_parcels_truck - self.num_parcels_uav
        self.weight_probabilities = [0.8, 0.1, 0.1]
        
        # map parameters
        self.map_size = 10_000 # m as unit here
        self.grid_edge = 250 # m as unit here
        
        # The action space of the truck is, choosing a meaningful target point to go to
        # that is warehouse point or customer points which truck can delivery:
        # action_space_truck = Discrete(self.num_customer_truck + 1)
        # The action space of the carried uav is similar to truck
        # action_space_uav_carried[num_customer_uav + 1] is set to empty action:
        # action_space_uav_carried = Discrete(self.num_customer_uav + 1)
        # the action space of the returning uav is chasing the truck in any direction:
        # action_space_uav_returning = Box(low=0, high=2*np.pi, shape=(1, ))
        self.action_spaces = {
            agent: (
                Discrete(self.num_customer_truck + 1) if match("truck", agent) 
                else Discrete(self.num_customer_uav + 1) if match("carried", agent)
                else Box(low=0, high=2*np.pi, shape=(1, ))
            ) 
            for agent in self.possible_agents
        }
        
        ##########################################################################################
        # agent positions, warehouse positions, customer positions, 
        # action masks to prevent invalid actions to be taken
        self.agents = None
        self.truck_position = None
        self.uav_position = None
        self.uav_battery_remaining = None
        # variables used to help representing the movements of the agent in step()
        self.uav_dist = None
        self.uav_target_dist = None
        self.uav_target_angle_sin = None
        self.uav_target_angle_cos = None
        self.truck_path = None

        # The following three *_mask will be assigned as slices of action_mask. 
        # The slicing operation will return the view of the original data, 
        # so that the modification of *_mask will have an impact on the original data.
        # self.customer_both_masks = None
        # self.customer_truck_masks = None
        # self.customer_uav_masks = None
        self.truck_masks = None
        self.uav_masks = None
        
        # infos contains information about the state of the agents.
        self.infos = None
        
        
        ##########################################################################################
        # parameters below The following variables will be assigned in reset(), 
        # and not allowed to be changed afterwards.
        # through invalid action masks to prevent the agent from going to the customer point,
        # where the delivery has been completed
        self.action_masks = None
        # Use *_load_mask and uav_mask to multiply bitwise to get the final uav action mask
        self.uav0_load_masks = None
        self.uav1_load_masks = None
        
        self.warehouse_position = None
        self.customer_position_truck = None
        self.customer_position_uav = None
        self.customer_position_both = None
        
        # parcel weights probability distribution: 
        # <= 3.6kg: 0.8, 3.6kg - 10kg: 0.1, > 10kg: 0.1
        self.parcels_weight = None
        
        # 15_001 here is the size of city map by m, should be parameterized later...
        # 1(warehouse) + 1(truck itself)
        # 1(warehouse) + 1(truck) + 1(uav itself)
        self.observation_spaces = {
            agent: (
                MultiDiscrete(np.full([(1 + 1 + self.num_uavs + self.num_customer_truck), 2], self.map_size + 1)) if match("truck", agent) 
                else MultiDiscrete(np.full([(1 + 1 + 1 + self.num_customer_uav), 2], self.map_size + 1))
            ) 
            for agent in self.possible_agents
        }
    
    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self.observation_spaces[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def generate_weight(self):
        probability = random.random()  # generate a random number between [0, 1)
        if probability < 0.8:
            # 0.8：<= 3.6kg
            return random.uniform(0, 3.6)
        elif probability < 0.9:
            # 0.1：3.6kg - 10kg
            return random.uniform(3.6, 10)
        else:
            # 0.1：>= 10kg
            return random.uniform(10.1, 50)

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - prisoner x and y coordinates
        - guard x and y coordinates
        - escape x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        # set the time step to 0 initially
        self.time_step = 0
        # Initially, all UAVs are in state of loaded in truck
        self.agents = ["truck", 
                       "carried_uav_0_0", "carried_uav_0_1", 
                       "carried_uav_1_0", "carried_uav_1_1", 
                       "carried_uav_1_2", "carried_uav_1_3", 
                       ]
        # The warehouse is located in the center of the map
        self.warehouse_position = np.array([self.map_size / 2, self.map_size / 2])
        # All customer points are distributed at the edge of the road grid
        self.customer_position_truck = np.array(
            [[random.randint(0, self.map_size), random.randint(0, 40)*self.grid_edge] if i % 2 
             else [random.randint(0, 40)*self.grid_edge, random.randint(0, self.map_size)] 
             for i in range(self.num_parcels_truck)]
            )
        self.customer_position_uav = np.array(
            [[random.randint(0, self.map_size), random.randint(0, 40)*self.grid_edge] if i % 2 
             else [random.randint(0, 40)*self.grid_edge, random.randint(0, self.map_size)] 
             for i in range(self.num_parcels_uav)]
            )
        self.customer_position_both = np.array(
            [[random.randint(0, self.map_size), random.randint(0, 40)*self.grid_edge] if i % 2 
             else [random.randint(0, 40)*self.grid_edge, random.randint(0, self.map_size)] 
             for i in range(self.num_customer_both)]
            )
        
        # Initially, the target points of all agents are not determined
        # So the uav_target_dist is set to inf and the truck path is set to empty
        self.uav_dist = np.full(self.num_uavs, -1)
        self.uav_target_dist = np.full(self.num_uavs, MAX_INT)
        self.uav_target_angle_cos = np.full(self.num_uavs, INVALID_ANGLE, dtype=float)
        self.uav_target_angle_sin = np.full(self.num_uavs, INVALID_ANGLE, dtype=float)
        self.truck_path = []
        
        # parcel weights probability distribution: 
        # <= 3.6kg: 0.8, 3.6kg - 10kg: 0.1, > 10kg: 0.1
        # Distribution of parcels space in parcels_weight: 
        # to customer_both —— to customer_uav
        #      10 here     ——      6 here    
        self.parcels_weight = np.array([self.generate_weight() for _ in range(self.num_customer_uav)])
        
        # Initially, the truck departs from the warehouse
        # uavs travel with trucks
        # The number of trucks is temporarily set to 1. 
        # If it is expanded to multiple trucks later
        # all the variable related to the trucks including here needs to be expanded.
        self.truck_position = copy(self.warehouse_position)
        self.uav_position = np.array([copy(self.truck_position) for _ in range(self.num_uavs)])
        # Set the initial power of the uav to the full charge
        self.uav_battery_remaining = np.zeros(self.num_uavs)
        for agent in self.agents:
            if not match("truck", agent):
                uav_info = findall(r'\d+', agent)
                uav_info = [int(num) for num in uav_info]
                uav_no = uav_info[0] * self.num_uavs_0 + uav_info[1]
                self.uav_battery_remaining[uav_no] = self.uav_range[uav_info[0]]
        
        # self.uav_position = {
        #     uav: copy(self.truck_position) 
        #     for uav in self.possible_agents
        #     if not match("truck", uav)
        #     }
        
        # Distribution of action space in action_masks: 
        # to warehouse —— to customer_truck —— to customer_both —— to customer_uav —— uav do nothing
        #       1      ——      4 here       ——      10 here     ——      6 here     ——        1
        # thats for convenience of slicing
        self.action_masks = np.ones(1 + self.num_parcels + 1)
        self.truck_masks = self.action_masks[: 1 + self.num_customer_truck]
        self.uav_masks = self.action_masks[1 + self.num_parcels_truck : 1 + self.num_parcels + 1]
        self.uav0_load_masks = np.ones(1 + self.num_customer_uav)
        self.uav1_load_masks = np.ones(1 + self.num_customer_uav)
        # uav_0_masks = copy(self.uav_masks)
        # uav_1_masks = copy(self.uav_masks)
        for i in range(self.num_customer_uav):
            if self.parcels_weight[i] > self.uav_capacity[0]:
                # uav_0_masks[i] = 0
                self.uav0_load_masks[i] = 0
            if self.parcels_weight[i] > self.uav_capacity[1]:
                # uav_1_masks[i] = 0
                self.uav1_load_masks[i] = 0
        
        uav_0_masks = self.uav0_load_masks * self.uav_masks
        uav_1_masks = self.uav1_load_masks * self.uav_masks
        
        current_action_masks = {
            agent: (self.truck_masks if match("truck", agent)
                    else uav_0_masks if match("carried_uav_0", agent)
                    else uav_1_masks if match("carried_uav_1", agent)
                    else None)
            for agent in self.possible_agents
        }
        observations = {
            self.possible_agents[i]: {
                "observation": (
                np.row_stack([[self.warehouse_position, self.truck_position], self.uav_position, 
                                 self.customer_position_both, self.customer_position_truck]) if i < self.num_truck 
                else np.row_stack([[self.warehouse_position, self.truck_position, 
                                    self.uav_position[(i - self.num_truck) % self.num_uavs]], 
                                   self.customer_position_both, self.customer_position_uav])), 
                "action_mask": current_action_masks[self.possible_agents[i]]
                }
            for i in range(len(self.possible_agents))
        }

        # # Get dummy infos. Necessary for proper parallel_to_aec conversion
        
        # True means alive while False means dead
        # "alive" means that the agent is taking action
        # "ready" means that decision-making is needed.
        self.infos = {
            a: {
                "IsAlive": False, 
                "IsReady": False
                } if match("returning_uav", a)
                else {
                    "IsAlive": True, 
                    "IsReady": True
                }
            for a in self.possible_agents
            }

        return observations, self.infos
    
    # When the truck performs a new action, it first generates a refined path through genarate_truck_path(),
    # and then moves in truck_move() according to the generated path before reaching the target 
    # (that is, before generating a new action).
    def genarate_truck_path(self, action):
        # target point x, y coordinate
        target = None
        if action == 0:
            target = self.warehouse_position
        elif 0 < action <= self.num_parcels_truck:
            target = self.customer_position_truck[action - 1]
        else:
            target = self.customer_position_both[action - self.num_customer_truck - 1]
            
        # get the id of the grid which truck and target located here...
        id_grid_truck_x = self.truck_position[0] / self.grid_edge
        id_grid_target_x = target[0] / self.grid_edge
        id_grid_truck_y = self.truck_position[1] / self.grid_edge
        id_grid_target_y = target[1] / self.grid_edge
    
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
            return True
        else:
            return False
                
    def carried_uav_move(self, uav, action):
        # get the uav info: kind and no.
        uav_info = findall(r'\d+', uav)
        uav_info = [int(num) for num in uav_info]
        uav_no = uav_info[0] * self.num_uavs_0 + uav_info[1]
        
        # Recovering an action in progress
        if action is None:
            action = self.uav_dist[uav_no]
        
        # get the target coordinate
        target = None
        if action < self.num_customer_both:
            target = self.customer_position_both[action]
        else:
            target = self.customer_position_uav[action - self.num_customer_both]
        
        # Initialize the moving angle and target for the first execution.
        if self.uav_target_dist[uav_no] == MAX_INT:
            self.uav_dist[uav_no] = action
            self.uav_target_dist[uav_no] = np.sqrt(np.sum(np.square(self.uav_position[uav_no] - target)))
            self.uav_target_angle_cos[uav_no] = float(target[0] - self.uav_position[uav_no][0]) / self.uav_target_dist[uav_no]
            self.uav_target_angle_sin[uav_no] = float(target[1] - self.uav_position[uav_no][1]) / self.uav_target_dist[uav_no]
        
        if self.uav_target_dist[uav_no] <= self.step_len * self.uav_velocity[uav_info[0]]:
            self.uav_position[uav_no] = target
            self.uav_battery_remaining[uav_no] -= self.uav_target_dist[uav_no]
            return True
        else:
            self.uav_target_dist[uav_no] -= self.step_len * self.uav_velocity[uav_info[0]]
            self.uav_position[uav_no][0] += int(self.uav_target_angle_cos[uav_no] * self.step_len * self.uav_velocity[uav_info[0]])
            self.uav_position[uav_no][1] += int(self.uav_target_angle_sin[uav_no] * self.step_len * self.uav_velocity[uav_info[0]])
            self.uav_battery_remaining[uav_no] -= self.step_len * self.uav_velocity[uav_info[0]]
            return False
    
    
    def returning_uav_move(self, uav, action):
        # get the uav info: kind and no.
        uav_info = findall(r'\d+', uav)
        uav_info = [int(num) for num in uav_info]
        uav_no = uav_info[0] * self.num_uavs_0 + uav_info[1]
        
        # print(int(np.cos(action) * self.step_len * self.uav_velocity[uav_info[0]]))
        # print(action)
        dist = self.step_len * self.uav_velocity[uav_info[0]]
        self.uav_position[uav_no][0] += int(np.cos(action) * dist)
        self.uav_position[uav_no][1] += int(np.sin(action) * dist)
        
        # update the remaining battery of uav
        self.uav_battery_remaining[uav_no] -= dist
        
        # typically, the uav will returning to the truck to get recovery and load
        if np.sqrt(np.sum(np.square(self.uav_position[uav_no] - self.truck_position))) < DIST_THRESHOLD:
            self.uav_position[uav_no] = copy(self.truck_position)
            return 1
        # The uav may also be returned directly to the warehouse, 
        # but note that in this case, this uav will not be activated again.
        elif np.sqrt(np.sum(np.square(self.uav_position[uav_no] - self.warehouse_position))) < DIST_THRESHOLD:
            self.uav_position[uav_no] = copy(self.warehouse_position)
            return -1
        else:
            return 0
            
    def updata_action_mask(self, agent, action):
        if match("truck", agent):
            if action != 0:
                self.truck_masks[action] = 0
        elif match("carried", agent):
            self.uav_masks[action] = 0
        else:
            pass

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - truck x and y coordinates
        - uavs x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        rewards = {
            "Global": -1, # get -1 reward every transitions to encourage faster delivery
            }
        # Execute actions
        ####
        # processes to be performed here include. 
        # - Update agent coordinate
        # - Update action mask
        # - Update agent state (i.e. infos)
        # - Get the rewards including global rewards and returning rewards
        for agent in actions:
            if match("truck", agent):
                # in the first movement, a refined path needs to be generated.
                if not self.truck_path:
                    self.genarate_truck_path(actions[agent])
                    self.updata_action_mask(agent, actions[agent])
                if self.truck_move():
                    self.infos[agent]["IsReady"] = True
                    # calculate reward when arriving to target
                    if actions[agent] == 0:
                        if np.count_nonzero(self.action_masks) == 2:
                            rewards["Global"] += REWARD_VICTORY
                            self.agents.remove[agent]
                    else:
                        rewards["Global"] += REWARD_DELIVERY
                        
                else:
                    self.infos[agent]["IsReady"] = False # a little bit redundant
        for agent in actions:
            if match("carried", agent):
                uav_info = findall(r'\d+', agent)
                uav_info = [int(num) for num in uav_info]
                uav_no = uav_info[0] * self.num_uavs_0 + uav_info[1]
                
                if actions[agent] != self.num_customer_uav:
                    if self.uav_target_dist[uav_no] == MAX_INT:
                        self.updata_action_mask(agent, actions[agent])
                        
                    uav_moving_result = self.carried_uav_move(agent, actions[agent])
                    if self.uav_battery_remaining[uav_no] <= 0:
                        # that means uav wrecked
                        rewards["Global"] += REWARD_UAV_WRECK
                        self.infos[agent]["IsAlive"] = False
                        self.infos[agent]["IsReady"] = False
                        self.agents.remove(agent)
                    elif uav_moving_result:
                        self.infos[agent]["IsAlive"] = False
                        self.infos[agent]["IsReady"] = False
                        self.infos[agent.replace("carried", "returning")]["IsAlive"] = True
                        self.infos[agent.replace("carried", "returning")]["IsReady"] = True
                        self.agents.remove(agent)
                        self.agents.append(agent.replace("carried", "returning"))
                        # calculate reward when arriving to customer
                        rewards["Global"] += REWARD_DELIVERY
                    else:
                        # can move upwards to execute with update_action_mask
                        self.infos[agent]["IsReady"] = False 
                # When uav doesn't launching, synchronize the coordinates of uav to truck.
                else:
                    self.uav_position[uav_no] = copy(self.truck_position)
                    self.infos[agent]["IsReady"] = True # redundant?
            elif match("returning", agent):
                returning_result = self.returning_uav_move(agent, actions[agent])
                
                uav_info = findall(r'\d+', agent)
                uav_info = [int(num) for num in uav_info]
                uav_no = uav_info[0] * self.num_uavs_0 + uav_info[1]
                if self.uav_battery_remaining[uav_no] <= 0:
                    # that means uav wrecked
                    rewards["Global"] += REWARD_UAV_WRECK
                    self.infos[agent]["IsAlive"] = False
                    self.infos[agent]["IsReady"] = False
                    self.agents.remove(agent)
                
                elif returning_result == 1:
                    self.infos[agent]["IsAlive"] = False
                    self.infos[agent]["IsReady"] = False
                    self.infos[agent.replace("returning", "carried")]["IsAlive"] = True
                    self.infos[agent.replace("returning", "carried")]["IsReady"] = True
                    self.agents.remove(agent)
                    self.agents.append(agent.replace("returning", "carried"))
                    self.uav_battery_remaining[uav_no] = self.uav_range[uav_info[0]]
                    # calculate reward when arriving to customer
                    rewards[agent] = 1
                # when the uav return to warehouse directly, 
                # it won't be re-activated as carried_uav
                elif returning_result == -1:
                    self.infos[agent]["IsAlive"] = False
                    self.infos[agent]["IsReady"] = False
                    self.agents.remove(agent)
                    rewards[agent] = 1
                else:
                    self.infos[agent]["IsReady"] = True # redundant?


        # Check termination conditions
        ####
        terminations = {a: False for a in self.agents}
        if not self.agents:
            terminations = {a: True for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        ####
        truncations = {a: False for a in self.agents}
        if self.time_step > self.MAX_STEP:
            truncations = {a: True for a in self.agents}
            self.agents = []
        self.time_step += 1

        # Get observations
        ####
        # current_action_masks = {
        #     agent: (self.truck_masks if match("truck", agent)
        #             else self.uav_masks if match("carried", agent)
        #             else None)
        #     for agent in self.possible_agents
        # }
        uav_0_masks = self.uav0_load_masks * self.uav_masks
        uav_1_masks = self.uav1_load_masks * self.uav_masks

        current_action_masks = {
            agent: (self.truck_masks if match("truck", agent)
                    else uav_0_masks if match("carried_uav_0", agent)
                    else uav_1_masks if match("carried_uav_1", agent)
                    else None)
            for agent in self.possible_agents
        }
        
        # No difference from the observations and action_masks at reset()
        observations = {
            self.possible_agents[i]: {
                "observation": (
                np.row_stack([[self.warehouse_position, self.truck_position], self.uav_position, 
                                 self.customer_position_both, self.customer_position_truck]) if i < self.num_truck 
                else np.row_stack([[self.warehouse_position, self.truck_position, 
                                    self.uav_position[(i - self.num_truck) % self.num_uavs]], 
                                   self.customer_position_both, self.customer_position_uav])), 
                "action_mask": current_action_masks[self.possible_agents[i]]
                }
            for i in range(len(self.possible_agents))
        }

        # Get dummy infos (not used in this example)
        ####
        
        return observations, rewards, terminations, truncations, self.infos

    def render(self):
        """Renders the environment."""
        # currently, render used mainly in testing rather than visualization :)
        if self.render_mode == None:
            return
        
        screen_width = 500
        screen_height = 500
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
        
        # Load the texture used to render the scene
        map_image = get_image(os.path.join("img", "Map5.png"))
        map_image = pygame.transform.scale(map_image, (screen_width, screen_height))
        warehouse_image = get_image(os.path.join("img", "Warehouse.png"))
        warehouse_image = pygame.transform.scale(warehouse_image, (grid_width * 0.8, grid_height * 0.8))
        truck_image = get_image(os.path.join("img", "Truck.png"))
        truck_image = pygame.transform.scale(truck_image, (grid_width * 0.8, grid_height * 0.8))
        uav_image = get_image(os.path.join("img", "UAV.png"))
        uav_image = pygame.transform.scale(uav_image, (grid_width * 0.6, grid_height * 0.6))
        customer_both_image = get_image(os.path.join("img", "CustomerBoth.png"))
        customer_both_image = pygame.transform.scale(customer_both_image, (grid_width * 0.8, grid_height * 0.8))
        customer_truck_image = get_image(os.path.join("img", "CustomerTruck.png"))
        customer_truck_image = pygame.transform.scale(customer_truck_image, (grid_width * 0.8, grid_height * 0.8))
        customer_uav_image = get_image(os.path.join("img", "CustomerUAV.png"))
        customer_uav_image = pygame.transform.scale(customer_uav_image, (grid_width * 0.8, grid_height * 0.8))
        
        self.screen.blit(map_image, (0, 0))
        
        # There is a scale between simulation coordinates and rendering coordinates, here it is 10:1
        self.screen.blit(warehouse_image, self.warehouse_position / scale - position_bias * 0.4)
        for customer in self.customer_position_both:
            self.screen.blit(customer_both_image, customer / scale - position_bias * 0.4)
        for customer in self.customer_position_truck:
            self.screen.blit(customer_truck_image, customer / scale - position_bias * 0.4)
        for customer in self.customer_position_uav:
            self.screen.blit(customer_uav_image, customer / scale - position_bias * 0.4)
        
        self.screen.blit(truck_image, self.truck_position / scale - position_bias * 0.4)
        for uav in self.uav_position:
            self.screen.blit(uav_image, uav / scale - position_bias * 0.3)
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(2)
            # # print(self.infos)
            # # print(observations)
            # print(self.uav_position[0])
            # # print(self.truck_position)
        
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