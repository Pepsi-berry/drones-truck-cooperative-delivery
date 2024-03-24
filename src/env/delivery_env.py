import functools
import random
from copy import copy
from re import match

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from pettingzoo import ParallelEnv

MAX_INT = 2**20

class DeliveryEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "render_mode": [None],
        "name": "delivery_environment_v0",
    }

    def __init__(self, render_mode=None):
        """The init method takes in environment arguments.


        These attributes should not be changed after initialization.
        """
        ##########################################################################################
        # These attributes should not be changed after initialization except time_step.
        self.render_mode = render_mode
        
        self.MAX_STEP = 1_000_000
        self.step_len = 10
        self.time_step = None
        
        self.possible_agents = ["truck", 
                                "carried_uav_1_1", "carried_uav_1_2", 
                                "carried_uav_2_1", "carried_uav_2_2", 
                                "carried_uav_2_3", "carried_uav_2_4", 
                                "returning_uav_1_1", "returning_uav_1_2", 
                                "returning_uav_2_1", "returning_uav_2_2", 
                                "returning_uav_2_3", "returning_uav_2_4", 
                                ]
        # uav parameters
        # unit here is m/s
        self.truck_velocity = 7
        self.uav_1_velocity = 12
        self.uav_2_velocity = 29
        # unit here is kg
        self.uav_1_capacity = 10
        self.uav_2_capacity = 3.6
        # unit here is m
        # self.uav_1_range = 15_000
        # self.uav_2_range = 20_000
        self.uav_1_range = 10_000
        self.uav_2_range = 15_000
        
        self.num_truck = 1
        self.num_uavs = 6
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
        self.map_size = 15_000 # m as unit here
        self.grid_edge = 250 # m as unit here
        
        # The action space of the truck is, choosing a meaningful target point to go to
        # that is warehouse point or customer points which truck can delivery
        self.action_space_truck = Discrete(self.num_customer_truck + 1)
        # The action space of the carried uav is similar to truck
        # action_space_uav_carried[num_customer_uav + 1] is set to empty action
        self.action_space_uav_carried = Discrete(self.num_customer_uav + 1)
        # the action space of the returning uav is chasing the truck in any direction
        self.action_space_uav_returning = Box(low=0, high=np.pi, shape=(1, ))
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
        # variables used to help representing the movement of the agent in step()
        self.uav_target_dist = None
        self.truck_path = None

        # The following three *_mask will be assigned as slices of action_mask. 
        # The slicing operation will return the view of the original data, 
        # so that the modification of *_mask will have an impact on the original data.
        # self.customer_both_masks = None
        # self.customer_truck_masks = None
        # self.customer_uav_masks = None
        self.truck_masks = None
        self.uav_masks = None
        
        
        ##########################################################################################
        # parameters below The following variables will be assigned in reset(), 
        # and not allowed to be changed afterwards.
        # through invalid action masks to prevent the agent from going to the customer point,
        # where the delivery has been completed
        self.action_masks = None
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
                MultiDiscrete(np.full([(1 + 1 + self.num_uavs + self.num_customer_truck), 2], 15_001)) if match("truck", agent) 
                else MultiDiscrete(np.full([(1 + 1 + 1 + self.num_customer_uav), 2], 15_001))
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
                       "carried_uav_1_1", "carried_uav_1_2", 
                       "carried_uav_2_1", "carried_uav_2_2", 
                       "carried_uav_2_3", "carried_uav_2_4", 
                       ]
        # The warehouse is located in the center of the map
        self.warehouse_position = np.array([self.map_size / 2, self.map_size / 2])
        # All customer points are distributed at the edge of the road grid
        self.customer_position_truck = np.array(
            [[random.randint(0, self.map_size), random.randint(0, 60)*self.grid_edge] if i % 2 
             else [random.randint(0, 60)*self.grid_edge, random.randint(0, self.map_size)] 
             for i in range(self.num_parcels_truck)]
            )
        self.customer_position_uav = np.array(
            [[random.randint(0, self.map_size), random.randint(0, 60)*self.grid_edge] if i % 2 
             else [random.randint(0, 60)*self.grid_edge, random.randint(0, self.map_size)] 
             for i in range(self.num_parcels_uav)]
            )
        self.customer_position_both = np.array(
            [[random.randint(0, self.map_size), random.randint(0, 60)*self.grid_edge] if i % 2 
             else [random.randint(0, 60)*self.grid_edge, random.randint(0, self.map_size)] 
             for i in range(self.num_customer_both)]
            )
        
        # Initially, the target points of all agents are not determined
        # So the uav_taeget_dist is set to inf and the truck path is set to empty
        self.uav_target_dist = np.full(self.num_uavs, MAX_INT)
        self.truck_path = []
        
        # parcel weights probability distribution: 
        # <= 3.6kg: 0.8, 3.6kg - 10kg: 0.1, > 10kg: 0.1
        self.parcels_weight = np.array([self.generate_weight() for _ in range(self.num_parcels)])
        
        # Initially, the truck departs from the warehouse
        # Drones travel with trucks
        # The number of trucks is temporarily set to 1. 
        # If it is expanded to multiple trucks later
        # all the variable related to the trucks including here needs to be expanded.
        self.truck_position = copy(self.warehouse_position)
        self.uav_position = np.array([copy(self.truck_position) for _ in range(self.num_uavs)])
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
        current_action_masks = {
            agent: (self.truck_masks if match("truck", agent)
                    else self.uav_masks if match("carried", agent)
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
        infos = {
            a: False if match("returning_uav", a)
                else True
            for a in self.possible_agents
            }

        return observations, infos
    
    # When the truck performs a new action, it first generates a refined path through genarate_truck_path(),
    # and then moves in truck_move() according to the generated path before reaching the target 
    # (that is, before generating a new action).
    def genarate_truck_path(self, action):
        # target point x, y coordinate
        target = None
        if action == 0:
            target = self.warehouse_position
        elif 0 < action <= self.num_customer_truck:
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
    
    def truck_move(self, action):
        # target point x, y coordinate
        time_left = self.step_len
        while self.truck_path:
            if time_left == 0:
                break
            if abs(self.truck_position[0] + self.truck_position[1] - self.truck_path[0][0] - self.truck_path[0][1]) <= self.truck_velocity * time_left:
                self.truck_position[0] = self.truck_path[0][0]
                self.truck_position[1] = self.truck_path[1][1]
                time_left -= abs(self.truck_position[0] + self.truck_position[1] - self.truck_path[0][0] - self.truck_path[0][1]) / float(self.truck_velocity)
                self.truck_path.pop(0)
            elif self.truck_position[0] == self.truck_path[0][0]:
                self.truck_position[1] += (time_left * self.truck_velocity if self.truck_position[1] < self.truck_path[0][1] 
                                           else time_left * self.truck_velocity * (-1))
                time_left = 0
        if not self.truck_path:
            return True
        else:
            return False
                
    def carried_uav_move(self, uav, action):
        pass
    
    def returning_uav_move(self, uav, action):
        pass
    
    def updata_action_mask(self, agent, action):
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
        # Execute actions
        ####
        for agent in actions:
            if match("truck", agent):
                self.truck_move(actions[agent]) # processing the return is to be done here...
            elif match("carried", agent):
                if actions[agent] != 0:
                    self.carried_uav_move(agent, actions[agent] - 1)
            else:
                self.returning_uav_move(agent, actions[agent])


        # ~~Generate~~ update action masks
        ####
        
        for a in actions:
            if match("truck", actions[agent]):
                if 0 < actions[agent] < self.num_customer_both:
                    # self.action_masks[a] = 0
                    pass
        # prisoner_action_mask = np.ones(4, dtype=np.int8)
        # if self.prisoner_x == 0:
        #     prisoner_action_mask[0] = 0  # Block left movement
        # elif self.prisoner_x == 6:
        #     prisoner_action_mask[1] = 0  # Block right movement
        # if self.prisoner_y == 0:
        #     prisoner_action_mask[2] = 0  # Block down movement
        # elif self.prisoner_y == 6:
        #     prisoner_action_mask[3] = 0  # Block up movement

        # guard_action_mask = np.ones(4, dtype=np.int8)
        # if self.guard_x == 0:
        #     guard_action_mask[0] = 0
        # elif self.guard_x == 6:
        #     guard_action_mask[1] = 0
        # if self.guard_y == 0:
        #     guard_action_mask[2] = 0
        # elif self.guard_y == 6:
        #     guard_action_mask[3] = 0


        # Check termination conditions
        ####
        # terminations = {a: False for a in self.agents}
        # rewards = {a: 0 for a in self.agents}
        # if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
        #     rewards = {"prisoner": -1, "guard": 1}
        #     terminations = {a: True for a in self.agents}
        #     self.agents = []

        # elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
        #     rewards = {"prisoner": 1, "guard": -1}
        #     terminations = {a: True for a in self.agents}
        #     self.agents = []

        # Check truncation conditions (overwrites termination conditions)
        ####
        # truncations = {"prisoner": False, "guard": False}
        # if self.timestep > 100:
        #     rewards = {"prisoner": 0, "guard": 0}
        #     truncations = {"prisoner": True, "guard": True}
        #     self.agents = []
        # self.timestep += 1
        self.time_step += 1

        # Get observations
        ####
        # observation = (
        #     self.prisoner_x + 7 * self.prisoner_y,
        #     self.guard_x + 7 * self.guard_y,
        #     self.escape_x + 7 * self.escape_y,
        # )
        # observations = {
        #     "prisoner": {
        #         "observation": observation,
        #         "action_mask": prisoner_action_mask,
        #     },
        #     "guard": {"observation": observation, "action_mask": guard_action_mask},
        # }

        # Get dummy infos (not used in this example)
        ####
        # infos = {"prisoner": {}, "guard": {}}

        # return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        # grid = np.zeros((7, 7))
        # grid[self.prisoner_y, self.prisoner_x] = "P"
        # grid[self.guard_y, self.guard_x] = "G"
        # grid[self.escape_y, self.escape_x] = "E"
        # print(f"{grid} \n")
        pass
