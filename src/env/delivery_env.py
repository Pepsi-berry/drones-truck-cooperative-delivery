import functools
import random
from copy import copy
from re import match

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from pettingzoo import ParallelEnv


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
        # These attributes should not be changed after initialization.
        self.render_mode = render_mode
        
        self.possible_agents = ["truck", 
                                "carried_uav_1_1", "carried_uav_1_2", 
                                "carried_uav_2_1", "carried_uav_2_2", 
                                "carried_uav_2_3", "carried_uav_2_4", 
                                "returning_uav_1_1", "returning_uav_1_2", 
                                "returning_uav_2_1", "returning_uav_2_2", 
                                "returning_uav_2_3", "returning_uav_2_4", 
                                ]
        # uav parameters
        # unit here is km/h
        self.truck_velocity = 25
        self.uav_1_velocity = 44
        self.uav_2_velocity = 104
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
        self.weight_probabilities = [0.8, 0.1, 0.1]
        
        # map parameters
        self.map_size = 15_000 # m as unit here
        self.grid_edge = 250 # m as unit here
        
        # The action space of the truck is, choosing a meaningful target point to go to
        # that is warehouse point or customer points which truck can delivery
        self.action_space_truck = Discrete(self.num_customer_truck + 1)
        # The action space of the carried uav is similar to truck
        # action_space_uav_carried[0] is set to empty action
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
        self.agents = ["truck", 
                       "carried_uav_1_1", "carried_uav_1_2", 
                       "carried_uav_2_1", "carried_uav_2_2", 
                       "carried_uav_2_3", "carried_uav_2_4", 
                       ]
        self.truck_position = None
        self.uav_position = None
        
        ##########################################################################################
        # parameters below The following variables will be assigned in reset(), 
        # and not allowed to be changed afterwards.
        # through invalid action masks to prevent the agent from going to the customer point,
        # where the delivery has been completed
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
        self.MAX_STEP = 1_000_000
        self.timestep = None
    
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

    def generate_weight():
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
             for i in range(self.num_parcels - self.num_parcels_truck - self.num_parcels_uav)]
            )
        
        # parcel weights probability distribution: 
        # <= 3.6kg: 0.8, 3.6kg - 10kg: 0.1, > 10kg: 0.1
        self.parcels_weight = np.array([self.generate_weight() for _ in range(self.num_parcels)])
        
        # Initially, the truck departs from the warehouse
        # Drones travel with trucks
        self.truck_position = copy(self.warehouse_position)
        self.uav_position = np.array([copy(self.truck_position) for _ in range(self.num_uavs)])
        # self.uav_position = {
        #     uav: copy(self.truck_position) 
        #     for uav in self.possible_agents
        #     if not match("truck", uav)
        #     }
        
        # Action masks is to be set here...
        action_mask = {
            agent: (np.ones(self.num_customer_truck + 1) if match("truck", agent)
                    else np.ones(self.num_customer_uav + 1) if match("carried", agent)
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
                "action_mask": ()
                }
            for i in range(len(self.possible_agents))
        }

        # observation = (
        #     self.prisoner_x + 7 * self.prisoner_y,
        #     self.guard_x + 7 * self.guard_y,
        #     self.escape_x + 7 * self.escape_y,
        # )
        # observations = {
        #     "prisoner": {"observation": observation, "action_mask": [0, 1, 1, 0]},
        #     "guard": {"observation": observation, "action_mask": [1, 0, 0, 1]},
        # }

        # # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - prisoner x and y coordinates
        - guard x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        # Execute actions
        prisoner_action = actions["prisoner"]
        guard_action = actions["guard"]

        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1
        elif prisoner_action == 1 and self.prisoner_x < 6:
            self.prisoner_x += 1
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1
        elif prisoner_action == 3 and self.prisoner_y < 6:
            self.prisoner_y += 1

        if guard_action == 0 and self.guard_x > 0:
            self.guard_x -= 1
        elif guard_action == 1 and self.guard_x < 6:
            self.guard_x += 1
        elif guard_action == 2 and self.guard_y > 0:
            self.guard_y -= 1
        elif guard_action == 3 and self.guard_y < 6:
            self.guard_y += 1

        # Generate action masks
        prisoner_action_mask = np.ones(4, dtype=np.int8)
        if self.prisoner_x == 0:
            prisoner_action_mask[0] = 0  # Block left movement
        elif self.prisoner_x == 6:
            prisoner_action_mask[1] = 0  # Block right movement
        if self.prisoner_y == 0:
            prisoner_action_mask[2] = 0  # Block down movement
        elif self.prisoner_y == 6:
            prisoner_action_mask[3] = 0  # Block up movement

        guard_action_mask = np.ones(4, dtype=np.int8)
        if self.guard_x == 0:
            guard_action_mask[0] = 0
        elif self.guard_x == 6:
            guard_action_mask[1] = 0
        if self.guard_y == 0:
            guard_action_mask[2] = 0
        elif self.guard_y == 6:
            guard_action_mask[3] = 0

        # Action mask to prevent guard from going over escape cell
        if self.guard_x - 1 == self.escape_x:
            guard_action_mask[0] = 0
        elif self.guard_x + 1 == self.escape_x:
            guard_action_mask[1] = 0
        if self.guard_y - 1 == self.escape_y:
            guard_action_mask[2] = 0
        elif self.guard_y + 1 == self.escape_y:
            guard_action_mask[3] = 0

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards = {"prisoner": -1, "guard": 1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            rewards = {"prisoner": 1, "guard": -1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        # Check truncation conditions (overwrites termination conditions)
        truncations = {"prisoner": False, "guard": False}
        if self.timestep > 100:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
            self.agents = []
        self.timestep += 1

        # Get observations
        observation = (
            self.prisoner_x + 7 * self.prisoner_y,
            self.guard_x + 7 * self.guard_y,
            self.escape_x + 7 * self.escape_y,
        )
        observations = {
            "prisoner": {
                "observation": observation,
                "action_mask": prisoner_action_mask,
            },
            "guard": {"observation": observation, "action_mask": guard_action_mask},
        }

        # Get dummy infos (not used in this example)
        infos = {"prisoner": {}, "guard": {}}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        # grid = np.zeros((7, 7))
        # grid[self.prisoner_y, self.prisoner_x] = "P"
        # grid[self.guard_y, self.guard_x] = "G"
        # grid[self.escape_y, self.escape_x] = "E"
        # print(f"{grid} \n")
        pass
