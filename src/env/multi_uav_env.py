import os
import functools
from copy import copy
from re import match, findall
# from itertools import combinations

import numpy as np
from gymnasium.spaces import Box, Dict, MultiBinary
from gymnasium.utils import seeding

from pettingzoo import ParallelEnv
import pygame

MAX_INT = 2**20
INVALID_ANGLE = 10
# When the distance between the returning uav and the truck is less than this threshold, 
# the return is considered complete.
DIST_RESTRICT_UAV = 20
DIST_RESTRICT_OBSTACLE = 75
# rewards in various situations
REWARD_DELIVERY = 20
REWARD_VICTORY = 100
REWARD_UAV_WRECK = -2
REWARD_UAV_VIOLATE = -2
REWARD_UAV_ARRIVAL = 20
REWARD_URGENCY = -0.2
REWARD_APPROUCHING = 0.02 # get REWARD_APPROUCHING when get closer to target
REWARD_UAVS_DANGER = float(-400) # coefficient of the penalty for being to close with other uavs
# REWARD_OBSTACLE_AVOIDANCE = -2e-4 # to encourage agents to keep themselves away from obstacles
# REWARD_SLOW = -0.02
# color used when rendering no-fly zones and obstacles
COLOR_RESTRICTION = (255, 122, 122)
COLOR_OBSTACLE = (254, 195, 106)

# it seems that wrapping truck and uav into classes would make programming significantly less difficult...
# but when I realize this, it had gone much much too far...

# In once decision, there may be more than one uav selected the same customer as the target point
# special treatment of this needs to be refined(maybe AEV is more suitable? :(

class MultiUAVsTrainingEnvironmentWithObstacle(ParallelEnv):
    """The metadata holds environment constants.
    
    The "name" metadata allows the environment to be pretty printed.
    
    """

    metadata = {
        "render_modes": [None, "human", "rgb_array"],
        "name": "training_environment_v2",
    }

    def __init__(
        self, 
        MAX_STEP=2_000, 
        step_len=10, 
        truck_velocity=7, 
        uav_velocity=np.array([12, 29]), 
        uav_range=np.array([10_000, 15_000]), 
        uav_obs_range=150, 
        num_truck=1, 
        num_uavs=6, 
        num_uavs_0=2, 
        num_uavs_1=4, 
        num_uav_obstacle=20, 
        num_no_fly_zone=8, 
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
        
        # self.MAX_STEP = 1_000_000
        self.MAX_STEP = MAX_STEP
        self.step_len = step_len
        self.time_step = 0
        self.curriculum_reservation = -1
        self.curriculum = None
        self.dist_threshold = 20
        self.generative_range = 750 #  + int(self.RNG.integers(1, 4) / 4) * 1000
        
        # uav parameters
        # unit here is m/s
        self.truck_velocity = truck_velocity
        self.uav_velocity = copy(uav_velocity)

        # unit here is m
        # self.uav_range = 15_000, 20_000
        self.uav_range = copy(uav_range)
        self.uav_obs_pooling_kernal = 5
        self.uav_obs_range = uav_obs_range + self.uav_obs_pooling_kernal
                
        self.num_truck = num_truck
        self.num_uavs = num_uavs
        self.num_uavs_0 = num_uavs_0
        self.num_uavs_1 = num_uavs_1
        
        
        # map parameters
        self.map_size = 10_000 # m as unit here
        self.grid_edge = 250 # m as unit here
        
        # obstacle parameters
        self.num_uav_obstacle = num_uav_obstacle
        self.num_no_fly_zone = num_no_fly_zone
        
        self.possible_agents = ["uav_0_" + str(i) for i in range(self.num_uavs_0)] + ["uav_1_" + str(i) for i in range(self.num_uavs_1)]
        
        self.uav_name_mapping = dict(zip([agent for agent in self.possible_agents], list(range(self.num_uavs))))
                
        # the action space of the uav is moving direction and speed
        self.action_spaces = {
            agent: (
                Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
            ) 
            for agent in self.possible_agents
        }
        
        ##########################################################################################
        # agent positions, warehouse positions, customer positions, 
        self.agents = None
        self.truck_position = None
        self.uav_position = None
        # variables used to help representing the movements of the agent in step()
        self.uav_target_positions = None
        self.truck_target_position = None
        self.truck_path = None
        
        ##########################################################################################
        # parameters below The following variables will be assigned in reset(), 
        # and not allowed to be changed afterwards.
        ###########################################################################################
        # Indicates the current stage of uavs
        # IMPORTANT
        # -1 means not launched, 0 means returning, 1 means delivery
        self.uav_stages = None

        self.customer_position_uav = None
        
        # Represents areas that uavs cannot pass and 
        # obstacles such as buildings that need to be avoided by uavs
        # The main difference is the size of the range.
        self.no_fly_zones = None
        self.uav_obstacles = None
        # uav_positions_transfer stores the first and last coordinates of uavs moving at current step, 
        # which is used to detect path intersections and determine conflicts between uavs.
        self.uav_positions_transfer = None
        
        # 1(warehouse) + 1(truck itself)
        ###########################################################################################
        # 1(uav info: no. and velocity) + 1(dist_uav and dist_obstacle)
        # 1(moving target(current)) + 1(moving target(destination))
        self.observation_spaces = {
            agent: (
                Dict(
                    {
                        "surroundings": MultiBinary([3, self.uav_obs_range, self.uav_obs_range]), 
                        # 2 vectors representing uav to target, target to target of target
                        "vecs": Box(low=-1 * self.map_size, high=self.map_size, shape=[4, ], dtype=np.int32)
                    }
                )
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
    
    def get_uav_info(self, uav):
        # get the uav info: kind and no.
        uav_info = findall(r'\d+', uav)
        uav_info = [int(num) for num in uav_info]
        uav_no = uav_info[0] * self.num_uavs_0 + uav_info[1]
        return uav_info + [uav_no]


    # It is necessary to ensure that the no-fly zone does not contain warehouse and 
    # any customer points that only support delivery by uav.
    # The size of the no-fly zone would better not exceed 3 grids in both the x-axis and y-axis directions.
    # The no-fly zone should probably be distributed a little bit closer to the center of the map
    # In the subsequent update of the environment, 
    # Gaussian distribution may be used instead of uniform sampling.
    def generate_no_fly_zone(self):
        # testing use
        while True:
            not_suitable = False
            upper_left_corner = np.array([self.RNG.integers(4 * self.grid_edge, self.map_size - 5 * self.grid_edge),
                                        self.RNG.integers(4 * self.grid_edge, self.map_size - 5 * self.grid_edge)])
            range_size = np.array([self.RNG.integers(0.8 * self.grid_edge, 1.2 * self.grid_edge), 
                                self.RNG.integers(0.8 * self.grid_edge, 1.2 * self.grid_edge)])
            lower_right_corner = upper_left_corner + range_size
            for customer in self.customer_position_uav:
                if customer[0] > upper_left_corner[0] and customer[0] < lower_right_corner[0] and customer[1] > upper_left_corner[1] and customer[1] < lower_right_corner[1]:
                    not_suitable = True
            for uav in self.uav_position:
                if uav[0] > upper_left_corner[0] and uav[0] < lower_right_corner[0] and uav[1] > upper_left_corner[1] and uav[1] < lower_right_corner[1]:
                    not_suitable = True
            if not_suitable:
                continue
            return np.array([upper_left_corner, range_size])
        
    
    # Obstacles need to be situated inside the road grid and preferably should not intersect with the road
    def generate_uav_obstacle(self, grid_num):
        upper_left_corner = np.array([self.RNG.integers(grid_num * 0.2, grid_num * 0.8) * self.grid_edge + self.RNG.integers(self.grid_edge * 0.2, self.grid_edge * 0.4), 
                          self.RNG.integers(grid_num * 0.2, grid_num * 0.8) * self.grid_edge + self.RNG.integers(self.grid_edge * 0.2, self.grid_edge * 0.4)])
        obstacle_size = self.RNG.integers(int(self.grid_edge * 0.3), int(self.grid_edge * 0.5))
        
        return np.array([upper_left_corner, [obstacle_size, obstacle_size]])
        
        
    def zones_intersection(self, zone, xlo , xhi, ylo, yhi):
        lower_left = zone[0]
        upper_right = zone[0] + zone[1]
        if xlo > upper_right[0] or xhi < lower_left[0] or ylo > upper_right[1] or yhi < lower_left[1]:
            return None
        else:
            return np.array([
                max(xlo, lower_left[0]), 
                min(xhi, upper_right[0]) + 1, 
                max(ylo, lower_left[1]), 
                min(yhi, upper_right[1]) + 1])
            
            
    def zones_intersection_fast_check(self, lower_left, upper_right, xlo, xhi, ylo, yhi):
        return not (xlo > upper_right[0] or xhi < lower_left[0] or ylo > upper_right[1] or yhi < lower_left[1])
    
            
    def uav_tjc_zone_intersect(self, zone, src_pos, dst_pos):
        lower_left = zone[0]
        upper_right = zone[0] + zone[1]
        # check if one of the end of tjc inside the rect zone
        if (src_pos[0] > lower_left[0] and src_pos[0] < upper_right[0] and src_pos[1] > lower_left[1] and src_pos[1] < upper_right[1]) or (dst_pos[0] > lower_left[0] and dst_pos[0] < upper_right[0] and dst_pos[1] > lower_left[1] and dst_pos[1] < upper_right[1]):
            return True
        
        # check if the tjc intersect with one of the diagonal
        # fast check
        tjc_xlo, tjc_xhi, tjc_ylo, tjc_yhi = (
            min(src_pos[0], dst_pos[0]), 
            max(src_pos[0], dst_pos[0]), 
            min(src_pos[1], dst_pos[1]), 
            max(src_pos[1], dst_pos[1])
        )
        if not self.zones_intersection_fast_check(lower_left, upper_right, tjc_xlo, tjc_xhi, tjc_ylo, tjc_yhi):
            return False
        # cross product check
        lower_right = np.array([upper_right[0], lower_left[1]])
        upper_left = np.array([lower_left[0], upper_right[1]])
        # ignore the case of cross * cross == 0, 
        # which may means that the line segments are on the same line
        # but not happened here, just mention :)
        if (
            cross_product_2d_array(dst_pos - src_pos, lower_left - src_pos) * cross_product_2d_array(dst_pos - src_pos, upper_right - src_pos) < 0 
            and 
            cross_product_2d_array(upper_right - lower_left, src_pos - lower_left) * cross_product_2d_array(upper_right - lower_left, dst_pos - lower_left) < 0
            ):
            return True
        if (cross_product_2d_array(dst_pos - src_pos, lower_right - src_pos) * cross_product_2d_array(dst_pos - src_pos, upper_left - src_pos) < 0
            and 
            cross_product_2d_array(lower_right - upper_left, src_pos - upper_left) * cross_product_2d_array(lower_right - upper_left, dst_pos - upper_left) < 0
            ):
            return True
        
        return False
    
    
    def uav_tjcs_intersect(self, src_pos, dst_pos):
        uav_names = []
        for this_start_pos, this_end_pos, uav_name in self.uav_positions_transfer:
            # cross product check
            # ignore the case of cross * cross == 0, 
            # which may means that the line segments are on the same line
            # but not happened here, just mention :)
            if (
                not np.array_equal(np.concatenate([src_pos, dst_pos]), np.concatenate([this_start_pos, this_end_pos]))
                and
                np.int64(cross_product_2d_array(dst_pos - src_pos, this_start_pos - src_pos)) * cross_product_2d_array(dst_pos - src_pos, this_end_pos - src_pos) <= 0 
                and 
                np.int64(cross_product_2d_array(this_end_pos - this_start_pos, src_pos - this_start_pos)) * cross_product_2d_array(this_end_pos - this_start_pos, dst_pos - this_start_pos) <= 0
                ):
                uav_names.append(uav_name)
        
        if uav_names:
            return uav_names
        
        return False    


    # return if the dist between uav l and r is safe through this transition
    def uav_safe_distance_detection(self, traj_l, traj_r, num):
        # skip the initial point
        trajs_l = np.linspace(traj_l[1], traj_l[0], num=num, endpoint=False)
        trajs_r = np.linspace(traj_r[1], traj_r[0], num=num, endpoint=False)
        
        dist_min = np.inf
        for point_l, point_r in zip(trajs_l, trajs_r):
            dist = np.sqrt(np.sum(np.square(point_l - point_r)))
            dist_min = min(dist_min, dist)
        
        return dist_min > DIST_RESTRICT_UAV


    def get_obs_by_uav(self, uav):
        uav_position = self.uav_position[self.uav_name_mapping[uav]]
        uav_obs = np.zeros([3, self.uav_obs_range, self.uav_obs_range], dtype=np.int8)
        uav_obs[0] = 1
        obs_radius = int(self.uav_obs_range / 2)
        x_offset = uav_position[0] - obs_radius
        y_offset = uav_position[1] - obs_radius
        
        xlo , xhi, ylo, yhi = (
            max(uav_position[0] - obs_radius, 0), 
            min(uav_position[0] + obs_radius, self.map_size), 
            max(uav_position[1] - obs_radius, 0), 
            min(uav_position[1] + obs_radius, self.map_size)
        )
        # no-fly zones obs
        uav_obs[0][int(xlo - x_offset) : int(xhi - x_offset) + 1, int(ylo - y_offset) : int(yhi - y_offset) + 1] = 0
        
        for nfz in self.no_fly_zones:
            intersection = self.zones_intersection(nfz, xlo, xhi, ylo, yhi)
            if intersection is None:
                continue
            uav_obs[0][int(intersection[0] - x_offset) : int(intersection[1] - x_offset), int(intersection[2] - y_offset) : int(intersection[3] - y_offset)] = 1
        
        # obstacles obs
        for obstacle in self.uav_obstacles:
            intersection = self.zones_intersection(obstacle, xlo, xhi, ylo, yhi)
            if intersection is None:
                continue
            uav_obs[0][int(intersection[0] - x_offset) : int(intersection[1] - x_offset), int(intersection[2] - y_offset) : int(intersection[3] - y_offset)] = 1
        
        # uav obs
        boundary_max_x = uav_position[0] + obs_radius
        boundary_max_y = uav_position[1] + obs_radius
        boundary_min_x = uav_position[0] - obs_radius
        boundary_min_y = uav_position[1] - obs_radius
        for uav_no in range(self.num_uavs):
            if self.uav_position[uav_no][0] < xhi and self.uav_position[uav_no][0] > xlo and self.uav_position[uav_no][1] < yhi and self.uav_position[uav_no][1] > ylo and self.uav_stages[uav_no] >= 0 and uav_no != self.uav_name_mapping[uav]: # the uav need to be launched
                uav_obs[1][int(self.uav_position[uav_no][0] - x_offset)][int(self.uav_position[uav_no][1] - y_offset)] = 1
                # add the position of other observable agents’ goals 
                # (clip into the boundary of the FOV if outside of the FOV
                uav_target = None
                if self.uav_stages[uav_no] == 1:
                    uav_target = copy(self.uav_target_positions[uav_no])
                else:
                    uav_target = copy(self.truck_position)
                uav_obs[2][int(np.clip(uav_target[0], boundary_min_x, boundary_max_x) - x_offset)][int(np.clip(uav_target[1], boundary_min_y, boundary_max_y) - y_offset)] = 1
        
        return uav_obs
    
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
        # Initially, all UAVs are in state of delivery
        self.agents = self.possible_agents.copy()
        self.uav_stages = np.ones(self.num_uavs)
        # All customer points are distributed at the edge of the road grid
        grid_num = self.map_size / self.grid_edge
        self.truck_target_position = np.array(
            [self.RNG.integers(0.4 * self.map_size, 0.6 * self.map_size), self.RNG.integers(0.4 * grid_num, 0.6 * grid_num)*self.grid_edge] if self.RNG.integers(0, 1, endpoint=True) 
             else [self.RNG.integers(0.4 * grid_num, 0.6 * grid_num)*self.grid_edge, self.RNG.integers(0.4 * self.map_size, 0.6 * self.map_size)], 
             dtype=np.int32
            )
        self.truck_position = np.array(
            [self.RNG.integers(0.4 * self.map_size, 0.6 * self.map_size), self.RNG.integers(0.4 * grid_num, 0.6 * grid_num)*self.grid_edge] if self.RNG.integers(0, 1, endpoint=True) 
             else [self.RNG.integers(0.4 * grid_num, 0.6 * grid_num)*self.grid_edge, self.RNG.integers(0.4 * self.map_size, 0.6 * self.map_size)], 
             dtype=np.int32
            )
        self.customer_position_uav = np.array(
            [[self.RNG.integers(0.4 * self.map_size, 0.6 * self.map_size), self.RNG.integers(0.4 * grid_num, 0.6 * grid_num)*self.grid_edge] if i % 2 
             else [self.RNG.integers(0.4 * grid_num, 0.6 * grid_num)*self.grid_edge, self.RNG.integers(0.4 * self.map_size, 0.6 * self.map_size)] 
             for i in range(self.num_uavs)], dtype=np.int32
            )

        self.truck_path = []
        self.uav_target_positions = copy(self.customer_position_uav)
        self.uav_positions_transfer = []
        
        generative_lower_bound = 0
        self.uav_position = copy(self.uav_target_positions)
        # offset = np.array([self.RNG.integers(-1 * self.generative_range, self.generative_range), self.RNG.integers(-1 * self.generative_range, self.generative_range)], dtype=np.int32)
        for uav in self.uav_position:
            offset = np.array([
                self.RNG.choice([self.RNG.integers(-1 * self.generative_range, -1 * generative_lower_bound), self.RNG.integers(generative_lower_bound, self.generative_range)]), 
                self.RNG.choice([self.RNG.integers(-1 * self.generative_range, -1 * generative_lower_bound), self.RNG.integers(generative_lower_bound, self.generative_range)])
            ])
            
            uav += offset
        # print("*uav positions*", self.uav_target_positions, self.uav_position)
        
        self.no_fly_zones = np.array([self.generate_no_fly_zone() for _ in range(self.num_no_fly_zone)])
        self.uav_obstacles = [self.generate_uav_obstacle(grid_num) for _ in range(self.num_uav_obstacle)]
        
        coordi = {
            agent: np.concatenate(
                [
                    (
                        self.uav_target_positions[self.uav_name_mapping[agent]] if self.uav_stages[self.uav_name_mapping[agent]] == 1
                        else self.truck_position # when the stage == -1, result doesn't matter
                    ) - self.uav_position[self.uav_name_mapping[agent]], 
                    (
                        np.array([0, 0]) if self.uav_stages[self.uav_name_mapping[agent]] == 1
                        else self.truck_target_position - self.truck_position
                    )
                ]
            )
            for agent in self.agents
        }
        observations = {
            agent: 
                    dict({
                        "surroundings" : self.get_obs_by_uav(agent), 
                        "vecs" : coordi[agent].astype(np.int32)
                    })
                    for agent in self.agents
        }
        
        # # Get dummy infos. Necessary for proper parallel_to_aec conversion
        
        # infos contain if agent is available for TA
        # if there is at least one agent(or truck?) available, then run the upper solver
        infos = {
            a: {
                "training_enabled": True
            }
            for a in self.agents
        }
        
        return observations, infos
    
    
    def reserve_curriculum(self, curri):
        if self.curriculum_reservation == -1:
            self.curriculum_reservation = curri
    
    
    # set the difficulty of the env
    def set_curriculum(self, curri):
        if self.curriculum is None:
            self.curriculum = curri
        else:
            self.curriculum = max(self.curriculum, curri)
        if self.curriculum == 0:
            self.num_uavs = 8
            self.num_uavs_0 = 4
            self.num_uavs_1 = 4
            self.num_uav_obstacle = 0
            self.num_no_fly_zone = 0
            self.dist_threshold = 50
            self.generative_range = 750
        elif self.curriculum == 1:
            self.num_uavs = 20
            self.num_uavs_0 = 10
            self.num_uavs_1 = 10
            self.num_uav_obstacle = 10
            self.num_no_fly_zone = 4
            self.dist_threshold = 30
            self.generative_range = 1000
        elif self.curriculum == 2:
            self.num_uavs = 30
            self.num_uavs_0 = 15
            self.num_uavs_1 = 15
            self.num_uav_obstacle = 20
            self.num_no_fly_zone = 6
            self.dist_threshold = 20
            self.generative_range = 1000
        else:
            raise ValueError("Unknown curriculum setting: ", curri)
        
        self.reset()
    
    
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
            return True
        else:
            return False
    
    # obstacle judgement is to be added...
    # update the location of uav and return the result of movement
    def uav_move(self, uav, action):
        # get the uav info: kind and no.
        uav_no = self.uav_name_mapping[uav]
        
        src_pos = copy(self.uav_position[uav_no])
        
        dist = self.step_len * action[1]
        # int-ify or not?
        self.uav_position[uav_no][0] += np.cos(action[0]) * dist
        self.uav_position[uav_no][1] += np.sin(action[0]) * dist
        
        uav_target = None
        if self.uav_stages[uav_no] == 1:
            uav_target = copy(self.uav_target_positions[uav_no])
        else:
            uav_target = copy(self.truck_position)
        
        # append the first and last coordinates of the uav that has completed the movement
        self.uav_positions_transfer.append([src_pos, copy(self.uav_position[uav_no]), uav])
        
        for obstacle in self.uav_obstacles:
            # if insersect or not
            if self.uav_tjc_zone_intersect(obstacle, src_pos, self.uav_position[uav_no]):
                self.uav_position[uav_no] = obstacle[0] + obstacle[1] / 2
                return -2
            
        # move the collision judgment to step() and change to distance detection
        # is_intersect = self.uav_tjcs_intersect(src_pos, self.uav_position[uav_no])
        # if is_intersect:
        #     return is_intersect
        
        for nfz in self.no_fly_zones:
            if self.uav_tjc_zone_intersect(nfz, src_pos, self.uav_position[uav_no]):
                return -1
        
        # when the distance between uav and target is less than a threshold
        # then consider the uav as arrival 
        if np.sqrt(np.sum(np.square(self.uav_position[uav_no] - uav_target))) < self.dist_threshold:
            self.uav_position[uav_no] = copy(uav_target)
            return 1
        # The uav may also be returned directly to the warehouse, 
        # but note that in this case, this uav will not be activated again.
        else:
            return 0
    
    
    # Convert the normalized action back to the range of the original action distribution
    def denormalize_action(self, actions):
        # print("action infos", actions)
        for agent in actions:
            actions[agent].setflags(write=True)
            actions[agent][0] = actions[agent][0] * np.pi + np.pi
            actions[agent][1] = actions[agent][1] * (self.uav_velocity[1]) / 2 + (self.uav_velocity[1]) / 2


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
        self.denormalize_action(actions)
        
        # print(actions)
        
        # only the transfer of uav position infos in the current step is retained in uav_positions_transfer
        self.uav_positions_transfer = []
        # contains the name of the uav in which the uav-uav path collision occurred
        # uav_collision_set = set()
        
        rewards = {
            agent: REWARD_URGENCY * self.step_len for agent in self.agents
            # get -0.1 reward every transitions to encourage faster delivery
            }
        
        truck_position_before = copy(self.truck_position)
        
        # truck moves
        # in the first movement, a refined path needs to be generated.
        if not self.truck_path:
            self.genarate_truck_path(self.truck_target_position)
        if self.truck_move():
            # a new target point needs to be assigned to the truck before all uavs return to the truck
            grid_num = self.map_size / self.grid_edge
            self.truck_target_position = np.array(
                [self.RNG.integers(0.4 * self.map_size, 0.6 * self.map_size), self.RNG.integers(0.4 * grid_num, 0.6 * grid_num)*self.grid_edge] if self.RNG.integers(0, 1, endpoint=True) 
                else [self.RNG.integers(0.4 * grid_num, 0.6 * grid_num)*self.grid_edge, self.RNG.integers(0.4 * self.map_size, 0.6 * self.map_size)], 
                dtype=np.int32
                )                    

        positions_before = copy(self.uav_position)
        agents_before = copy(self.agents)
        
        truncations = {a: False for a in self.agents}
        terminations = {a: False for a in self.agents}
        infos = {
            a: {
                "training_enabled": True
            }
            for a in self.agents
        }
        for agent in actions: 
            # stage == -1 means unlaunched
            uav_no = self.uav_name_mapping[agent]
            if self.uav_stages[self.uav_name_mapping[agent]] != -1:
                # rewards[agent] += max((self.uav_velocity[1] * 0.4) - actions[agent][1], 0) * REWARD_SLOW
                
                uav_moving_result = self.uav_move(agent, actions[agent])
                
                if uav_moving_result == 1:
                    self.uav_stages[uav_no] -= 1
                    if self.uav_stages[uav_no] == -1:
                        # self.infos[agent] = True
                        terminations[agent] = True
                        self.agents.remove(agent)
                        # self.infos.pop(agent)
                    rewards[agent] += REWARD_UAV_ARRIVAL
                elif uav_moving_result == -1:
                    rewards[agent] += REWARD_UAV_VIOLATE
                    self.uav_position[uav_no] = positions_before[uav_no]
                elif uav_moving_result == -2:
                    # give negative reward without removing uav to improve training stability
                    rewards[agent] += REWARD_UAV_WRECK
                    self.uav_position[uav_no] = positions_before[uav_no]
                elif uav_moving_result == 0:
                    # calculate the distance before the uav moves, 
                    # to calculate the reward for the distance change of the uav movement
                    uav_target = None
                    uav_target_before = None
                    if self.uav_stages[uav_no] == 1:
                        uav_target = copy(self.uav_target_positions[uav_no])
                        uav_target_before = copy(self.uav_target_positions[uav_no])
                    else:
                        uav_target = copy(self.truck_position)
                        uav_target_before = truck_position_before
                    dist_before = np.sqrt(np.sum(np.square(positions_before[uav_no] - uav_target_before)))
                    dist_diff = dist_before - np.sqrt(np.sum(np.square(self.uav_position[uav_no] - uav_target)))
                    rewards[agent] += REWARD_APPROUCHING * dist_diff
                # else: # uav-and-uav collision case (to be modified...)
                #     uav_collision_set.add(agent)
                #     self.uav_position[uav_no] = positions_before[uav_no]
                #     for this_uav_name in uav_moving_result:
                #         uav_collision_set.add(this_uav_name)
                #         this_uav_no = self.uav_name_mapping[this_uav_name]
                #         self.uav_position[this_uav_no] = positions_before[this_uav_no]
                
            else:
                self.uav_position[uav_no] = copy(self.truck_position)
        
        # for uav_name in uav_collision_set:
        #     rewards[uav_name] += REWARD_UAV_WRECK
        
        # check if the distance between uavs is less than safe distance
        # for traj_l, traj_r in combinations(self.uav_positions_transfer, 2):
        #     if self.uav_safe_distance_detection(traj_l, traj_r, 1):
        #         rewards[traj_l[-1]] += REWARD_UAV_WRECK
        #         rewards[traj_r[-1]] += REWARD_UAV_WRECK
        
        # add uav-uav closing penalty, based on position gravity model
        uavs_surrounding = {
            uav: self.get_obs_by_uav(uav) for uav in agents_before
        }
        centroid_idx = int(self.uav_obs_range / 2)
        for uav in uavs_surrounding:
            surrounding = uavs_surrounding[uav]
            non_zero_idx = np.transpose(np.nonzero(surrounding[1]))
            if non_zero_idx.size != 0:
                rewards[uav] += np.sum(REWARD_UAVS_DANGER / (np.sum(
                    np.square(
                        non_zero_idx - np.array([centroid_idx, centroid_idx])
                    ), axis=1
                ) + 50))
        
        # Check termination conditions
        ####
        if np.sum(self.uav_stages) == (-1) * self.num_uavs or not self.agents:
            terminations = {a: True for a in terminations}
            self.agents = []

        # Check truncation conditions (overwrites termination conditions)
        ####
        if self.time_step >= self.MAX_STEP:
            truncations = {a: True for a in truncations}
            self.agents = []
        self.time_step += 1

        # Get observations
        ####
        coordi = {
            agent: np.concatenate(
                [
                    (
                        self.uav_target_positions[self.uav_name_mapping[agent]] if self.uav_stages[self.uav_name_mapping[agent]] == 1
                        else self.truck_position # when the stage == -1, result doesn't matter
                    ) - self.uav_position[self.uav_name_mapping[agent]], 
                    (
                        np.array([0, 0]) if self.uav_stages[self.uav_name_mapping[agent]] == 1
                        else self.truck_target_position - self.truck_position
                    )
                ]
            )
            for agent in agents_before
        }
        
        # No big difference from the observations at reset()
        observations = {
            agent: 
                    dict({
                        "surroundings" : uavs_surrounding[agent], 
                        "vecs" : coordi[agent].astype(np.int32)
                    })
                    for agent in agents_before
        }

        # Get dummy infos (not used in this example)
        ####
        
        # if not self.agents and self.curriculum_reservation >= 0:
        #     self.set_curriculum(self.curriculum_reservation)
        #     self.curriculum_reservation = -1
        #     print(
        #         "env curriculum config has been switch to: ", 
        #         {   
        #             'num_uavs_0': self.num_uavs_0, 
        #             'num_uavs_1': self.num_uavs_1, 
        #             'num_uavs': self.num_uavs, 
                    
        #             # obstacle parameters
        #             'num_uav_obstacle': self.num_uav_obstacle, 
        #             'num_no_fly_zone': self.num_no_fly_zone, 
                    
        #             'dist_threshold': self.dist_threshold, 
        #             'generative_range': self.generative_range, 
        #         }, 
        #     )
        
        return observations, rewards, terminations, truncations, infos

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
        for no_fly_zone in self.no_fly_zones:
            no_fly_zone_image = pygame.Surface(no_fly_zone[1] / scale)
            no_fly_zone_image.fill(COLOR_RESTRICTION)
            no_fly_zone_image.set_alpha(100)
            self.screen.blit(no_fly_zone_image, no_fly_zone[0] / scale)
        for uav_obstacle in self.uav_obstacles:
            # uav_obstacle_image = pygame.Surface(position_bias * 0.6, pygame.SRCALPHA)
            # pygame.draw.circle(uav_obstacle_image, COLOR_OBSTACLE, position_bias * 0.3, uav_obstacle[1] / scale)
            uav_obstacle_image = pygame.Surface(uav_obstacle[1] / scale)
            uav_obstacle_image.fill(COLOR_OBSTACLE)
            self.screen.blit(uav_obstacle_image, uav_obstacle[0] / scale)
        
        self.screen.blit(customer_truck_image, self.truck_target_position / scale - position_bias * 0.4)
        for customer in self.customer_position_uav:
            self.screen.blit(customer_uav_image, customer / scale - position_bias * 0.4)
        
        self.screen.blit(truck_image, self.truck_position / scale - position_bias * 0.4)
        for uav in self.uav_position:
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
