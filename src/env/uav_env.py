import os
import functools
import random
from copy import copy
from re import match, findall

import numpy as np
from gymnasium.spaces import Box, MultiDiscrete, Dict, MultiBinary
from gymnasium import Env

# from pettingzoo import ParallelEnv
import pygame

MAX_INT = 2**20
INVALID_ANGLE = 10
# When the distance between the returning uav and the truck is less than this threshold, 
# the return is considered complete.
DIST_THRESHOLD = 20
DIST_RESTRICT_UAV = 20
DIST_RESTRICT_OBSTACLE = 75
# rewards in various situations
REWARD_DELIVERY = 20
REWARD_VICTORY = 100
REWARD_UAV_WRECK = -2
REWARD_UAV_VIOLATE = -2
REWARD_UAV_ARRIVAL = 20
REWARD_URGENCY = -0.05
REWARD_APPROUCHING = 0.02 # get REWARD_APPROUCHING when get closer to target
REWARD_OBSTACLE_AVOIDANCE = -2e-4 # to encourage agents to keep themselves away from obstacles
REWARD_SLOW = -0.02
# color used when rendering no-fly zones and obstacles
COLOR_RESTRICTION = (255, 122, 122)
COLOR_OBSTACLE = (254, 195, 106)

# it seems that wrapping truck and uav into classes would make programming significantly less difficult...
# but when I realize this, it had gone much much too far...

# In once decision, there may be more than one uav selected the same customer as the target point
# special treatment of this needs to be refined(maybe AEV is more suitable? :(

class UAVTrainingEnvironmentWithObstacle(Env):
    """The metadata holds environment constants.
    
    The "name" metadata allows the environment to be pretty printed.
    
    """

    metadata = {
        "render_modes": [None, "human"],
        "name": "training_environment_v1",
    }

    def __init__(
        self, 
        MAX_STEP=10_000, 
        step_len=10, 
        truck_velocity=7, 
        uav_velocity=29, 
        uav_range=15_000, 
        uav_obs_range=150, 
        uav_alert_range=50, 
        num_uav_obstacle=20, 
        num_no_fly_zone=8, 
        render_mode=None
        ):
        """The init method takes in environment arguments.
        
        ensure that the uav_obs_range devisible by pooling_kernal_size=5
        
        These attributes should not be changed after initialization.
        """
        ##########################################################################################
        # These attributes should not be changed after initialization except time_step.
        self.render_mode = render_mode
        self.screen = None
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # self.MAX_STEP = 1_000_000
        self.MAX_STEP = MAX_STEP
        self.step_len = step_len
        self.time_step = 0
        self.option = 0
        
        # uav parameters
        # unit here is m/s
        self.truck_velocity = truck_velocity
        self.uav_velocity = uav_velocity
        # unit here is m
        # self.uav_range = 15_000, 20_000
        self.uav_range = uav_range
        self.uav_obs_pooling_kernal = 5
        self.uav_obs_range = uav_obs_range + self.uav_obs_pooling_kernal
        self.uav_alert_range = uav_alert_range
        
        # map parameters
        self.map_size = 10_000 # m as unit here
        self.grid_edge = 250 # m as unit here
        
        # obstacle parameters
        self.num_uav_obstacle = num_uav_obstacle
        self.num_no_fly_zone = num_no_fly_zone
        
        # The action space of the truck is, choosing a meaningful target point to go to
        # that is warehouse point or customer points which truck can delivery
        # the action space of the uav is moving in any direction and the moving speed
        self.action_space = Box(low=-1, high=1, 
                                 shape=(2, ), dtype=np.float32)
        
        ##########################################################################################
        # agent positions, warehouse positions, customer positions, 
        # action masks to prevent invalid actions to be taken
        self.truck_position = None
        self.customer_position = None
        self.uav_position = None
        self.uav_battery_remaining = None
        # variables used to help representing the movements of the agent in step()
        self.uav_target_position = None
        self.truck_target_position = None
        self.truck_path = None
        
        ##########################################################################################
        # parameters below The following variables will be assigned in reset(), 
        # and not allowed to be changed afterwards.
        
        # Represents areas that uavs cannot pass and 
        # obstacles such as buildings that need to be avoided by uavs
        # The main difference is the size of the range.
        self.no_fly_zones = None
        self.uav_obstacles = None
        
        # 1(warehouse) + 1(truck itself)
        ###########################################################################################
        # 1(uav info: no. and velocity) + 1(dist_uav and dist_obstacle)
        # 1(moving target(current)) + 1(moving target(destination))
        self.observation_space = Dict(
            {
                # 2 means no-fly zones, obstacles & uavs.
                "surroundings": MultiBinary([1, self.uav_obs_range, self.uav_obs_range]), 
                # 2 vectors representing uav to target, target to target of target
                # "vecs": MultiDiscrete(np.full([2 * 2], self.map_size * 2 + 1))
                "vecs": Box(low=-1 * self.map_size, high=self.map_size, shape=[4, ], dtype=np.int32)
            }
        )
    
    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    # @functools.lru_cache(maxsize=None)
    # def observation_space(self):
    #     return self.observation_space

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    # @functools.lru_cache(maxsize=None)
    # def action_space(self):
    #     return self.action_space

    # It is necessary to ensure that the no-fly zone does not contain warehouse and 
    # any customer points that only support delivery by uav.
    # The size of the no-fly zone would better not exceed 3 grids in both the x-axis and y-axis directions.
    # The no-fly zone should probably be distributed a little bit closer to the center of the map
    # In the subsequent update of the environment, 
    # Gaussian distribution may be used instead of uniform sampling.
    def generate_no_fly_zone(self):
        while True:
            not_suitable = False
            upper_left_corner = np.array([random.randint(4 * self.grid_edge, self.map_size - 5 * self.grid_edge),
                                        random.randint(4 * self.grid_edge, self.map_size - 5 * self.grid_edge)])
            range_size = np.array([random.randrange(0.6 * self.grid_edge, 1.2 * self.grid_edge, step=5), 
                                random.randrange(0.6 * self.grid_edge, 1.2 * self.grid_edge, step=5)])
            lower_right_corner = upper_left_corner + range_size
            if self.uav_position[0] > upper_left_corner[0] and self.uav_position[0] < lower_right_corner[0] and self.uav_position[1] > upper_left_corner[1] and self.uav_position[1] < lower_right_corner[1]:
                not_suitable = True
            if not_suitable:
                continue
            return np.array([upper_left_corner, range_size])
        
    
    # Obstacles need to be situated inside the road grid and preferably should not intersect with the road
    def generate_uav_obstacle(self, grid_num):
        upper_left_corner = np.array([random.randint(grid_num / 5, grid_num * 0.8) * self.grid_edge + random.randint(self.grid_edge * 0.2, self.grid_edge * 0.4), 
                          random.randint(grid_num / 5, grid_num * 0.8) * self.grid_edge + random.randint(self.grid_edge * 0.2, self.grid_edge * 0.4)])
        obstacle_size = random.randint(self.grid_edge * 0.3, self.grid_edge * 0.5)
        
        return np.array([upper_left_corner, [obstacle_size, obstacle_size]])
        
    def zones_intersection(self, zone, xlo , xhi, ylo, yhi):
        lower_left = zone[0]
        upper_right = zone[0] + zone[1]
        if xlo > upper_right[0] or xhi < lower_left[0] or ylo > upper_right[1] or yhi < lower_left[1]:
            return None
        else:
            return np.array([
                max(xlo, lower_left[0]), 
                min(xhi, upper_right[0]), 
                max(ylo, lower_left[1]), 
                min(yhi, upper_right[1])])
            
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
    
    def get_obs(self):
        uav_position = self.uav_position
        uav_obs = np.ones([1, self.uav_obs_range, self.uav_obs_range], dtype=np.int8)
        # uav_obs[0] = 1
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
        
        return uav_obs
    
    def reset(self, seed=None, options=None, p0=None):
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
        # 0 means truck, 1 means customer
        if options is None:
            # self.option = 1 - random.randint(0, 1) * random.randint(0, 1) * random.randint(0, 1)
            # self.option = 1 - int(random.randint(1, 32) / 32)
            self.option = 0 # + random.randint(0, 1)
        else:
            self.option = options
        
        grid_num = self.map_size / self.grid_edge
        
        rn = random.randint(0, 1)
        self.truck_position = np.array(
            [random.randint(0.35 * self.map_size, 0.65 * self.map_size), random.randint(0.35 * grid_num, 0.65 * grid_num)*self.grid_edge] if rn % 2 
             else [random.randint(0.35 * grid_num, 0.65 * grid_num)*self.grid_edge, random.randint(0.35 * self.map_size, 0.65 * self.map_size)], 
             dtype=np.int32
        )
        rn = random.randint(0, 1)
        self.customer_position = np.array(
            [random.randint(0.35 * self.map_size, 0.65 * self.map_size), random.randint(0.35 * grid_num, 0.65 * grid_num)*self.grid_edge] if rn % 2 
             else [random.randint(0.35 * grid_num, 0.65 * grid_num)*self.grid_edge, random.randint(0.35 * self.map_size, 0.65 * self.map_size)], 
             dtype=np.int32
        )
        rn = random.randint(0, 1)
        self.truck_target_position = np.array(
            [random.randint(0.35 * self.map_size, 0.65 * self.map_size), random.randint(0.35 * grid_num, 0.65 * grid_num)*self.grid_edge] if rn % 2 
             else [random.randint(0.35 * grid_num, 0.65 * grid_num)*self.grid_edge, random.randint(0.35 * self.map_size, 0.65 * self.map_size)], 
             dtype=np.int32
        )
        
        generative_range = 250 + (1 - random.randint(1, 8) / 8) * 1000
        offset = np.array([random.randint(-1 * generative_range, generative_range), random.randint(-1 * generative_range, generative_range)], dtype=np.int32)
        target_of_target = None
        if self.option:
            self.uav_target_position = self.customer_position
            target_of_target = self.customer_position
        else:
            self.uav_target_position = self.truck_position
            target_of_target = self.truck_target_position
        self.uav_position = self.uav_target_position + offset
        
        if not (p0 is None):
            self.customer_position = p0
            self.uav_target_position = p0
            target_of_target = p0
            self.uav_position = np.array([5000, 5000], dtype=np.int32)

        # Set the initial power of the uav to the full charge
        self.uav_battery_remaining = self.uav_range

        self.truck_path = []
        
        self.no_fly_zones = np.array([self.generate_no_fly_zone() for _ in range(self.num_no_fly_zone)])
        self.uav_obstacles = [self.generate_uav_obstacle(grid_num) for _ in range(self.num_uav_obstacle)] #  + [np.array([self.uav_position + 20, [100, 100]])]

        observations = dict({
            "surroundings" : self.get_obs(), 
            "vecs" : np.concatenate([
                self.uav_target_position - self.uav_position, 
                target_of_target - self.uav_target_position
                ])
            })
        
        # # Get dummy infos. Necessary for proper parallel_to_aec conversion
        
        # infos contain what? 
        info = {
            "dist" : 1
        }

        return observations, info
    
    
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
    def uav_move(self, action):
        # get the uav info: kind and no.
        src_pos = copy(self.uav_position)
        
        dist = self.step_len * action[1]
        # int-ify or not?
        self.uav_position[0] += np.cos(action[0]) * dist
        self.uav_position[1] += np.sin(action[0]) * dist
        
        uav_target = self.uav_target_position
        
        for obstacle in self.uav_obstacles:
            # if insersect or not
            if self.uav_tjc_zone_intersect(obstacle, src_pos, self.uav_position):
                # self.uav_position = obstacle[0] + obstacle[1] / 2
                return -2
        
        for nfz in self.no_fly_zones:
            # consider the outer area as no-fly-zone
            if self.uav_tjc_zone_intersect(nfz, src_pos, self.uav_position) or self.uav_position[0] > self.map_size or self.uav_position[1] > self.map_size or self.uav_position[0] < 0 or self.uav_position[1] < 0:
                return -1
        
        # update the remaining battery of uav
        self.uav_battery_remaining -= dist
        
        # when the distance between uav and target is less than a threshold
        # then consider the uav as arrival 
        if np.sqrt(np.sum(np.square(self.uav_position - uav_target))) < DIST_THRESHOLD:
            self.uav_position = copy(uav_target)
            return 1
        else:
            return 0
        
    
    # when uav collision happens, the action need to rollback
    def uav_move_rollback(self, action):
        dist = self.step_len * action[1]
        # int-ify or not?
        self.uav_position[0] -= np.cos(action[0]) * dist
        self.uav_position[1] -= np.sin(action[0]) * dist
        
        # update the remaining battery of uav
        self.uav_battery_remaining += dist
                
    
    # Convert the normalized action back to the range of the original action distribution
    def denormalize_action(self, action):
        action[0] = action[0] * np.pi + np.pi
        action[1] = action[1] * (self.uav_velocity) / 2 + (self.uav_velocity) / 2
        

    def step(self, action):
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
        
        self.denormalize_action(action)
        # get -0.2 reward every transitions to encourage faster delivery
        rewards = REWARD_URGENCY * self.step_len
        
        # Execute actions
        ####
        # processes to be performed here include. 
        # - Update agent coordinate
        # - Update agent state (i.e. infos)
        # - Get the rewards including global rewards and returning rewards
        
        position_before = copy(self.uav_position)
        dist_before = np.sqrt(np.sum(np.square(self.uav_position - self.uav_target_position)))
        
        # encourage faster speed when not too close
        if dist_before > self.uav_velocity * self.step_len:
            rewards += max((self.uav_velocity * 0.4) - action[1], 0) * REWARD_SLOW
        # truck moves
        # in the first movement, a refined path needs to be generated.
        if not self.truck_path:
            self.genarate_truck_path(self.truck_target_position)
            # self.update_action_mask(agent, self.truck_target_position)
        if self.truck_move():
            # assign a new target for truck here
            rn = random.randint(0, 1)
            grid_num = self.map_size / self.grid_edge
            self.truck_target_position = np.array(
                [random.randint(0.35 * self.map_size, 0.65 * self.map_size), random.randint(0.35 * grid_num, 0.65 * grid_num)*self.grid_edge] if rn % 2 
                else [random.randint(0.35 * grid_num, 0.65 * grid_num)*self.grid_edge, random.randint(0.35 * self.map_size, 0.65 * self.map_size)] 
            )

        terminated = False
        uav_moving_result = self.uav_move(action)
        if uav_moving_result == 1:
            rewards += REWARD_UAV_ARRIVAL
            terminated = True
        elif uav_moving_result == -1:
            rewards += REWARD_UAV_VIOLATE
            self.uav_position = position_before
            # terminated = True
        elif uav_moving_result == -2:
            rewards += REWARD_UAV_WRECK
            self.uav_position = position_before
            # terminated = True
        else:
            dist_diff = dist_before - np.sqrt(np.sum(np.square(self.uav_position - self.uav_target_position)))
            rewards += REWARD_APPROUCHING * dist_diff

        # Check truncation conditions (overwrites termination conditions)
        ####
        truncated = False
        if self.time_step >= self.MAX_STEP:
            truncated = True
        else:
            truncated = False
        self.time_step += 1

        # Get observations
        ####
        target_of_target = None
        if self.option:
            target_of_target = self.customer_position
        else:
            target_of_target = self.truck_target_position
        
        coordi = np.concatenate([
                self.uav_target_position - self.uav_position, 
                target_of_target - self.uav_target_position
                ])
        
        surroundings = self.get_obs()
        alert_area = surroundings[0][int(self.uav_obs_range / 2 - self.uav_alert_range) : int(self.uav_obs_range / 2 + self.uav_alert_range + 1), int(self.uav_obs_range / 2 - self.uav_alert_range) : int(self.uav_obs_range / 2 + self.uav_alert_range + 1)]
        rewards += np.count_nonzero(alert_area) * REWARD_OBSTACLE_AVOIDANCE
        # No big difference from the observations and action_masks at reset()
        observations = dict({
            "surroundings" : surroundings, 
            "vecs" : coordi
            })
        
        info = {
            "dist" : 1
        }

        
        return observations, rewards, terminated, truncated, info

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
        
        # Load the texture used to render the scene
        map_image = get_image(os.path.join("img", "Map5.png"))
        map_image = pygame.transform.scale(map_image, (screen_width, screen_height))
        warehouse_image = get_image(os.path.join("img", "Warehouse.png"))
        warehouse_image = pygame.transform.scale(warehouse_image, (grid_width * 0.8, grid_height * 0.8))
        truck_image = get_image(os.path.join("img", "Truck.png"))
        truck_image = pygame.transform.scale(truck_image, (grid_width * 0.8, grid_height * 0.8))
        uav_image = get_image(os.path.join("img", "UAV.png"))
        uav_image = pygame.transform.scale(uav_image, (grid_width * 0.6, grid_height * 0.6))
        # customer_both_image = get_image(os.path.join("img", "CustomerBoth.png"))
        # customer_both_image = pygame.transform.scale(customer_both_image, (grid_width * 0.8, grid_height * 0.8))
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
            uav_obstacle_image = pygame.Surface(uav_obstacle[1] / scale)
            uav_obstacle_image.fill(COLOR_OBSTACLE)
            self.screen.blit(uav_obstacle_image, uav_obstacle[0] / scale)
        
        self.screen.blit(customer_truck_image, self.truck_target_position / scale - position_bias * 0.4)
        self.screen.blit(customer_uav_image, self.customer_position / scale - position_bias * 0.4)
        
        self.screen.blit(truck_image, self.truck_position / scale - position_bias * 0.4)
        self.screen.blit(uav_image, self.uav_position / scale - position_bias * 0.3)
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(10)
        
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

