import numpy as np


curri_env_config = {
    'curri_0': {
        'MAX_STEP': 1_000, 
        'step_len': 2, 
        # uav parameters
        # unit here is m/s
        'truck_velocity': 4, 
        'uav_velocity': np.array([12, 29]), 
        
        'num_truck': 1, 
        'num_uavs_0': 4, 
        'num_uavs_1': 4, 
        'num_uavs': 8, 
        
        # obstacle parameters
        'num_uav_obstacle': 0, 
        'num_no_fly_zone': 0, 
        
        'dist_threshold': 50, 
        'generative_range': 750, 
        
        'render_mode': None, 
    }, 
    'curri_1': {
        'MAX_STEP': 1_000, 
        'step_len': 2, 
        # uav parameters
        # unit here is m/s
        'truck_velocity': 4, 
        'uav_velocity': np.array([12, 29]), 
        
        'num_truck': 1, 
        'num_uavs_0': 10, 
        'num_uavs_1': 10, 
        'num_uavs': 20, 
        
        # obstacle parameters
        'num_uav_obstacle': 10, 
        'num_no_fly_zone': 4, 
        
        'dist_threshold': 30, 
        'generative_range': 1_000, 
        
        'render_mode': None, 
    },
    'curri_2': {
        'MAX_STEP': 1_000, 
        'step_len': 2, 
        # uav parameters
        # unit here is m/s
        'truck_velocity': 4, 
        'uav_velocity': np.array([12, 29]), 
        
        'num_truck': 1, 
        'num_uavs_0': 15, 
        'num_uavs_1': 15, 
        'num_uavs': 30, 
        
        # obstacle parameters
        'num_uav_obstacle': 20, 
        'num_no_fly_zone': 6, 
        
        'dist_threshold': 20, 
        'generative_range': 1_000, 
        
        'render_mode': None, 
    }
}