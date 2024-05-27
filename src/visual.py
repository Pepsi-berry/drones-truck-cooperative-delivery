import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import torch as th
# from torch import nn
# from base import CustomFeatureExtractor
# from env.uav_env import UAVTrainingEnvironmentWithObstacle
# import hiddenlayer as hl

# from torchviz import make_dot, make_dot_from_trace

data = pd.read_csv('experiment_results_2.csv')

sns.set_theme(style="whitegrid")

params = {'customer_params', 'uav_num', 'obstacle_params', 'time_len_mean', 'time_len_std'}

plt.figure(figsize=(12, 6))
sns.lineplot(x='obstacle_params', y='time_len_mean', hue='Solution', data=data, errorbar=None)
plt.xlabel('UAV num')
plt.ylabel('Time cost')
plt.title('Comparison of Time_Len Across Different Scenarios')
plt.legend(title='Solution')
# plt.show()
plt.savefig('Comparison of Time_Len Across Different Obstacle Density')
plt.close()

