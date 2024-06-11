import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import ast
import matplotlib.font_manager as fm
# import torch as th
# from torch import nn
# from base import CustomFeatureExtractor
# from env.uav_env import UAVTrainingEnvironmentWithObstacle
# import hiddenlayer as hl

# from torchviz import make_dot, make_dot_from_trace
font_path = 'src/visualize/SimHei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

data = pd.read_csv('experiment_results_2_1.csv') 
# data['customer_params'] = data['customer_params'].apply(ast.literal_eval)
# data['num_customer'] = data['customer_params'].apply(lambda x: x['num_parcels_truck'] + x['num_parcels_uav'])
# data = data.sort_values(by='num_customer')

sns.set_theme(style="whitegrid")

params = {'customer_params', 'uav_num', 'obstacle_params', 'time_len_mean', 'time_len_std'}

plt.figure(figsize=(12, 6))
lineplot = sns.barplot(x='uav_num', y='time_len_mean', hue='Solution', data=data, errorbar=None)
# for line in lineplot.lines:
#     for x, y in zip(line.get_xdata(), line.get_ydata()):
#         lineplot.text(x, y, f'{y:.2f}', color='black', ha='right')
plt.xlabel('搭载无人机数目', fontproperties=font_prop)
plt.ylabel('平均配送用时', fontproperties=font_prop)
plt.title('改变搭载无人机数目时的不同方案配送用时对比图', fontproperties=font_prop)
plt.legend(title='Solution')
# plt.show()
plt.savefig('Comparison of Time_Len Across Different Number of UAVs')
plt.close()

