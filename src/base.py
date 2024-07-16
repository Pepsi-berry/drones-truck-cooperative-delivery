import os
from typing import Union
from gymnasium import Space
from ray.rllib.utils.typing import ModelConfigDict
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.callbacks import DefaultCallbacks
# from ray.rllib.utils.typing import TensorType, ModelConfigDict
# from ray.rllib.utils.spaces.space_utils import flatten_space

# BATCH_SIZE = 1024

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def  __init__(
        self, 
        observation_space: Space, 
        # net_arch: Union[List[int], Dict[str, List[int]]],
        # features_dim: int = 0, 
        device: Union[th.device, str] = "auto", 
    ) -> None:
        super().__init__(observation_space, features_dim=1)
        device = get_device(device)
        
        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "surroundings":
                n_input_channel = observation_space[key].shape[0]
                extractors[key] = nn.Sequential(
                    # input: 155 * 155 * 1
                    nn.MaxPool2d(kernel_size=5), # output: 31 * 31 * 1
                    nn.Conv2d(n_input_channel, 64, kernel_size=3, stride=1, padding=1), 
                    # padding=1 equals to padding="same" while padding=0 equals to padding="valid"
                    nn.ReLU(), 
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(), 
                    # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
                    # nn.ReLU(), 
                    nn.MaxPool2d(kernel_size=2), # output: 15 * 15 * 64
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(), 
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(), 
                    # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
                    # nn.ReLU(), 
                    nn.MaxPool2d(kernel_size=2), # output: 7 * 7 * 128
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(), 
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(), 
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(), 
                    nn.MaxPool2d(kernel_size=2), # output: 3 * 3 * 256
                    nn.Conv2d(256, 500, kernel_size=3, stride=1, padding=0), # output: 1 * 1 * 500
                    # nn.ReLU(), 
                    # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
                    # nn.ReLU(), 
                    # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
                    # nn.ReLU(), 
                    nn.Flatten(), 
                    nn.ReLU()
                ).to(device)
                # with th.no_grad():
                #     print(
                #         extractors[key](
                #             th.as_tensor(observation_space.sample()[key]).float()
                #         ).shape
                #     )
                total_concat_size += 500
            elif key == "vecs":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 12), 
                    nn.ReLU()
                ).to(device)
                total_concat_size += 12

        self.extractors = nn.ModuleDict(extractors)
        # print(extractors)
        
        self.extractor_concat = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Linear(512, 512), 
            nn.ReLU()
        ).to(device)
        
        # with th.no_grad():
        #     print(
        #         self.extractor_concat(
        #             th.cat(
        #                 [extractors["surroundings"](th.as_tensor(observation_space.sample()["surroundings"]).float())] + 
        #                 [extractors["vecs"](th.as_tensor(observation_space.sample()["vecs"]).unsqueeze(0).float())], dim=1                
        #             )   
        #         )
        #         # extractors["vecs"](th.as_tensor(observation_space.sample()["vecs"]).float()).shape           
        #     )
        
        # self.lstm = nn.LSTM(512, 512, batch_first=True).to(device)
        # self.h_state = th.zeros(1, 512).to(device)
        # self.c_state = th.zeros(1, 512).to(device)

        # Update the features dim manually
        self._features_dim = total_concat_size
        
        
        
    def forward(self, observations):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # print(key, observations[key].dtype)
            # print(key, extractor)
            # print(key, observations[key].shape, extractor)
            encoded_tensor_list.append(extractor(observations[key]))
            # print(key, extractor(observations[key]).shape)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # print(encoded_tensor_list[0].shape, encoded_tensor_list[1].shape, th.cat(encoded_tensor_list, dim=1).shape)
        out = self.extractor_concat(th.cat(encoded_tensor_list, dim=1))
        # out, _ = self.lstm(out, (self.h_state, self.c_state))
        # print(out.shape)
        # out = out[-1, :]
        # print(out.shape)
        return out

class CustomPPOPolicyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int, model_config: dict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        extractors = {}
        total_concat_size = 0
        # print("*Debug Infos*", obs_space, ", ", action_space, ", ", num_outputs, ", ", model_config)
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in obs_space.spaces.items():
            if key == "surroundings":
                n_input_channel = obs_space[key].shape[0]
                extractors[key] = nn.Sequential(
                    # input: 155 * 155 * 1
                    nn.MaxPool2d(kernel_size=5), # output: 31 * 31 * 1
                    nn.Conv2d(n_input_channel, 64, kernel_size=3, stride=1, padding=1), 
                    # padding=1 equals to padding="same" while padding=0 equals to padding="valid"
                    nn.ReLU(), 
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(), 
                    nn.MaxPool2d(kernel_size=2), # output: 15 * 15 * 64
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(), 
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(), 
                    nn.MaxPool2d(kernel_size=2), # output: 7 * 7 * 128
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(), 
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(), 
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(), 
                    nn.MaxPool2d(kernel_size=2), # output: 3 * 3 * 256
                    nn.Conv2d(256, 500, kernel_size=3, stride=1, padding=0), # output: 1 * 1 * 500
                    nn.Flatten(), 
                    nn.ReLU()
                )
                total_concat_size += 500
            elif key == "vecs":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 12), 
                    nn.ReLU()
                )
                total_concat_size += 12

        self.extractors = nn.ModuleDict(extractors)
        # print(extractors)
        
        self.extractor_concat = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Linear(512, 512), 
            nn.ReLU()
        )
                
        self.policy_net = nn.Sequential(
            nn.Linear(512, 64), 
            nn.Tanh(), 
            nn.Linear(64, 64), 
            nn.Tanh(), 
            nn.Linear(64, num_outputs)
        )
        self.value_net = nn.Sequential(
            nn.Linear(512, 64), 
            nn.Tanh(), 
            nn.Linear(64, 64), 
            nn.Tanh(), 
            nn.Linear(64, 1)
        )
        self._value_out = None
    
        
    def forward(
        self, 
        input_dict, 
        state, 
        seq_lens, 
        ):
        # print("*Debug Infos 2*", input_dict["obs"]['vecs'].dtype)
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # print(key, extractor)
            encoded_tensor_list.append(extractor(input_dict['obs'][key].to(th.float32)))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # print(encoded_tensor_list[0].shape, encoded_tensor_list[1].shape, th.cat(encoded_tensor_list, dim=1).shape)
        out = self.extractor_concat(th.cat(encoded_tensor_list, dim=1))
        action_out = self.policy_net(out)
        self._value_out = self.value_net(out)
        
        return action_out, state
    
    
    def value_function(self):
        return self._value_out.flatten()
    
    
class CustomCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.best_mean_reward = -float("inf")
        self.checkpoint_dir = "./training/models/mappo"

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        mean_reward = result['env_runners']["episode_reward_mean"]
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            checkpoint_path = algorithm.save(self.checkpoint_dir)
            print(f"New best mean reward: {mean_reward:.2f}. Checkpoint saved at {checkpoint_path}")
