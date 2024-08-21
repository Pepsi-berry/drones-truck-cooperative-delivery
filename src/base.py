import os
from typing import Union
# from functools import partial
from gymnasium import Space
# from ray.rllib.utils.typing import ModelConfigDict
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from env.training_config import curri_env_config
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
        # print('Input_Dict: ', input_dict["obs"])
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
    
    
class CustomSACPolicyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int, model_config: dict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        extractors = {}
        # print("*Debug Infos*", obs_space, ", ", action_space, ", ", num_outputs)
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
            elif key == "vecs":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 12), 
                    nn.ReLU()
                )

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
    
        
    def forward(
        self, 
        input_dict, 
        state, 
        seq_lens, 
        ):
        # print("*Debug Infos 2*", input_dict["obs"]['vecs'].dtype)
        # print('Input_Dict: ', input_dict["obs"])
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # print(key, extractor)
            encoded_tensor_list.append(extractor(input_dict['obs'][key].to(th.float32)))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # print(encoded_tensor_list[0].shape, encoded_tensor_list[1].shape, th.cat(encoded_tensor_list, dim=1).shape)
        out = self.extractor_concat(th.cat(encoded_tensor_list, dim=1))
        action_out = self.policy_net(out)
        
        return action_out, state

    
class CustomSACQModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int, model_config: dict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        extractors = {}
        # print("*Debug Infos*", obs_space, ", ", action_space, ", ", num_outputs, ", ", obs_space.spaces)
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        obs_space = obs_space[0]
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
                    nn.Conv2d(256, 480, kernel_size=3, stride=1, padding=0), # output: 1 * 1 * 500
                    nn.Flatten(), 
                    nn.ReLU()
                )
            elif key == "vecs":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 16), 
                    nn.ReLU()
                )

        self.extractors = nn.ModuleDict(extractors)
        # print(extractors)
        
        self.action_extractor = nn.Sequential(
            nn.Linear(action_space.shape[0], 16), 
            nn.ReLU()
        )
        
        self.extractor_concat = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Linear(512, 512), 
            nn.ReLU()
        )
                
        self.value_net = nn.Sequential(
            nn.Linear(512, 64), 
            nn.Tanh(), 
            nn.Linear(64, 64), 
            nn.Tanh(), 
            nn.Linear(64, num_outputs)
        )
    
        
    def forward(
        self, 
        input_dict, 
        state, 
        seq_lens, 
        ):
        # print("*Debug Infos 2*", input_dict["obs"]['vecs'].dtype)
        # print('Input_Dict: ', input_dict["obs"])
        obs = input_dict['obs'][0]
        actions = input_dict["obs"][1]
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # print(key, extractor)
            encoded_tensor_list.append(extractor(obs[key].to(th.float32)))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # print(encoded_tensor_list[0].shape, encoded_tensor_list[1].shape, th.cat(encoded_tensor_list, dim=1).shape)
        encoded_tensor_list.append(self.action_extractor(actions.to(th.float32)))
        out = self.extractor_concat(th.cat(encoded_tensor_list, dim=1))
        q_value_out = self.value_net(out)
        
        return q_value_out, state
    

class CustomRNNSACPolicyModel(RecurrentNetwork, nn.Module):
    def __init__(
        self, 
        obs_space: Space, 
        action_space: Space, 
        num_outputs: int, 
        model_config: dict, 
        name: str
        ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        extractors = {}
        # print("*Debug Infos*", obs_space, ", ", action_space, ", ", num_outputs)
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
            elif key == "vecs":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 12), 
                    nn.ReLU()
                )

        self.extractors = nn.ModuleDict(extractors)
        # print(extractors)
        
        self.extractor_concat = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Linear(512, 512), 
            nn.ReLU()
        )
        
        self.gru = nn.GRU(input_size=512, hidden_size=512, batch_first=True)
        
        self.policy_net = nn.Sequential(
            nn.Linear(512, 64), 
            nn.Tanh(), 
            nn.Linear(64, 64), 
            nn.Tanh(), 
            nn.Linear(64, num_outputs)
        )
        
    
    # return initial hidden state
    def get_initial_state(self):
        h = th.zeros(1, 512)
        
        return [h]
    
    
    def forward(self, 
                input_dict: th.Dict, 
                state: th.List, 
                seq_lens) -> th.Tuple:
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        flat_inputs = { k: v.float() for k, v in input_dict["obs_flat"].items() }
        # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
        # as input_dict may have extra zero-padding beyond seq_lens.max().
        # Use add_time_dimension to handle this
        self.time_major = self.model_config.get("_time_major", False)
        
        print(flat_inputs['surroundings'].shape)
        inputs = {
            k: add_time_dimension(
                v, 
                seq_lens=seq_lens, 
                framework="torch", 
                time_major=self.time_major, 
            )
            for k, v in flat_inputs.items()
        }
        # inputs = add_time_dimension(
        #     flat_inputs,
        #     seq_lens=seq_lens,
        #     framework="torch",
        #     time_major=self.time_major,
        # )
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = th.reshape(output, [-1, self.num_outputs])
        return output, new_state
    
      
    def forward_rnn(
        self, 
        input_dict, 
        state, 
        seq_lens, 
        ):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(input_dict[key].to(th.float32)))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        out = self.extractor_concat(th.cat(encoded_tensor_list, dim=1))
        rnn_out, h_state = self.gru(out, state[0])
        action_out = self.policy_net(rnn_out)
        
        return action_out, h_state

    
class CustomRNNSACQModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int, model_config: dict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        extractors = {}
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        obs_space = obs_space[0]
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
                    nn.Conv2d(256, 480, kernel_size=3, stride=1, padding=0), # output: 1 * 1 * 500
                    nn.Flatten(), 
                    nn.ReLU()
                )
            elif key == "vecs":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 16), 
                    nn.ReLU()
                )

        self.extractors = nn.ModuleDict(extractors)
        
        self.action_extractor = nn.Sequential(
            nn.Linear(action_space.shape[0], 16), 
            nn.ReLU()
        )
        
        self.extractor_concat = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Linear(512, 512), 
            nn.ReLU()
        )
        
        self.gru = nn.GRU(input_size=512, hidden_size=512, batch_first=True)
                
        self.value_net = nn.Sequential(
            nn.Linear(512, 64), 
            nn.Tanh(), 
            nn.Linear(64, 64), 
            nn.Tanh(), 
            nn.Linear(64, num_outputs)
        )
    
    
    # return initial hidden state
    def get_initial_state(self):
        h = th.zeros(1, 512)
        
        return [h]
    
    
    def forward(self, 
                input_dict: th.Dict, 
                state: th.List, 
                seq_lens) -> th.Tuple:
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        flat_inputs = { k: v.float() for k, v in input_dict["obs_flat"].items() }
        # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
        # as input_dict may have extra zero-padding beyond seq_lens.max().
        # Use add_time_dimension to handle this
        self.time_major = self.model_config.get("_time_major", False)
        
        inputs = {
            k: add_time_dimension(
                v, 
                seq_lens=seq_lens, 
                framework="torch", 
                time_major=self.time_major, 
            )
            for k, v in flat_inputs.items()
        }

        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = th.reshape(output, [-1, self.num_outputs])
        return output, new_state
    
        
    def forward_rnn(
        self, 
        input_dict, 
        state, 
        seq_lens, 
        ):
        obs = input_dict['obs'][0]
        actions = input_dict["obs"][1]
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(obs[key].to(th.float32)))
        
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        encoded_tensor_list.append(self.action_extractor(actions.to(th.float32)))
        out = self.extractor_concat(th.cat(encoded_tensor_list, dim=1))
        rnn_out, h_state = self.gru(out, state[0])
        q_value_out = self.policy_net(rnn_out)
        
        return q_value_out, h_state
    


def _remote_fn(env_runner, new_curri: int):
    # We recreate the entire env object by changing the env_config on the worker,
    # then calling its `make_env()` method.
    env_runner.config.environment(env_config=curri_env_config["curri_" + str(new_curri)])
    env_runner.make_env()
    
    
class CustomCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.best_mean_reward = -float("inf")
        self.checkpoint_dir = "./training/models/masac"

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        mean_reward = result['env_runners'].get("episode_reward_mean", -float("inf"))
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            checkpoint_path = algorithm.save(self.checkpoint_dir)
            print(f"New best mean reward: {mean_reward:.2f}. Checkpoint saved at {checkpoint_path}")
        
        # curri = algorithm._counters.get("current_env_curri", 0)
        # if mean_reward > 300 and curri < 2:
        #     curri = curri + 1
        #     algorithm._counters["current_env_curri"] = curri

        #     algorithm.workers.foreach_worker(
        #         lambda ev: ev.foreach_env(
        #             lambda env: env.par_env.reserve_curriculum(curri)
        #         )
        #     )
        #     # algorithm._counters["current_env_curri"] = curri
        #     print("Curriculum has been set to: ", curri)

