from typing import Union
from gymnasium import Space
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device

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

        # Update the features dim manually
        self._features_dim = total_concat_size
        
        
        
    def forward(self, observations):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # print(key, extractor)
            # print(key, observations[key].shape, extractor)
            encoded_tensor_list.append(extractor(observations[key]))
            # print(encoded_tensor_list)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # print(encoded_tensor_list[0].shape, encoded_tensor_list[1].shape, th.cat(encoded_tensor_list, dim=1).shape)
        return self.extractor_concat(th.cat(encoded_tensor_list, dim=1))
