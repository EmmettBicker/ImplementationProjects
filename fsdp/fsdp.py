import os
import torch
import torch.nn as nn
import torch.distributed as dist

class ShardedLinear(nn.Module):
    def __init__(self, in_features, out_features, devices):
        super().__init__()
        
        self.devices = devices
        n_devices = len(self.devices)
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for device in devices:
            self.weights.append(nn.Parameter(torch.randn((out_features * in_features) //n_devices)).to(device))
            self.biases.append(nn.Parameter(torch.randn(out_features//n_devices)).to(device))

        self.streams = [] 
        for _ in devices:
            self.streams.append(torch.cuda.Stream())

    def forward(self, input_tensors: list[torch.Tensor]):
        
                
        local_outputs = [None for _ in self.devices]
        local_device_weights = [None for _ in self.devices]
        local_device_biases = [None for _ in self.devices]
        for idx, (device, stream) in enumerate(zip(self.devices, self.streams)):
            with torch.cuda.stream(stream):
                # # all gather
                w_tensor = torch.cat(tuple(weight.to(device) for weight in self.weights))
                local_device_weights[idx] = w_tensor.view(self.out_features, self.in_features)
                b_tensor = torch.cat(tuple(bias.to(device) for bias in self.biases))
                local_device_biases[idx] = b_tensor.view(self.out_features)
                        
                local_outputs[idx] = nn.functional.linear(input_tensors[idx].to(device), local_device_weights[idx], local_device_biases[idx]) #.to(device)
        torch.cuda.synchronize()
        return local_outputs

class ShardedBatchNorm(nn.Module):
    def __init__(self, num_features, devices):
        super().__init__()
        self.devices = devices
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(num_features).to(device) for device in self.devices])
        
    
    def forward(self, input_tensors: list[torch.Tensor]):
        outpt_data = []
        for idx, batch_norm in enumerate(self.batch_norms):
            curr_tensor = input_tensors[idx].to(self.devices[idx])
            outpt_data.append(batch_norm(curr_tensor))
        return outpt_data
        
class DistributedGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.GELU = nn.GELU()
    
    def forward(self, input_tensors: list[torch.Tensor]):
        
        outputs = [self.GELU(tensor) for tensor in input_tensors]
       
        return outputs
        

class FSDP_NN_Block(nn.Module):
    def __init__(self, inpt_features, outpt_features, devices):
        super().__init__()
        self.sharded_linear = ShardedLinear(inpt_features, outpt_features, devices)
        self.gelu = DistributedGELU()
        self.sharded_batch_norm = ShardedBatchNorm(outpt_features, devices)
    
        self.streams = [] 
        for _ in devices:
            self.streams.append(torch.cuda.Stream())
    
    def forward(self, input_tensors: list[torch.Tensor]):
        input_tensors = self.sharded_linear(input_tensors)
        input_tensors = self.gelu(input_tensors)
        input_tensors = self.sharded_batch_norm(input_tensors)
        return input_tensors


class FSDP_MNIST(nn.Module):
    def __init__(self, devices, input_features = 784, hidden_features = 128, output_features = 10):
        super().__init__()
        self.first_layer = FSDP_NN_Block(784, hidden_features*4, devices)
        self.second_layer = FSDP_NN_Block(hidden_features*4, hidden_features*2, devices)
        self.third_layer = FSDP_NN_Block(hidden_features*2, hidden_features, devices)
        self.fourth_layer = ShardedLinear(hidden_features, output_features, devices)
    
    def forward(self, input_tensors: list[torch.Tensor]):
        input_tensors = self.first_layer(input_tensors)
        input_tensors = self.second_layer(input_tensors)
        input_tensors = self.third_layer(input_tensors)
        input_tensors = self.fourth_layer(input_tensors)
        return input_tensors