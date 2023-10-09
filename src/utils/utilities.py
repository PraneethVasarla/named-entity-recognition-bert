import torch

def get_torch_gpu_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device("cpu")