import numpy as np
import torch


def type_check_numpy(arr) -> np.ndarray:
    if isinstance(arr, list) or isinstance(arr, tuple):
        return np.array(arr)

    elif isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    
    elif isinstance(arr, int) or isinstance(arr, float):
        Warning(f"@type_check_numpy, the input is a scalar, it is not converted to np.ndarray")
        return arr

    else:
        raise ValueError(f"{type(arr)} is an unknown type")


def type_check_torch_tensor(tensor, strict: bool) -> torch.Tensor:
    if strict:
        assert isinstance(tensor, torch.Tensor)
        return 
    
    else:
        tensor = torch.from_numpy(type_check_numpy(tensor))
        return tensor
    