import torch
import random

def crop(data, cropsize):
    assert (cropsize <= data.shape[-1] and cropsize <= data.shape[-2])
    w1 = random.randint(0, data.shape[-2] - cropsize)
    h1 = random.randint(0, data.shape[-1] - cropsize)
    w2 = w1 + cropsize
    h2 = h1 + cropsize
    data = data[:, :, w1:w2, h1:h2]
    return data


def pre_process(data, DR,low=False):
    DR = int(DR)
    data = (20 * torch.log10(data / (torch.max(data))))
    data = torch.clip(data, -DR, 0)
    min = torch.min(data)
    max = torch.max(data)
    data = (data - min) / (max - min)
    if low and DR<=40:
        data = torch.pow(data, 2/3)
    return data