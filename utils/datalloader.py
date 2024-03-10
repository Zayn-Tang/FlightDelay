import torch
import numpy as np


class Stg1Dataset(torch.utils.data.Dataset):
    def __init__(self, args, delay, time_slot):
        if isinstance(delay, np.ndarray):
            delay = torch.from_numpy(delay)
        if isinstance(time_slot, np.ndarray):
            time_slot = torch.from_numpy(time_slot)
        self.delay = delay.float()
        self.time_slot = time_slot.float()
        self.hist = args.num_hist
        self.pred = args.num_pred

    def __len__(self):
        return self.delay.shape[0] - self.hist - self.pred
    
    def __getitem__(self, ts):
        return self.delay[ts:ts+self.hist], self.time_slot[ts:ts+self.hist+self.pred] ,self.delay[ts+self.hist:ts+self.hist+self.pred]


