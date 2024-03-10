import torch
import pickle
import numpy as np
from utils.datalloader import Stg1Dataset
from datetime import datetime
from utils.logger import get_logger
import os
from model.transformer import transformer

def get_log_dir(config):
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    log_dir = os.path.join(current_dir, 'logs', config.dataset, current_time)
    return log_dir 


def normalize_data(data, scalar_type='Standard'):
    scalar = None
    if scalar_type == 'MinMax01':
        scalar = MinMax01Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'MinMax11':
        scalar = MinMax11Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'Standard':
        scalar = StandardScaler(mean=data.mean(), std=data.std())
    else:
        raise ValueError('scalar_type is not supported in data_normalization.')
    return scalar


class MinMax01Scaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)

class MinMax11Scaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean


class HuberLoss(torch.nn.Module):
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        residual = torch.abs(y_pred - y_true)
        delta = self.delta
        
        loss = torch.where(residual < delta, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))
        return loss.mean()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.epochs = args.epochs

        assert self.args.running_mode in ["train", "test", "finetune"]

        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args, args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, f'best_model.pth')
        
        self.logger.info("Running on: {}".format(device))
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

        self.load_data()

        self.model = transformer(args).to(device)
        self.loss_func = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr_init, weight_decay=1e-3)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.1, patience=3)



        if self.args.load_model and self.args.best_path != "None":
            ckp = torch.load(self.args.best_path)
            self.model.load_state_dict(ckp["model"])
            self.optim.load_state_dict(ckp["optimizer"])


    def load_data(self):
        # dataset assertion
        assert self.args.dataset in ["cdata", "udata"]

        if self.args.dataset=="cdata":
            delay = np.load("dataset/cdata/delay.npy")
            self.args.num_nodes = delay.shape[0]
            self.time_slot = delay.shape[1]
            
            od = np.load("dataset/cdata/od_mx.npy")
            adj = np.load("dataset/cdata/dist_mx.npy")
            self.od = torch.tensor(od, device=self.device, dtype=torch.float32)
            self.adj = torch.tensor(adj, device=self.device, dtype=torch.float32)

            train_num = int(self.time_slot * self.args.train_ratio)
            val_num = int(self.time_slot * self.args.valid_ratio)
            train_val = delay[:train_num+val_num]
            self.scaler = normalize_data(train_val[~np.isnan(train_val)], self.args.scalar_type)

            weather = np.load("dataset/cdata/weather_cn.npy")
            weather = np.expand_dims(weather, -1)
            delay = np.concatenate([delay, weather], axis=-1).transpose(1,0, 2)
            delay[np.isnan(delay)] = 0
            self.delay = delay

            time_step = np.arange(self.time_slot).reshape(-1, 1)

            train_dataset = Stg1Dataset(self.args, delay[:train_num], time_step[:train_num])
            val_dataset = Stg1Dataset(self.args, delay[train_num:train_num+val_num], time_step[train_num:train_num+val_num])
            test_dataset = Stg1Dataset(self.args, delay[train_num+val_num:], time_step[train_num+val_num:])

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)


        if self.args.dataset=="udata":
            delay = np.load("dataset/udata/udelay.npy")
            self.args.num_nodes = delay.shape[0]
            self.time_slot = delay.shape[1]
            
            od = np.load("dataset/udata/od_pair.npy")
            adj = np.load("dataset/udata/adj_mx.npy")
            self.od = torch.tensor(od, device=self.device, dtype=torch.float32)
            self.adj = torch.tensor(adj, device=self.device, dtype=torch.float32)

            train_num = int(self.time_slot * self.args.train_ratio)
            val_num = int(self.time_slot * self.args.valid_ratio)
            train_val = delay[:train_num+val_num]
            self.scaler = normalize_data(train_val[~np.isnan(train_val)], self.args.scalar_type)

            weather = np.load("dataset/udata/weather2016_2021.npy")
            weather = np.expand_dims(weather, -1)
            delay = np.concatenate([delay, weather], axis=-1).transpose(1,0, 2)
            delay[np.isnan(delay)] = 0
            self.delay = delay

            time_step = np.arange(self.time_slot).reshape(-1, 1)

            train_dataset = Stg1Dataset(self.args, delay[:train_num], time_step[:train_num])
            val_dataset = Stg1Dataset(self.args, delay[train_num:train_num+val_num], time_step[train_num:train_num+val_num])
            test_dataset = Stg1Dataset(self.args, delay[train_num+val_num:], time_step[train_num+val_num:])

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)

        self.logger.info(f"{self.args.dataset} Dataset Load Finished.")


    def train_epoch(self):
        self.model.train()
        total_loss = []

        for idx, (X, TE, Y) in enumerate(self.train_loader):
            X = X.to(self.device)
            TE = TE.to(self.device)
            Y = Y.to(self.device)

            self.optim.zero_grad()
            pred = self.model(X, [self.od, self.adj], TE)

            loss = self.loss_func(pred, Y[..., :2])
            # loss.backward(retrain_graph=True)
            loss.backward()
            self.optim.step()
            total_loss.append(loss.item())

            if idx%100==0:
                self.logger.info(f"Batch: {idx},  Train Loss: {loss.item()}")
        return np.mean(total_loss)


    def val_epoch(self):
        self.model.eval()
        total_loss = []

        for idx, (X, TE, Y) in enumerate(self.val_loader):
            X = X.to(self.device)
            TE = TE.to(self.device)
            Y = Y.to(self.device)

            self.optim.zero_grad()
            with torch.no_grad():
                pred = self.model(X, [self.od, self.adj], TE)

            loss = self.loss_func(pred, Y[..., :2])
            total_loss.append(loss.item())

            if idx%100==0:
                self.logger.info(f"Batch: {idx},  Valid Loss: {loss.item()}")
        return np.mean(total_loss)

    def train(self):
        best_loss = float('inf')
        not_improved_count = 0

        for es in range(self.epochs):
            train_loss = self.train_epoch()
            # print("Epoch:", es, ", Training loss",  train_loss)
            self.logger.info(f"Epoch: {es}, Train Loss: {train_loss}")

            val_loss = self.val_epoch()
            self.logger.info(f"Epoch: {es}, Valid Loss: {val_loss}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = es
                not_improved_count = 0
                # save the best state

                save_dict = {
                    "epoch": best_epoch, 
                    "model": self.model.state_dict(), 
                    "optimizer": self.optim.state_dict(),
                    "best_loss": best_loss
                }

            if not_improved_count>self.args.early_stop_patience:
                break
            not_improved_count += 1
            self.sched.step(val_loss)
        self.logger.info('**************Current best model saved to {}'.format(self.best_path))
        torch.save(save_dict, self.best_path)


    