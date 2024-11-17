import torch
import numpy as np


class Metrics():
    def __init__(self):
        self.rmse_list = []
        self.absrel_list = []

    def RMSE_transit(self, depth, gt):
        self.rmse_list.append((depth - gt) ** 2)
    
    def RMSE_total(self):
        return np.mean(self.rmse_list) ** 0.5
    
    def AbsRel_transit(self, depth, gt):
        self.absrel_list.append((torch.abs(depth - gt) / gt))
    
    def AbsRel_total(self):
        return np.mean(self.absrel_list)


if __name__ == "__main__":
    metrics = Metrics()
    depth = torch.tensor([1, 2, 3, 4])
    gt = torch.tensor([1, 2, 3, 5])

    metrics.RMSE_transit(depth, gt)
    metrics.AbsRel_transit(depth, gt)

    print(metrics.RMSE_total())
    print(metrics.AbsRel_total())

