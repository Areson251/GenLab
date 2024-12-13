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
        safe_gt = np.where(gt == 0, np.nan, gt)
        self.absrel_list.append(abs(depth - gt) / safe_gt)
    
    def AbsRel_total(self):
        all_values = np.concatenate(self.absrel_list)  
        valid_values = all_values[~np.isnan(all_values)]  
        if valid_values.size == 0:
            return 0  
        return np.mean(valid_values)


if __name__ == "__main__":
    metrics = Metrics()
    depth = torch.tensor([1, 2, 3, 4])
    gt = torch.tensor([1, 2, 3, 5])

    metrics.RMSE_transit(depth, gt)
    metrics.AbsRel_transit(depth, gt)

    print(metrics.RMSE_total())
    print(metrics.AbsRel_total())

