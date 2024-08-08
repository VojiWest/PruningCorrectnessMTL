import torch
import numpy as np
from typing import List, Callable

import numpy as np
import cv2
import random

import matplotlib.pyplot as plt


class PerturbationConfidenceMetric:
    def __init__(self, perturbation):
        self.perturbation = perturbation

    def __call__(self, input_tensor: torch.Tensor,
                 cams: np.ndarray,
                 targets: List[Callable],
                 model: torch.nn.Module,
                 return_visualization=False,
                 return_diff=True):

        if return_diff:
            with torch.no_grad():
                outputs = model(input_tensor)
                scores = [target(output).cpu().numpy()
                          for target, output in zip(targets, outputs)]
                scores = np.float32(scores)

        batch_size = input_tensor.size(0)
        perturbated_tensors = []
        for i in range(batch_size):
            if cams.ndim > 2:
                cam = cams[i]
            else:
                cam = cams
            tensor = self.perturbation(input_tensor[i, ...].cpu(),
                                       torch.from_numpy(cam))  
            tensor = tensor.to(input_tensor.device)
            perturbated_tensors.append(tensor.unsqueeze(0))
        perturbated_tensors = torch.cat(perturbated_tensors)

        # with torch.no_grad():
        #     outputs_after_imputation = model(perturbated_tensors)
        # scores_after_imputation = [
        #     target(output).cpu().numpy() for target, output in zip(
        #         targets, outputs_after_imputation)]
        # scores_after_imputation = np.float32(scores_after_imputation)

        # if return_diff:
        #     result = scores_after_imputation - scores
        # else:
        #     result = scores_after_imputation

        # if return_visualization:
        #     return result, perturbated_tensors
        # else:
        #     return result

        if return_visualization:
            return perturbated_tensors


class RemoveMostRelevantFirst:
    def __init__(self, percentile, imputer):
        self.percentile = percentile
        self.imputer = imputer

    def __call__(self, input_tensor, mask):
        imputer = self.imputer
        if self.percentile != 'auto':
            mask_non_zero = mask[mask != 0]
            # laplace_mask = mask
            # for i in range(1, mask.shape[0] - 1):
            #     for j in range(1, mask.shape[1] - 1):
            #         if mask[i, j] == 0:
            #             laplace_mask[i, j] = random.uniform(0.00001, 0.001)
            # threshold = np.percentile(mask.cpu().numpy(), self.percentile) # Changed from global percentile to non-zero local
            threshold = np.percentile(mask_non_zero.cpu().numpy(), self.percentile)
            # threshold = np.percentile(laplace_mask.cpu().numpy(), self.percentile)
            # print("Threshold in RMRF:", threshold, "Max:", np.max(mask.cpu().numpy()), "Percentile:", self.percentile)
            binary_mask = np.float32(mask < threshold)
            # binary_mask = np.float32(laplace_mask < threshold)
        else:
            _, binary_mask = cv2.threshold(
                np.uint8(mask * 255), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary_mask = torch.from_numpy(binary_mask)
        binary_mask = binary_mask.to(mask.device)
        return imputer(input_tensor, binary_mask)


class RemoveLeastRelevantFirst(RemoveMostRelevantFirst):
    def __init__(self, percentile, imputer):
        super(RemoveLeastRelevantFirst, self).__init__(percentile, imputer)

    def __call__(self, input_tensor, mask):
        return super(RemoveLeastRelevantFirst, self).__call__(
            input_tensor, 1 - mask)


class AveragerAcrossThresholds:
    def __init__(
        self,
        imputer,
        percentiles=[
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90]):
        self.imputer = imputer
        self.percentiles = percentiles

    def __call__(self,
                 input_tensor: torch.Tensor,
                 cams: np.ndarray,
                 targets: List[Callable],
                 model: torch.nn.Module):
        scores = []
        for percentile in self.percentiles:
            imputer = self.imputer(percentile)
            scores.append(imputer(input_tensor, cams, targets, model))
        return np.mean(np.float32(scores), axis=0)
