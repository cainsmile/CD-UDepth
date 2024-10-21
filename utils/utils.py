"""
Functions of average meter, logger, guidedfilter, and result output
"""
import logging
#from cv2.ximgproc import guidedFilter
import numpy as np
import torch

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x, device='cpu'):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return x * std + mean


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}

class AverageMeter(object):
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

 
def get_logger(filename, verbosity=1, name=None):
    """Get logger"""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


def gf(mask, img):
    """Apply guidedfilter"""
    return guidedFilter(mask, img, 4, 0.2, -1)

def output_result(out, mask):
    """Prepare data and apply guidedfilter to depth map"""
    mask = np.float32(mask)
    img=out.detach().cpu().numpy()
    img_resized=img.reshape(240,320)
    max_item = max(max(row) for row in img_resized)
    img_resized = img_resized / max_item * 255

    result = gf(mask, img_resized)
    
    return result

def calculate_relation(tensor1, tensor2):
    # Calculate RMSE
    rmse = 1-torch.abs(tensor1 - tensor2)

    # Normalize to 0-1
    min_values = torch.min(torch.min(rmse, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]  # [B, 1, 1, 1]
    max_values = torch.max(torch.max(rmse, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]  # [B, 1, 1, 1]
    rmse = (rmse - min_values) / (max_values - min_values)

    return rmse

def generate_soft_labels(depth1, depth2, ground_truth):
    soft_labels = torch.zeros_like(depth1)
    
    depth1_error = torch.abs(depth1 - ground_truth)
    depth2_error = torch.abs(depth2 - ground_truth)

    soft_labels[depth1_error < depth2_error] = 1
    
    return soft_labels
