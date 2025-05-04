import time
import torch
import argparse
import os
import numpy as np
import random
from torch import nn
from torch.utils.data import DataLoader
from val_data_functions import ValData
from utils import validation, validation_val
from transweather_model import Transweather

parser = argparse.ArgumentParser(description='Test on AllWeather')
parser.add_argument('--val_batch_size', default=1, type=int)
parser.add_argument('--exp_name', type=str, required=True, help='folder under ./weights/ where best model is saved')
parser.add_argument('--seed', default=19, type=int)
args = parser.parse_args()

# Set seed
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
print(f"Seed: {seed}")

# Set device
device_ids = [i for i in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load validation data
val_data_dir = './data/train/AllWeather/'
val_filename = 'allweather.txt'
val_data_loader = DataLoader(
    ValData(val_data_dir, val_filename),
    batch_size=args.val_batch_size,
    shuffle=False,
    num_workers=8
)

# Load model
net = Transweather().cuda()
net = nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(torch.load('./weights/Transweather/Transweather_ep212_checkpoint.pth'))

net.eval()
category = "AllWeather"
os.makedirs(f'./results/{category}/{args.exp_name}', exist_ok=True)

print("===== Testing Starts! =====")
start_time = time.time()
val_psnr, val_ssim = validation_val(val_data_loader=val_data_loader,
                                net=net,
                                device=device,
                                exp_name=args.exp_name,
                                category=category,
                                save_tag=True)
end_time = time.time() - start_time

print(f'val_psnr: {val_psnr:.2f}, val_ssim: {val_ssim:.4f}')
print(f'Validation time is {end_time:.4f}')