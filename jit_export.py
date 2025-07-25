import numpy as np
import os
import torch

from modules.xfeat import XFeat
from modules.model import *
os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

xfeat = XFeatModel()
stastic_pt_path = "xfeat.pt"
xfeat.load_state_dict(torch.load("./weights/xfeat.pt", map_location='cpu'))

torch.jit.script(xfeat).save(stastic_pt_path)

print(f"Model saved to {stastic_pt_path}")

# 1. 读取 TorchScript 模型
device = torch.device("cpu")           # 如想 GPU，可改为 "cuda"
model = torch.jit.load("xfeat.pt", map_location=device)
model.eval()

B, C, H, W = 1, 3, 480, 640
img0 = torch.rand(B, C, H, W, device=device)

with torch.inference_mode():
    # detectAndCompute 分支
    output = model(img0)

print("img0 -> kpts:{output[1].shape}")
print("img0 -> feats:{output[0].shape}")
print("img0 -> heatmap:{output[2].shape}")
print("TorchScript 模型 xfeat.pt 测试通过！")