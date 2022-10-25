import os

import cv2
import numpy as np
import torch

from net import *
from utils import *
from data import *
from torchvision.utils import save_image
from PIL import Image
net=UNet(3).cuda() # 3 类别数

weights='params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')

_input=input('please input JPEGImages path:')

img=keep_image_size_open_rgb(_input)
# print(img.shape)
img_data=transform(img).cuda() #转换成张量
# print(img_data.shape)
img_data=torch.unsqueeze(img_data,dim=0) #升维
# print(img_data.shape)

# net.eval()
out=net(img_data)
save_image(out,'result/result.jpg')
# out=torch.argmax(out,dim=1)
# out=torch.squeeze(out,dim=0)
# out=out.unsqueeze(dim=0)
# print(set((out).reshape(-1).tolist()))
# out=(out).permute((1,2,0)).cpu().detach().numpy()
# cv2.imwrite('result/result.png',out)
# cv2.imshow('out',out*255.0)
# cv2.waitKey(0)

