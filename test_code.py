import numpy as np
import torch
import cv2
import os
import os.path as osp

#
# a = torch.tensor(np.array([[[1, 5], [2, 7]], [[3, 1], [4, 1]]]))
# print((a[:, :, 0] > 1).float())
# print((a[:, :, 0] > 1).float().unsqueeze(2))
# print(a.view(-1, 2))

# a = torch.tensor(np.array([[1, 2, 3],[4, 5, 6]]))
# b = torch.tensor(np.array([[1],[3]]))
# c = torch.tensor(np.array([[6],[6]]))
# seq = (a, b, c)
# image_pred = torch.cat(seq, 1)
# print(image_pred)

# a = [[1, 2], [3, 4]]
# b = torch.FloatTensor(a).repeat(1, 2)
# print(b)

a = cv2.imread("./imgs/1.jpg")
imlist = [osp.join(osp.realpath('.'), "imgs", img).replace('\\', '/') for img in os.listdir("imgs")]
# cv2.imwrite(imlist[0], a)