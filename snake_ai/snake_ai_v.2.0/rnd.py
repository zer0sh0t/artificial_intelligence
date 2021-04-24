import torch
import time
from torch import nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from grab_screen import screenshot

s = time.time()
for _ in range(1):
	# _ = 10 + 20
	# _ = 100 + 20
	# _ = 50 + 20
	# _ = 10 % 20
	# _ = 10 * 20
	# _ = 10 + 20
	# _ = 10 + 200
	# _ = 10 / 20
	# _ = 10 - 20
	im = screenshot("Command Prompt - python  rnd.py") # super slow
print(time.time() - s)
plt.imshow(im)
plt.show()

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, 4)
#         self.conv2 = nn.Conv2d(32, 64, 8)
#         self.fc1 = nn.Linear(64*214*214, 32)
#         self.fc2 = nn.Linear(32, 3)

#     def forward(self, x):
#     	x = F.relu(self.conv1(x))
#     	x = F.relu(self.conv2(x))
#     	x = x.view(x.shape[0], -1)
#     	x = F.relu(self.fc1(x))
#     	x = self.fc2(x)
#     	return x

model = models.resnet18(pretrained=True)
model.fc = nn.Identity()
print(model)

# class Net(nn.Module):
# 	def __init__(self):
# 		super().__init__()
# 		self.conv1 = nn.Conv2d(3, 32, 4)
# 		self.conv2 = nn.Conv2d(32, 64, 4)

# 	def forward(self, x):
# 		x = self.conv1(x)
# 		x = self.conv2(x)
# 		return x

# model = Net()
x = torch.randn(1, 3, 224, 224)
print(model(x).shape)