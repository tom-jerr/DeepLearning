import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os

model_path_1= 'models/discriminatorConv.pth'
model_path_2= 'models/generatorConv.pth'
dis_model = torch.load(model_path_1)
gen_model = torch.load(model_path_2)


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)   #将输入变成（min,max）范围内的值
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 64
num_epoch = 100
z_dimension = 100  # noise dimension

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
mnist = datasets.MNIST('./data', transform=img_transform,download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=False)

# dis_model.eval()
gen_model.eval()
total_loss, total_acc=0,0
with torch.no_grad:
    for i,(input,labels) in enumerate(dataloader):
        pred_fake=gen_model(input.cuda())
        criterion = nn.BCELoss()
        loss = criterion(pred_fake.cpu(),labels)
        acc = (pred_fake.argmax(1)==labels).sum()
        total_loss+=loss.items()
        total_acc+=acc
print('loss:{}  accuracy:{}'.format(total_loss,total_acc))




