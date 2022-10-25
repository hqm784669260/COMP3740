import torch
pretrained_weights  = torch.load('D:/detr-r50-e632da11.pth')

#NWPU数据集，10类
num_class = 26    #类别数+1，1为背景
print(print(pretrained_weights["model"].keys()))

pretrained_weights["model"]["class_embed.weight"].resize_(num_class+1, 256)
pretrained_weights["model"]["class_embed.bias"].resize_(num_class+1)
torch.save(pretrained_weights, "D:/detr-r50_%d.pth"%num_class)
