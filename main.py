import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from model import SiameseNetwork, ContrastiveLoss, train
from utils import SiameseNetworkDataset

opt = config.args()
folder_dataset = torchvision.datasets.ImageFolder(root=opt.training_dir)

# 定义图像dataset
transform = transforms.Compose([transforms.Resize((100, 100)),  # 有坑，传入int和tuple有区别
                                transforms.ToTensor()])
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transform,
                                        should_invert=False)

# 定义图像dataloader
train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              batch_size=opt.batch_size)

net = SiameseNetwork().cuda() #定义模型且移至GPU
print(net)
criterion = ContrastiveLoss(margin=2.0) #定义损失函数
optimizer = optim.Adam(net.parameters(), lr=opt.lr) #定义优化器

train(net, optimizer, criterion, train_dataloader, opt)
