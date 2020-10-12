import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import config
from torchvision import transforms

from model import SiameseNetwork
from utils import SiameseNetworkDataset, imshow

opt = config.args()
folder_dataset_test = torchvision.datasets.ImageFolder(root=opt.testing_dir)
# 定义图像dataset
transform_test = transforms.Compose([transforms.Resize((100, 100)),
                                     transforms.ToTensor()])
siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                             transform=transform_test,
                                             should_invert=False)

# 定义图像dataloader
test_dataloader = DataLoader(siamese_dataset_test,
                             shuffle=True,
                             batch_size=1)

net = SiameseNetwork().cuda()
net.load_state_dict(torch.load('checkpoint/siameseNet49.pth'))
# 生成对比图像
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)

for i in range(10):
    _, x1, label2 = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)
    output1, output2 = net(x0.cuda(), x1.cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
