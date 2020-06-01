import torch
from triplet_sperate import *
from resnet import *
from loss import *
import torch.optim as optim

from torch.optim.lr_scheduler import MultiStepLR


net = resnet50()
net_dict = net.state_dict()
pretrain_dict = torch.load('./model/resnet50.pth')
feed_dict = {k:v for k, v in pretrain_dict.items() if k in net_dict}
net_dict.update(feed_dict)
net.load_state_dict(net_dict)
net = nn.DataParallel(net).cuda()
net.train()


batch_size = 160
person_num = 4
dataset = Market1501('./')
t = build_transforms()
img_dataset = ImageDataset(dataset.train,t)
train_loader = DataLoader(img_dataset, batch_size = batch_size, sampler=RandomIdentiyiSampler(dataset.train,batch_size, person_num))


optimizer = optim.SGD( net.parameters(), lr = 0.0005, momentum=0.9 )
lr_scheduler = MultiStepLR(optimizer, milestones=[20,40])
loss_function = Features_Loss()

all_epoch = 60
for epoch in range(all_epoch):
    print("{}/{} epoch " .format(epoch+1, all_epoch))
    for data in train_loader:
        imgs, pid, camid, img_path = data
        if (imgs.shape[0] < batch_size):
            continue
        imgs = imgs.cuda()
        features, _, _ = net(imgs)

        loss = loss_function( features, batch_size, person_num )
        lr_scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss :', loss.item())

torch.save(net.cpu().state_dict(),'epoch60.pth')
print('successfully save model')






