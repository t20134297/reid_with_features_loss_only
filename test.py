import torch
from triplet_sperate import *
from resnet import *
from loss import *
from scipy.spatial.distance import cdist

batch_size = 32
dataset = Market1501('./')
t = build_transforms(is_train = False)

gallery_dataset = ImageDataset(dataset.gallery,t)
gallery_loader = DataLoader(gallery_dataset, batch_size = batch_size,shuffle=True)
query_dataset = ImageDataset(dataset.query,t)
query_loader =  DataLoader(query_dataset, batch_size = batch_size,shuffle=True)



def extract_features(loader, model):
    batch_long = len(loader)
    batch_done = 0

    features = []
    labels = []
    for data in loader:
        batch_done = batch_done + 1
        print('done / all = {} / {} '.format(batch_done, batch_long))
        imgs, label, _, _ = data
        #num = num + imgs.shape[0]
        # if num > 6:
        #     break
        #imgs = imgs.cuda()
        with torch.no_grad():
            temp_features = model(imgs)[0]
            features.append(temp_features)
            labels.append(label)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim = 0)
    return features, labels

#single gpu test
net = resnet50()
pre_dict = torch.load('./epoch60.pth')
new_pre = {}
for k,v in pre_dict.items():
    name = k[7:]
    new_pre[name] = v

# net = nn.DataParallel(resnet50())
# net.load_state_dict(torch.load('./epoch1.pth'))
#net = nn.DataParallel(net)
net.eval()
#net.cuda()

gallery_features, gallery_labels = extract_features(gallery_loader, net)
print('gallery extract finish')
query_features, query_labels = extract_features(query_loader, net)
print('query extract finish')

gallery_labels = np.asarray( gallery_labels.detach() )
query_labels = np.asarray( query_labels.detach() )


gallery_features = np.asarray(gallery_features.detach())
query_features = np.asarray(query_features.detach())

gallery_num = gallery_features.shape[0]
query_num = query_features.shape[0]


#distance = cdist( gallery_features, query_features )


correct_num = 0
for i in range(query_num):
    distance = cdist( gallery_features, query_features[i:i+1],metric='cosine' )
    distance = distance[:,0]
    ranking_list = np.argsort(distance)
    if query_labels[i] == gallery_labels[ranking_list[0]]:
        correct_num = correct_num + 1
    print('{} /  {} '.format(correct_num, i))

rank_1 = correct_num*1.0/query_num
print('rank1_correct accuracy: {}' .format(rank_1))





#
# for data in train_loader:
#     imgs, pid, camid, img_path = data
#     print(imgs.shape)
#     print(pid)
#
#     break
