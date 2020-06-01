import torch
import torch.nn as nn

class Features_Loss(nn.Module):
    def __init__(self):
        super(Features_Loss,self).__init__()

    def forward(self, features, batch_size, person_num):
        loss_all = 0
        loss_temp = 0
        for id_index in range( batch_size // person_num ):

            features_temp_list = features[id_index*person_num:(id_index+1)*person_num+1]
            loss_temp = 0
            distance = torch.mm(features_temp_list,features_temp_list.t())
            distance = 1 -distance

            for i in range(person_num):
                for j in range( i + 1, person_num ):
                    loss_temp = loss_temp + distance[i][j]
            loss_temp = loss_temp / ( person_num * (person_num-1)/2 )
            loss_all = loss_all +  loss_temp
        loss_all = loss_all / ( batch_size // person_num )
        return loss_all



