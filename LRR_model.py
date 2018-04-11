from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torchvision.utils as vutils
import torch.nn.functional as F

class LRR32s(nn.Module):
    def __init__(self):
        super(LRR32s, self).__init__()
        self.learned_billinear = False
        self.n_classes = 21

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096 ,1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes*10, 5,padding = 2),
            nn.ConvTranspose2d(self.n_classes*10, self.n_classes, 8, stride= 8, padding= 4,  groups= self.n_classes),)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        score = self.classifier(conv5)
        return score


    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        #Stores each layer specs into a list entry
        features = list(vgg16.features.children())    
        # idx - stores the index of the sequential blocks
        # conv_block - sequential model inside a block
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    #print l1.weight.data.size()
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        #print(self.classifier[7].weight.data.size())
        DeconvW = np.load('FiltW.npy')
        DeconvW = torch.from_numpy(DeconvW).float()
        DeconvW = DeconvW.permute(3,2,1,0)
        self.classifier[7].weight.data = DeconvW 
        
        
class LRR(nn.Module):
    def __init__(self):
        super(LRR, self).__init__()
        self.n_classes = 21
        model_32s = LRR32s();
        #print(model_32s)
        model_32s = torch.nn.DataParallel(model_32s, device_ids=range(torch.cuda.device_count()))
        
        checkpoint = torch.load('Models/LRR_8s_pascal_best_model.pkl');
        
        model_32s.load_state_dict(checkpoint['model_state'],True);
        
        self.conv_block1 = nn.Sequential(*list(model_32s.module.conv_block1))

        self.conv_block2 = nn.Sequential(*list(model_32s.module.conv_block2))

        self.conv_block3 = nn.Sequential(*list(model_32s.module.conv_block3))
        
        self.conv_block4 = nn.Sequential(*list(model_32s.module.conv_block4))
        
        self.conv_block5 = nn.Sequential(*list(model_32s.module.conv_block5))
        
        self.classifier8x = nn.Sequential(*list(model_32s.module.classifier))
        
        self.classifier4x = nn.Sequential(
            nn.Conv2d(256, self.n_classes*10, 5,padding = 2),
            nn.ConvTranspose2d(self.n_classes*10, self.n_classes, 8, stride= 2,  groups= self.n_classes),)
        
        self.classifier2x = nn.Sequential(
            nn.Conv2d(128, self.n_classes*10,7,padding = 5),
            nn.ConvTranspose2d(self.n_classes*10, self.n_classes, 8, stride= 2, padding = 1 ,groups= self.n_classes),)
        
        
    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        output8x = self.classifier8x(conv5)
        
        Input_from8x_4x = F.upsample(output8x, scale_factor= 2, mode='bilinear');  ## 2x upsampling
        Soft_Input_from8x_4x = F.softmax(Input_from8x_4x,1); ##Softmax       
        ##Masking = max_pool(Nsoftmax) + max_pool(softmax) 
        N_Max_Soft_Input_from8x_4x =  F.max_pool2d(-1*Soft_Input_from8x_4x,5, stride=1, padding=2, dilation=1);
        Max_Soft_Input_from8x_4x =  F.max_pool2d(Soft_Input_from8x_4x,5, stride=1, padding=2, dilation=1);
        Mask_out4x = N_Max_Soft_Input_from8x_4x + Max_Soft_Input_from8x_4x;
        ## Element Wise multiplication
        conv_output4x = self.classifier4x(conv3);
        output4x = torch.mul(conv_output4x, Mask_out4x);
        output4x = Input_from8x_4x + output4x;
        
        Input_from4x_2x = F.upsample(output4x, scale_factor = 2, mode = 'bilinear');
        Soft_Input_from4x_2x = F.softmax(Input_from4x_2x,1); ##Softmax
        ##Masking = max_pool(Nsoftmax) + max_pool(softmax) 
        N_Max_Soft_Input_from4x_2x =  F.max_pool2d(-1*Soft_Input_from4x_2x,3, stride=1, padding=1, dilation=1);
        Max_Soft_Input_from4x_2x =  F.max_pool2d(Soft_Input_from4x_2x,3, stride=1, padding=1, dilation=1);
        Mask_out2x = N_Max_Soft_Input_from4x_2x + Max_Soft_Input_from4x_2x;
        ## Element Wise multiplication
        conv_output2x = self.classifier2x(conv2);
        output2x = torch.mul(conv_output2x, Mask_out2x);
        output2x = Input_from4x_2x + output2x;
        ##        
        return output8x,output4x,output2x
    def init_decov_2x_4x_params(self):
        DeconvW = np.load('FiltW.npy')
        DeconvW = torch.from_numpy(DeconvW).float()
        DeconvW = DeconvW.permute(3,2,1,0)
        self.classifier4x[1].weight.data = DeconvW
        self.classifier2x[1].weight.data = DeconvW