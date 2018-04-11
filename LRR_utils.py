import os
import numpy as np
import scipy
import torch 
def label_downscale(labels, d_scale, batch_size_voc, img_row):

    tlabels = labels.numpy()
    tlabels = tlabels.astype(int)        
    labels = np.zeros((batch_size_voc,int(img_row/d_scale),int(img_row/d_scale)), dtype=int)
    for img_n in range(batch_size_voc):
            ttlabels = scipy.misc.imresize(tlabels[img_n,:,:],(int(img_row/d_scale),int(img_row/d_scale)), 'nearest',mode='F')
            labels[img_n,:,:] = ttlabels;
    labels = torch.from_numpy(labels).long()
    return labels

