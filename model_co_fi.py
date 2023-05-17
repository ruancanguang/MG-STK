from torch.nn import Parameter
import torch.nn.functional as F
import torch
import config
from torch import nn
from torchvision import utils, transforms
import matplotlib.pyplot as plt
import math
arg = config.Config.config()
device = arg['device']
trees = arg['trees']

def w_h(gs_species, wscore, hscore, inputs, b, h, w,threshold):
    linputs = torch.zeros([b, 3, h, w]).cuda()
    for i in range(b):
        # topN for MCAR method
        gs_inv, gs_ind = gs_species[i].sort(descending=True)
        xs = wscore[i, gs_ind[0], :].squeeze()
        ys = hscore[i, gs_ind[0], :].squeeze()
        if xs.max() == xs.min():
            xs = xs / xs.max()
        else:
            xs = (xs - xs.min()) / (xs.max() - xs.min())
        if ys.max() == ys.min():
            ys = ys / ys.max()
        else:
            ys = (ys - ys.min()) / (ys.max() - ys.min())
        x1, x2 = obj_loc(xs, threshold)
        y1, y2 = obj_loc(ys, threshold)
        linputs[i:i + 1] = F.interpolate(inputs[i:i + 1,:, y1:y2, x1:x2], size=(h, w), mode='bilinear', align_corners=True)
    return linputs

def obj_loc(score, threshold):
    smax, sdis, sdim = 0, 0, score.size(0)
    minsize = int(math.ceil(sdim * 0.125))  #0.125
    snorm = (score - threshold).sign()
    snormdiff = (snorm[1:] - snorm[:-1]).abs()

    szero = (snormdiff==2).nonzero()
    if len(szero)==0:
       zmin, zmax = int(math.ceil(sdim*0.125)), int(math.ceil(sdim*0.875))
       return zmin, zmax

    if szero[0] > 0:
       lzmin, lzmax = 0, szero[0].item()
       lzdis = lzmax - lzmin
       lsmax, _ = score[lzmin:lzmax].max(0)
       if lsmax > smax:
          smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
       if lsmax == smax:
          if lzdis > sdis:
             smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if szero[-1] < sdim:
       lzmin, lzmax = szero[-1].item(), sdim
       lzdis = lzmax - lzmin
       lsmax, _ = score[lzmin:lzmax].max(0)
       if lsmax > smax:
          smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
       if lsmax == smax:
          if lzdis > sdis:
             smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if len(szero) >= 2:
       for i in range(len(szero)-1):
           lzmin, lzmax = szero[i].item(), szero[i+1].item()
           lzdis = lzmax - lzmin
           lsmax, _ = score[lzmin:lzmax].max(0)
           if lsmax > smax:
              smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
           if lsmax == smax:
              if lzdis > sdis:
                 smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if zmax - zmin <= minsize:
        pad = minsize-(zmax-zmin)
        if zmin > int(math.ceil(pad/2.0)) and sdim - zmax > pad:
            zmin = zmin - int(math.ceil(pad/2.0)) + 1
            zmax = zmax + int(math.ceil(pad/2.0))
        if zmin < int(math.ceil(pad/2.0)):
            zmin = 0
            zmax =  minsize
        if sdim - zmax < int(math.ceil(pad/2.0)):
            zmin = sdim - minsize + 1
            zmax = sdim

    return zmin, zmax

class Hierarchical(nn.Module):
    def __init__(self,model, num_c_cls):
        super(Hierarchical, self).__init__()
        self.model = model
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.fc_class_first = nn.Linear(256, 13)
        self.dict_label = arg['class_c_f_dict']
        self.dict_label_first = arg['class_c_dict_first']
        self.relu = nn.LeakyReLU(0.2)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        self.num_classes_species = 200
        self.num_classes_family = 38
        self.num_classes_order = 13
        self.num_features = 2048
        self.threshold = 0.5
        self.convclass_species = nn.Conv2d(self.num_features, self.num_classes_species, kernel_size=1, stride=1, padding=0, bias=True)
        self.convclass_species1 = nn.Conv2d(self.num_features, self.num_classes_species, kernel_size=1, stride=1, padding=0, bias=True)
        self.convclass_family = nn.Conv2d(self.num_features, self.num_classes_family, kernel_size=1, stride=1, padding=0, bias=True)
        self.convclass_family1 = nn.Conv2d(self.num_features, self.num_classes_family, kernel_size=1, stride=1,padding=0, bias=True)
        self.convclass_order = nn.Conv2d(self.num_features, self.num_classes_order, kernel_size=1, stride=1, padding=0, bias=True)
        self.convclass_order1 = nn.Conv2d(self.num_features, self.num_classes_order, kernel_size=1, stride=1, padding=0, bias=True)
        # self.threshold_species = 0.3
        # self.threshold_family = 0.5
        # self.threshold_order = 0.7
        """上部分"""

    def forward(self, inputs):
        # std = (0.229, 0.224, 0.225)
        # mean = (0.485, 0.456, 0.406)
        # std = torch.tensor(std)
        # mean = torch.tensor(mean)
        # mean = - mean / std
        # std = 1. / std
        # image = transforms.Normalize(mean, std)(inputs)
        # image = transforms.ToPILImage()(image[0])
        # plt.imshow(image)
        # plt.show()

        b, c, h, w = inputs.size()
        feature_no_pooling = feature = self.features(inputs)
        feature = self.pooling(feature)

        """gf_order"""
        gf_order = self.convclass_order(feature)
        feature_order = torch.squeeze(gf_order)
        # from global to local
        camscore_order = self.convclass_order(feature_no_pooling.detach())
        camscore_order = torch.sigmoid(camscore_order)
        camscore_order = F.interpolate(camscore_order, size=(h, w), mode='bilinear', align_corners=True)
        wscore_order = F.max_pool2d(camscore_order, (h, 1)).squeeze(dim=2)
        hscore_order = F.max_pool2d(camscore_order, (1, w)).squeeze(dim=3)
        linputs_order = w_h(feature_order, wscore_order, hscore_order, inputs, b, h, w, self.threshold)
        la_order = self.features(linputs_order.detach())
        lf_order = self.pooling(la_order)

        ls_order = self.convclass_order1(lf_order)
        ls_order = torch.squeeze(ls_order)
        feature_order = (feature_order + ls_order) / 2

        """gf_family"""
        gf_family = self.convclass_family(feature)
        feature_family = torch.squeeze(gf_family)
        # from global to local
        camscore_family = self.convclass_family(feature_no_pooling.detach())
        camscore_family = torch.sigmoid(camscore_family)
        camscore_family = F.interpolate(camscore_family, size=(h, w), mode='bilinear', align_corners=True)
        wscore_family = F.max_pool2d(camscore_family, (h, 1)).squeeze(dim=2)
        hscore_family = F.max_pool2d(camscore_family, (1, w)).squeeze(dim=3)
        linputs_family = w_h(feature_family, wscore_family, hscore_family, inputs, b, h, w, self.threshold)

        la_family = self.features(linputs_family.detach())
        lf_family = self.pooling(la_family)

        ls_family = self.convclass_family1(lf_family) + self.convclass_family1(lf_order)
        ls_family = torch.squeeze(ls_family)
        feature_family = (feature_family + ls_family) / 2


        """gf_species"""
        gf_species = self.convclass_species(feature)
        feature_species = torch.squeeze(gf_species)
        # from global to local
        camscore_species = self.convclass_species(feature_no_pooling.detach())
        camscore_species = torch.sigmoid(camscore_species)
        camscore_species = F.interpolate(camscore_species, size=(h, w), mode='bilinear', align_corners=True)
        wscore_species = F.max_pool2d(camscore_species, (h, 1)).squeeze(dim=2)
        hscore_species = F.max_pool2d(camscore_species, (1, w)).squeeze(dim=3)
        linputs_species = w_h(feature_species, wscore_species, hscore_species, inputs, b, h, w, self.threshold)
        la_species = self.features(linputs_species.detach())
        lf_species = self.pooling(la_species)
        # lf_species = torch.cat((lf_species, lf_family), dim=1)
        ls_species = self.convclass_species1(lf_species) + self.convclass_species1(lf_family)
        ls_species = torch.squeeze(ls_species)
        feature_species = (feature_species + ls_species)/2

        # for i in range(200):
        #     value_species = trees[i][0] - 1
        #     value_family = trees[i][2] - 1
        #     value_order = trees[i][1] - 1
        #     feature_species[:, value_species] = feature_family[:, value_family] + feature_order[:, value_order] + feature_species[:, value_species]


        return  feature_species, feature_family, feature_order


