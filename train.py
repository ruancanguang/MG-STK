import os
import random
import config
import numpy as np

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from torch import nn
from model_co_fi import Hierarchical
from tqdm import tqdm
from data_loder import data_lode
from LabelSmoothing import LabelSmoothingLoss
from torchvision.models import resnet50


torch.backends.cudnn.enabled = False
arg = config.Config.config()

batch_size = arg['batch_size']
nb_epoch = arg['epoch']
device = arg['device']
nb_class = arg['num_class']
lr_begin = arg['learning_rate']
dict_label = arg['class_c_f_dict']
dict_label_first = arg['class_c_dict_first']
num_class_c = arg['num_class_c']
trees = arg['trees']
seed = arg['seed']
datasets_dir = arg['dir']
note = arg['note']
##### Random seed setting
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#######################
##### 1 - Setting #####
exp_dir = 'result/{}{}'.format(datasets_dir, note)  # the folder to save model
train_loader, eval_loader = data_lode()
model = resnet50(pretrained=True)  # to use more models, see https://pytorch.org/vision/stable/models.html
net = Hierarchical(model, num_class_c)
##### optimizer setting
LSLoss = LabelSmoothingLoss(
    classes=nb_class, smoothing=0.1
)  # label smoothing to improve performance
criterion_multi = nn.MultiLabelSoftMarginLoss().to(device)
optimizer = torch.optim.SGD(
    net.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4
)
##### file/folder prepare
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
with open(os.path.join(exp_dir, 'train_log.csv'), 'w+') as file:
    file.write('Epoch, lr, Train_Loss, Train_Acc, Test_Acc_order,Test_Acc_family, Test_Acc_spice, \n')
print('\n===== Using Torch AMP =====')
with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
    file.write('===== Using Torch AMP =====\n')
########################
##### 2 - Training #####
########################
net.cuda()
min_train_loss = float('inf')
max_eval_acc = 0
criterion_multi = nn.MultiLabelSoftMarginLoss().to(device)
con_data_save = torch.ones(size=(251, 256)).to(device)
# criterion = torch.nn.CrossEntropyLoss().to(device)
criterion1 = nn.BCELoss()


def compute_loss(input, target):
    input_soft = torch.softmax(input, dim=-1)
    loss = criterion1(input_soft, target)
    return loss

for epoch in range(nb_epoch):
    print('\n===== Epoch: {} ====='.format(epoch))
    con_data_save_tem = torch.zeros(size=(251, 256)).to(device)
    class_count = torch.zeros(size=(251, 1)).to(device)
    net.train()  # set model to train mode, enable Batch Normalization and Dropout
    lr_now = optimizer.param_groups[0]['lr']
    con_loss = train_loss = train_correct_first = train_correct_c = train_correct = train_total = idx = 0
    train_loss_f = 0.0
    train_loss_c = 0.0
    train_loss_first = 0.0
    with torch.autograd.set_detect_anomaly(True):
        for batch_idx, (inputs, targets, targets_c, targets_first) in enumerate(tqdm(train_loader, ncols=80)):
            idx = batch_idx
            optimizer.zero_grad()  # Sets the gradients to zero
            inputs, targets, targets_c, targets_first, = inputs.cuda(), targets.cuda(), targets_c.cuda(), targets_first.cuda()
            species, family, order = net(inputs)

            species_soft = torch.softmax(species, dim=1)
            family_soft = torch.softmax(family, dim=1)
            order_soft = torch.softmax(order, dim=1)
            _, pred_species = species_soft.max(dim=1)
            _, pred_family = family_soft.max(dim=1)
            _, pred_order = order_soft.max(dim=1)
            loss_species_tem = 0.0
            loss_family_tem = 0.0
            loss_order_tem = 0.0
            order_focal = 0.0
            family_focal = 0.0
            species_focal = 0.0
            add_num = 1e-5
            for order_index in range(len(pred_order)):
                order_focal = 1 - order_soft[order_index][int(targets_first[order_index])]
                if pred_order[order_index] == targets_first[order_index]:
                    order_value_dict = int(pred_order[order_index]) + 1
                    order_dict = dict_label_first[order_value_dict]
                    family_num = order_dict.index(int(targets_c[order_index]+1))
                    order_dict = torch.tensor(order_dict)
                    value_family = family[order_index]
                    init_family = torch.zeros(size=order_dict.size()).to(device)
                    order_dict = order_dict.to(device)
                    order_dict -= 1
                    init_family = family[order_index].gather(0, order_dict)
                    init_family = torch.unsqueeze(init_family, 0)
                    family_num = torch.tensor([family_num]).to(device)
                    init_family_soft = torch.softmax(init_family, dim=1)
                    argmax_family = init_family_soft.argmax()

                    family_focal = 1 - init_family_soft[0][argmax_family]
                    family_focal_norm = family_focal/(family_focal+order_focal + add_num)
                    loss_family_tem = loss_family_tem + torch.exp(family_focal_norm) * LSLoss(init_family, family_num.long())
                    pred_family[order_index] = order_dict[argmax_family]
                else:
                    family_focal = 1 - family_soft[order_index][int(targets_c[order_index])]
                    family_focal_norm = family_focal / (family_focal + order_focal + add_num)
                    loss_family_tem = loss_family_tem + torch.exp(family_focal_norm) * LSLoss(torch.unsqueeze(family[order_index], 0), torch.unsqueeze(targets_c[order_index], dim=0).long())
                order_focal_norm = order_focal / (family_focal + order_focal + add_num)
                loss_order_tem = loss_order_tem + torch.exp(order_focal_norm) * LSLoss(torch.unsqueeze(order[order_index], 0), torch.unsqueeze(targets_first[order_index], dim=0).long())

                if targets[order_index] > 200:
                    # print(targets[order_index])
                    pass
                else:
                    if pred_family[order_index] == targets_c[order_index]:
                        family_value_dict = int(pred_family[order_index]) + 1
                        family_dict = dict_label[family_value_dict]
                        species_num = family_dict.index(int(targets[order_index] + 1))
                        family_dict = torch.tensor(family_dict)
                        value_species = species[order_index]
                        init_species = torch.zeros(size=family_dict.size()).to(device)
                        family_dict = family_dict.to(device)
                        family_dict -= 1
                        init_species = species[order_index].gather(0, family_dict)
                        init_species = torch.unsqueeze(init_species, 0)
                        species_num = torch.tensor([species_num]).to(device)
                        init_species_soft = torch.softmax(init_species, dim=1)
                        argmax_species = init_species_soft.argmax()
                        pred_species[order_index] = family_dict[argmax_species]

                        species_focal = 1 - init_species_soft[0][argmax_species]
                        species_focal_norm = (species_focal / (species_focal + family_focal + add_num) + 0.5) * torch.exp(family_focal_norm)
                        loss_species_tem = loss_species_tem + species_focal_norm * LSLoss(init_species, species_num.long())
                    else:
                        species_focal = 1 - species_soft[order_index][int(targets[order_index])]
                        species_focal_norm = (species_focal / (species_focal + family_focal + add_num) + 0.5) * torch.exp(family_focal_norm)
                        loss_species_tem = loss_species_tem + species_focal_norm*LSLoss(torch.unsqueeze(species[order_index], 0), torch.unsqueeze(targets[order_index], dim=0).long())

            loss_species = loss_species_tem/len(pred_order)
            loss_family = loss_family_tem/len(pred_order)
            loss_order = loss_order_tem/len(pred_order)
            running_loss = loss_species + loss_family + loss_order
            running_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_total += targets.size(0)
            train_correct += pred_species.eq(targets.data).cpu().sum()
            train_correct_c += pred_family.eq(targets_c.data).cpu().sum()
            train_correct_first += pred_order.eq(targets_first.data).cpu().sum()
            train_loss_f += loss_species
            train_loss_c += loss_family
            train_loss_first += loss_order

        train_acc = 100.0 * float(train_correct) / train_total
        train_acc_c = 100.0 * float(train_correct_c) / train_total
        train_correct_first = 100.0 * float(train_correct_first) / train_total

        train_loss_f = train_loss_f / (idx + 1)
        train_loss_c = train_loss_c / (idx + 1)
        train_loss_first = train_loss_first / (idx + 1)

        con_loss = con_loss / (idx + 1)
        print(
            'Train | lr: {:.4f}  | Loss_first: {:.4f} | Loss_f: {:.4f} | Loss_c: {:.4f} | Con_Loss: {:.4f}|Acc_first: {:.3f}%  Acc_c: {:.3f}% ({}/{})|  Acc: {:.3f}% ({}/{})'.format(
                lr_now, train_loss_first, train_loss_f, train_loss_c, con_loss, train_correct_first, train_acc_c,
                train_correct_c, train_total, train_acc, train_correct, train_total
            )
        )

    ##### Evaluating model with test data every epoch
    with torch.no_grad():
        net.eval()  # set model to eval mode, disable Batch Normalization and Dropout
        eval_correct_first = eval_correct_c = eval_correct = eval_total = 0
        for _, (inputs, targets, targets_c, targets_first) in enumerate(tqdm(eval_loader, ncols=80)):
            inputs, targets, targets_c, targets_first = inputs.cuda(), targets.cuda(), targets_c.cuda(), targets_first.cuda()
            species, family, order = net(inputs)

            species_soft = torch.softmax(species, dim=1)
            family_soft = torch.softmax(family, dim=1)
            order_soft = torch.softmax(order, dim=1)
            _, pred_order = order_soft.max(dim=1)
            _, pred_species = species_soft.max(dim=1)
            _, pred_family = family_soft.max(dim=1)
            for order_index in range(len(pred_order)):
                if pred_order[order_index] == targets_first[order_index]:
                    order_value_dict = int(pred_order[order_index]) + 1
                    order_dict = torch.tensor(dict_label_first[order_value_dict])
                    value_family = family_soft[order_index]
                    init_family = torch.zeros(size=order_dict.size())
                    for family_num in range(len(order_dict)):
                        family_value = order_dict[family_num] - 1
                        init_family[family_num] = value_family[family_value]
                    argmax_family = init_family.argmax()
                    pred_family[order_index] = order_dict[argmax_family] - 1

                if pred_family[order_index] == targets_c[order_index]:
                    family_value_dict = int(pred_family[order_index]) + 1
                    family_dict = torch.tensor(dict_label[family_value_dict])
                    value_species = species_soft[order_index]
                    init_species = torch.zeros(size=family_dict.size())
                    for species_num in range(len(family_dict)):
                        species_value = family_dict[species_num] - 1
                        init_species[species_num] = value_species[species_value]
                    argmax_species = init_species.argmax()
                    pred_species[order_index] = family_dict[argmax_species] - 1
            eval_total += targets.size(0)
            eval_correct += pred_species.eq(targets.data).cpu().sum()
            eval_correct_c += pred_family.eq(targets_c.data).cpu().sum()
            eval_correct_first += pred_order.eq(targets_first.data).cpu().sum()
        eval_acc = 100.0 * float(eval_correct) / eval_total
        eval_acc_c = 100.0 * float(eval_correct_c) / eval_total
        eval_acc_first = 100.0 * float(eval_correct_first) / eval_total
        print(
            '| Acc_first: {:.3f}% | Acc_c: {:.3f}% | Acc: {:.3f}% ({}/{})'.format(
                eval_acc_first, eval_acc_c, eval_acc, eval_correct, eval_total
            )
        )

        ##### Logging
        with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
            file.write(
                '{}, {:.4f}, {:.4f}, {:.3f}%, {:.3f}%, {:.3f}%, {:.3f}%\n'.format(
                    epoch, lr_now, train_loss, train_acc, eval_acc_first, eval_acc_c, eval_acc
                )
            )

        ##### save model with highest acc
        if eval_acc > max_eval_acc:
            max_eval_acc = eval_acc
            torch.save(
                net.state_dict(),
                os.path.join(exp_dir, 'max_acc.pth'),
                _use_new_zipfile_serialization=False,
            )
