import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import config
import math
arg = config.Config.config()
class_c_f_dict = arg['class_c_f_dict']
label_c = np.load("label_c.npy",allow_pickle=True).item()
label_c_first = np.load("label_c_first.npy",allow_pickle=True).item()
class CUB(Dataset):
    def __init__(self, path, train=True, transform=None, target_transform=None):

        self.root = path
        self.is_train = train
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = {}
        self.proportion = 0
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id
        self.class_c_ids = label_c
        self.class_c_first_ids = label_c_first

        self.data_id = []
        if self.is_train:
            class_image = {}
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if int(is_train):
                        self.data_id.append(image_id)
                        if str(self.class_ids[image_id]) not in class_image.keys():
                            class_image[str(self.class_ids[image_id])] = {'index': []}
                            class_image[str(self.class_ids[image_id])]["index"].append(image_id)
                        else:
                            class_image[str(self.class_ids[image_id])]["index"].append(image_id)
                labels = []
                images = []
                for key in class_image.keys():
                    images += class_image[key]['index']
                    labels += [int(key)] * math.ceil(len(class_image[key]['index']) * self.proportion)
                    rest = len(class_image[key]['index']) - math.ceil(len(class_image[key]['index']) * self.proportion)
                    labels += [1000] * rest
                    for key_label in range(len(labels)):
                        image_index = images[key_label]
                        self.class_ids[image_index] = labels[key_label]


        if not self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if not int(is_train):
                        self.data_id.append(image_id)

    def __len__(self):
        return len(self.data_id)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.data_id[index]
        class_id = int(self._get_class_by_id(image_id)) - 1
        class_c_ids = int(self._get_class_by_id_c(image_id))-1
        class_c_ids_first = int(self._get_class_by_id_c_first(image_id))-1
        path = self._get_path_by_id(image_id)
        image = cv2.imread(os.path.join(self.root, 'images', path))
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            class_id = self.target_transform(class_id)
            class_c_ids = self.target_transform(class_c_ids)
            class_c_ids_first = self.target_transform(class_c_ids_first)

        return image, class_id, class_c_ids, class_c_ids_first

    def _get_path_by_id(self, image_id):

        return self.images_path[image_id]

    def _get_class_by_id(self, image_id):

        return self.class_ids[image_id]
    def _get_class_by_id_c(self, image_id):

        return self.class_c_ids[image_id]
    def _get_class_by_id_c_first(self, image_id):

        return self.class_c_first_ids[image_id]
