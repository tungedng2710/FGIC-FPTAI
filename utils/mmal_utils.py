import torch
from skimage import measure
import numpy as np
import os

from collections import OrderedDict
import cv2
from config import proposalN, init_lr

def auto_load_resume(model, path, status):
    if status == 'train':
        pth_files = os.listdir(path)
        nums_epoch = [int(name.replace('epoch', '').replace('.pth', '')) for name in pth_files if '.pth' in name]
        if len(nums_epoch) == 0:
            return 0, init_lr
        else:
            max_epoch = max(nums_epoch)
            pth_path = os.path.join(path, 'epoch' + str(max_epoch) + '.pth')
            print('Load model from', pth_path)
            checkpoint = torch.load(pth_path)
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            epoch = checkpoint['epoch']
            lr = checkpoint['learning_rate']
            print('Resume from %s' % pth_path)
            return epoch, lr
    elif status == 'test':
        print('Load model from', path)
        checkpoint = torch.load(path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if 'module.' == k[:7]:
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        epoch = checkpoint['epoch']
        print('Resume from %s' % path)
        return epoch

def calculate_iou(coor1, coor2):
    """
    :param coor1:dtype = np.array, shape = [:,4]
    :param coor2:
    :return:
    """
    start_max = np.maximum(coor1[:, 0:2], coor2[:, 0:2])  # [338,2]
    end_min = np.minimum(coor1[:, 2:4], coor2[:, 2:4])  # [338,2]
    lengths = end_min - start_max + 1  # [338,2]

    intersection = lengths[:, 0] * lengths[:, 1]
    intersection[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0

    union = ((coor1[:, 2] - coor1[:, 0] + 1) * (coor1[:, 3] - coor1[:, 1] + 1)
             + (coor2[:, 2] - coor2[:, 0] + 1) * (coor2[:, 3] - coor2[:, 1] + 1)
             - intersection)

    iou = intersection / union  # (338,)
    return iou

def AOLM(fms, fm1):
    A = torch.sum(fms, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > a).float()

    A1 = torch.sum(fm1, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    M1 = (A1 > a1).float()

    coordinates = []
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(14, 14)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))

        intersection = ((component_labels==(max_idx+1)).astype(int) + (M1[i][0].cpu().numpy()==1).astype(int)) ==2
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox
        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates

def compute_window_nums(ratios, stride, input_size):
    size = input_size / stride
    window_nums = []

    for _, ratio in enumerate(ratios):
        window_nums.append(int((size - ratio[0]) + 1) * int((size - ratio[1]) + 1))

    return window_nums

def computeCoordinate(image_size, stride, indice, ratio):
    size = int(image_size / stride)
    column_window_num = (size - ratio[1]) + 1
    x_indice = indice // column_window_num
    y_indice = indice % column_window_num
    x_lefttop = x_indice * stride - 1
    y_lefttop = y_indice * stride - 1
    x_rightlow = x_lefttop + ratio[0] * stride
    y_rightlow = y_lefttop + ratio[1] * stride
    # for image
    if x_lefttop < 0:
        x_lefttop = 0
    if y_lefttop < 0:
        y_lefttop = 0
    coordinate = np.array((x_lefttop, y_lefttop, x_rightlow, y_rightlow)).reshape(1, 4)

    return coordinate

def indices2coordinates(indices, stride, image_size, ratio):
    batch, _ = indices.shape
    coordinates = []

    for j, indice in enumerate(indices):
        coordinates.append(computeCoordinate(image_size, stride, indice, ratio))

    coordinates = np.array(coordinates).reshape(batch,4).astype(int)       # [N, 4]
    return coordinates

def image_with_boxes(image, coordinates=None, color=None):
    '''
    :param image: image array(CHW) tensor
    :param coordinate: bounding boxs coordinate, coordinates.shape = [proposalN, 4], coordinates[0] = (x0, y0, x1, y1)
    :return:image with bounding box(HWC)
    '''
    if type(image) is not np.ndarray:
        image = image.clone().detach()

        rgbN = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]

        # Anti-normalization
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
        image[0] = image[0] * std[0] + mean[0]
        image[1] = image[1] * std[1] + mean[1]
        image[2] = image[2].mul(std[2]) + mean[2]
        image = image.mul(255).byte()

        image = image.data.cpu().numpy()

        image.astype(np.uint8)

        image = np.transpose(image, (1, 2, 0))  # CHW --> HWC
        image = image.copy()

    if coordinates is not None:
        for i, coordinate in enumerate(coordinates):
            if color:
                image = cv2.rectangle(image, (int(coordinate[1]), int(coordinate[0])),
                                      (int(coordinate[3]), int(coordinate[2])),
                                      color, 2)
            else:
                if i < proposalN:
                # coordinates(x, y) is reverse in numpy
                    image = cv2.rectangle(image, (int(coordinate[1]), int(coordinate[0])), (int(coordinate[3]), int(coordinate[2])),
                                          rgbN[i], 2)
                else:
                    image = cv2.rectangle(image, (int(coordinate[1]), int(coordinate[0])),
                                          (int(coordinate[3]), int(coordinate[2])),
                                          (255, 255, 255), 2)
    return image