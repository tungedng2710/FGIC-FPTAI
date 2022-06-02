from utils.indices2coordinates import indices2coordinates
from utils.compute_window_nums import compute_window_nums
import numpy as np

cuda_id = 2

save_checkpoint = False
loss_function_name = "ArcFace"
model_name = 'Mainstream'
stride = 32
channels = 2048
input_size = 448

batch_size = 4
vis_num = batch_size 
eval_trainset = False  
save_interval = 5
max_checkpoint_num = 10
num_epochs = 40

# Optimizer config
optim_name = "SGD"
init_lr = 0.01
lr_milestones = [60, 100]
lr_decay_rate = 0.1
weight_decay = 1e-4

# The path of pretrained model
pretrain_path = '/home/ubuntu/tungn197/AirCraft_Cls/MMAL-Net/models/pretrained/resnet50-19c8e357.pth'

N_list = [3, 2, 1]
proposalN = sum(N_list)  # proposal window num
window_side = [192, 256, 320]
iou_threshs = [0.25, 0.25, 0.25]
ratios = [[6, 6], [5, 7], [7, 5],
            [8, 8], [6, 10], [10, 6], [7, 9], [9, 7],
            [10, 10], [9, 11], [11, 9], [8, 12], [12, 8]]

model_path = './weights'      # pth save path
root_dir = '/home/ubuntu/tungn197/AirCraft_Cls/fgvc-aircraft-2013b'  # dataset path
num_classes = 100


'''indice2coordinates'''
window_nums = compute_window_nums(ratios, stride, input_size)
indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # 每个window在image上的坐标
coordinates_cat = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]
window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:8]), sum(window_nums[8:])]
