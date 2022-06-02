#coding=utf-8
import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
from torch import nn
import shutil
import time
from config import *

from trainer import train_mmal, train_mainstream_model
from models.MMALNet.model import MainNet
from models.mainstream_model import MainStreamModel, DistributedMainStreamModel

from utils.dataset import FGVC_aircraft_loader
from utils.losses import get_loss
from utils.optimizers import SAM, Lamb

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

def train():
    device = torch.device("cuda:"+str(cuda_id) if torch.cuda.is_available() else "cpu")

    dataloader = FGVC_aircraft_loader(root_dir=root_dir, 
                                      input_size=input_size, 
                                      batch_size=batch_size)
    trainloader, testloader = dataloader.get_dataloader()

    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

    # criterion = nn.CrossEntropyLoss()
    criterion = get_loss(name=loss_function_name)
    parameters = model.parameters()
    lr = init_lr
    
    if optim_name == "SGD":
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
        use_sam_optim = False
    elif optim_name == "SAM":
        optimizer = SAM(parameters, lr=lr, momentum=0.9, rho=0.05, adaptive=False)
        use_sam_optim = True
    elif optim_name == "ASAM":
        optimizer = SAM(parameters, lr=lr, momentum=0.9, rho=0.05, adaptive=True)
        use_sam_optim = True
    elif optim_name == "Adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        use_sam_optim = False
    
    save_path = os.path.join(model_path, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # model= nn.DataParallel(model,device_ids = [1, 2])
    model = model.to(device)

    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate)

    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))

    model = MainStreamModel(input_size==[input_size, input_size])
    model = model.to(device)
    # model = DistributedMainStreamModel()
    train_mainstream_model(model,
                            trainloader,
                            testloader,
                            criterion,
                            optimizer,
                            scheduler,
                            cuda_id,
                            num_epochs)
    # train_mmal(model=model,
    #       trainloader=trainloader,
    #       testloader=testloader,
    #       criterion=criterion,
    #       optimizer=optimizer,
    #       scheduler=scheduler,
    #       save_path=save_path,
    #       num_epochs=num_epochs,
    #       save_interval=save_interval,
    #       use_sam_optim=use_sam_optim,
    #       cuda_id=cuda_id)


if __name__ == '__main__':
    train()