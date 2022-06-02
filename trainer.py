import os
import glob
import torch
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import max_checkpoint_num, proposalN, eval_trainset, save_checkpoint
from utils.eval_model import eval

def train_mmal(model,
               trainloader,
               testloader,
               criterion,
               optimizer,
               scheduler,
               save_path,
               num_epochs,
               save_interval,
               use_sam_optim,
               cuda_id):
    device = torch.device("cuda:"+str(cuda_id) if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        model.train()
        print('Epoch ', epoch+1)
        print('Training...')
        lr = next(iter(optimizer.param_groups))['lr']

        for i, data in enumerate(tqdm(trainloader)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _ = model(images, epoch, i, 'train')
            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                            labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            if epoch < 2:
                total_loss = raw_loss
            else:
                total_loss = raw_loss + local_loss + windowscls_loss

            if use_sam_optim:
                total_loss.backward()
                optimizer.first_step(zero_grad=True)
                proposalN_windows_score, proposalN_windows_logits, indices, \
                window_scores, _, raw_logits, local_logits, _ = model(images, epoch, i, 'train')
                raw_loss = criterion(raw_logits, labels)
                local_loss = criterion(local_logits, labels)
                windowscls_loss = criterion(proposalN_windows_logits,
                                labels.unsqueeze(1).repeat(1, proposalN).view(-1))
                if epoch < 2:
                    total_loss = raw_loss
                else:
                    total_loss = raw_loss + local_loss + windowscls_loss
                total_loss.backward()
                optimizer.second_step(zero_grad=True) 

            else:
                total_loss.backward()
                optimizer.step()

        scheduler.step()

        # evaluation every epoch
        if eval_trainset:
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, local_loss_avg\
                = eval(model, trainloader, criterion, 'train', save_path, epoch)

            print(
                'Train set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                    100. * raw_accuracy, 100. * local_accuracy))

            # tensorboard
            with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='train') as writer:

                writer.add_scalar('Train/learning rate', lr, epoch)
                writer.add_scalar('Train/raw_accuracy', raw_accuracy, epoch)
                writer.add_scalar('Train/local_accuracy', local_accuracy, epoch)
                writer.add_scalar('Train/raw_loss_avg', raw_loss_avg, epoch)
                writer.add_scalar('Train/local_loss_avg', local_loss_avg, epoch)
                writer.add_scalar('Train/windowscls_loss_avg', windowscls_loss_avg, epoch)
                writer.add_scalar('Train/total_loss_avg', total_loss_avg, epoch)

        # eval testset
        raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
        local_loss_avg\
            = eval(model, testloader, criterion, 'test', save_path, epoch)

        print(
            'Test set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                100. * raw_accuracy, 100. * local_accuracy))

        # tensorboard
        with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='test') as writer:
            writer.add_scalar('Test/raw_accuracy', raw_accuracy, epoch)
            writer.add_scalar('Test/local_accuracy', local_accuracy, epoch)
            writer.add_scalar('Test/raw_loss_avg', raw_loss_avg, epoch)
            writer.add_scalar('Test/local_loss_avg', local_loss_avg, epoch)
            writer.add_scalar('Test/windowscls_loss_avg', windowscls_loss_avg, epoch)
            writer.add_scalar('Test/total_loss_avg', total_loss_avg, epoch)

        # save checkpoint
        if save_checkpoint:
            if (epoch % save_interval == 0) or (epoch == end_epoch):
                print('Saving checkpoint')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'learning_rate': lr,
                }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

            # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
            # and delete the redundant ones
            checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
            if len(checkpoint_list) == max_checkpoint_num + 1:
                idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
                min_idx = min(idx_list)
                os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))

def train_mainstream_model(model,
                           trainloader,
                           testloader,
                           criterion,
                           optimizer,
                           scheduler,
                           cuda_id,
                           num_epochs):
    best_acc = -1
    train_loss = 0.0
    device = torch.device("cuda:"+str(cuda_id) if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        model.train()
        print('Epoch ', epoch+1)
        print('Training...')
        lr = next(iter(optimizer.param_groups))['lr']
        for i, data in enumerate(tqdm(trainloader)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        acc = []
        val_loss = 0.0
        print("Testing...")
        for idx, (images, labels) in enumerate(tqdm(testloader)):
            loss, correct = test(model, criterion, images, labels, device)
            val_loss += loss.item()
            acc.append(correct)

        test_accuracy = sum(acc)/len(acc)
        if test_accuracy>=best_acc:
            best_acc = test_accuracy
        else:
            pass
        train_loss = train_loss / len(trainloader)
        val_loss = val_loss / len(testloader)
        print("Epoch:{epoch} |train loss: {train_loss} |val loss: {val_loss} |val accuracy: {cur_acc} |best accuracy: {best_acc}"\
                                                    .format(epoch=epoch+1, 
                                                    train_loss=round(train_loss, 4), 
                                                    val_loss=round(val_loss, 4),
                                                    cur_acc=round(test_accuracy.item(), 4),
                                                    best_acc=round(best_acc.item(), 4)))

def test(model, criterion, X_val, y_val, device):
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        loss = criterion(logits, y_val)
        y_probs = torch.softmax(logits, dim = 1) 
        correct = (torch.argmax(y_probs, dim = 1) == y_val).type(torch.FloatTensor)
    return loss, correct.mean()

def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)