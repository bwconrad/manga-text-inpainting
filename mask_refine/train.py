import torch
import time
import os
import shutil

from utils import *
from test import *


def update_learning_rate(schedulers, type):
    '''
        Update learning rates for all the networks, called at the end of every epoch
    '''
    for scheduler in schedulers:
        if type == 'plateau':
            scheduler.step(0)
        else:
            scheduler.step()

def train(net, train_loader, val_loader, optimizer, scheduler, criterion, 
          start_epoch, device, args, train_hist=None):
    
    if not train_hist:
        train_hist = {'T_losses': [],
                      'T_val_losses': [],
                      'Precision': [], 
                      'Recall': [],
                      'F1': []}
    start = time.time()

    print('\nStarting to train...')
    for epoch in range(start_epoch, args.epochs+1):
        net.train()
        start_epoch = time.time()

        # Batch losses of the current epoch
        losses = []
        
        for i, (images, masks, text_masks, _) in enumerate(train_loader):
            images, masks, text_masks = images.to(device), masks.to(device), text_masks.to(device)
            optimizer.zero_grad()   
            
            # Tversky loss
            outputs = net(torch.cat((images, masks), 1))
            loss = criterion(outputs, text_masks)

            loss.backward()
            optimizer.step()

            # Save batch losses
            losses.append(loss.detach().item())

            if (i+1)%args.batch_log_rate == 0:
                print('[Epoch {}/{}, Batch {}/{}] Tversky loss: {}'
                      .format(epoch, args.epochs, i+1, len(train_loader), np.mean(losses)))
            
        # Print epoch information
        print_epoch_stats(epoch, start_epoch, time.time(), losses, train_hist)
       
        # Evaluate on validation set
        print('Evaluating on validation set...')
        if epoch%args.save_samples_rate == 0:
            save_path = args.save_samples_path+'epoch{}/'.format(epoch)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        
            avg_precision, avg_recall, avg_t_loss = test(net, val_loader, device, criterion, save_batches=args.save_samples_batches, save_path=save_path)
        
        else:
            avg_precision, avg_recall, avg_t_loss = test(net, val_loader, device, criterion)

        f1 = (2*avg_precision*avg_recall) / (avg_precision+avg_recall)
        train_hist['Precision'].append(avg_precision)
        train_hist['Recall'].append(avg_recall)
        train_hist['F1'].append(f1)
        train_hist['T_val_losses'].append(avg_t_loss)
        print("Precision: {} Recall: {} F1: {} Tversky Loss: {}\n".format(avg_precision, avg_recall, f1, avg_t_loss))
        
        # Save model
        save_checkpoint({'epoch': epoch,
                         'state_dict': net.state_dict(),
                         'optimizer_state_dict' : optimizer.state_dict(),
                         'scheduler_state_dict' : scheduler.state_dict() if scheduler else None,
                         'args': args,
                         'train_hist': train_hist
                        }, epoch, args.checkpoint_path)



        # Save training history plot
        save_plots(train_hist, args.plot_path)

        # Update lr schedulers
        if scheduler:
            update_learning_rate([scheduler], args.scheduler)

    if args.archive:
        shutil.make_archive('images', 'zip', args.save_samples_path)
        shutil.make_archive('checkpoints', 'zip', args.checkpoint_path)
        shutil.make_archive('plots', 'zip', args.plot_path)


    hours, minutes, seconds = calculate_time(start, time.time())
    print('Training completed in {}h {}m {:04.2f}s'.format(hours, minutes, seconds))