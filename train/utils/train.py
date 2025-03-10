
# this function iterates through the batch
# one batch computation is processed and backward, metrics are recorded
# dataloader: train_dataloader or val_dataloader (return batched samples)
# model: models to train or validating
# optimizer: optimizer to optimize the model weight
# criterion: loss function (objective) to generate loss
# metrics: metrics for training or validating (loss and IOU can be recoreded)
# device: device to run the model
# mode: 'train' or 'test' -> 'train' for training model, 'test' for validating or testing model
def run_iter(dataloader, model, optimizer, criterion, metrics, device,  mode):
    for i, batch in enumerate(dataloader, 1):
        image, mask = batch
        image = image.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        prediction_mask = model(image)
        loss = criterion(prediction_mask, mask)
        loss.backward()
        optimizer.step()
        
        
        
        


#  for i, batch in enumerate(train_dataloader, 1): # batch, i follows 1-index ordering
    #         image, mask = batch
    #         image = image.to(local_gpu_id)
    #         mask = mask.to(local_gpu_id)
            
    #         optimizer.zero_grad()
            
    #         prediction_mask = convnext_unet(image)
    #         loss = criterion(prediction_mask, mask)
    #         loss.backward()
    #         optimizer.step()
            
    #         loss_container.append(loss.detach().item())
    #         dice_score = get_dice_score(prediction_mask.cpu().detach().numpy(), mask.cpu().detach().numpy())
    #         dice_container.append(dice_score)
    #         lane_score = get_lane_score(prediction_mask.cpu().detach().numpy(), mask.cpu().detach().numpy())
    #         lane_container.append(lane_score)

    #         running_loss += loss.detach().item()
    #         running_lane += lane_score
            
    #         if scheduler_type in ['one_cycle']: # in case of pytorch OneCycleLR scheduler, it update itself step-wise (only training process will update the scheduler)
    #             scheduler.step()
            
    #         if i % print_threshold == 0:
    #             print(f"training [{epoch+1}:{i:5d}/{total_train_iter}] loss: {running_loss/print_threshold:.5f} lane_score: {running_lane/print_threshold:.5f} lr: {optimizer.param_groups[0]['lr']}")
    #             if opts.rank==0:
    #                 train_logger.info(f"training [{epoch+1}:{i:5d}/{total_train_iter}] loss: {running_loss/print_threshold:.5f} lane_score: {running_lane/print_threshold:.5f} lr: {optimizer.param_groups[0]['lr']}")
    #             running_loss = 0.0
    #             running_lane = 0.0


# for i, batch in enumerate(val_dataloader, 1):
    #                 image, mask = batch
    #                 image = image.to(opts.rank)
    #                 mask = mask.to(opts.rank)
                    
    #                 prediction_mask = convnext_unet(image)
    #                 loss = criterion(prediction_mask, mask)
                    
    #                 loss_container.append(loss.detach().item())
                    
    #                 dice_score = get_dice_score(prediction_mask.cpu().detach().numpy(), mask.cpu().detach().numpy())
    #                 dice_container.append(dice_score)
    #                 lane_score = get_lane_score(prediction_mask.cpu().detach().numpy(), mask.cpu().detach().numpy())
    #                 lane_container.append(lane_score)
                    
    #                 running_loss += loss.detach().item()
    #                 running_lane += lane_score
                    
    #                 if i % val_print_threshold == 0:
    #                     print(f"Validation [{epoch+1}:{i:5d}/{total_val_iter}] loss: {running_loss/val_print_threshold:.5f} lane_score: {running_lane/val_print_threshold:.5f}")
    #                     val_logger.info(f"Validation [{epoch+1}:{i:5d}/{total_val_iter}] loss: {running_loss/val_print_threshold:.5f} lane_score: {running_lane/val_print_threshold:.5f}")
    #                     running_loss = 0.0
    #                     running_lane = 0.0
    

def run_epoch():
    pass


    # train_history = []
    # val_history = []

    # train_dice = []
    # val_dice = []
    
    # # calculates only lane part on the image
    # train_lane = []
    # val_lane = []
    
    # lr_history = []

    # train_num_prints = 25
    # val_num_prints = 10

    # total_train_iter = len(train_dataloader)
    # total_val_iter = len(val_dataloader)

    # print_threshold = int(total_train_iter/train_num_prints)
    # val_print_threshold = int(total_val_iter/val_num_prints)
        
    
    # for epoch in range(start_epoch, configs['epoch']): # epochs follows 0-index ordering
    #     convnext_unet.train()
    #     train_sampler.set_epoch(epoch)
        
    #     running_loss = 0.0 # save temporary dice loss
    #     running_lane = 0.0 # save temporary lane loss
    #     loss_container = []
    #     dice_container = []
    #     lane_container = []
        
    #     if opts.rank == 0:
    #         train_logger.info(f'epoch: {epoch+1} start!')

    #    
            
    #     loss = np.mean(loss_container)
    #     dice_coeff = np.mean(dice_container)
    #     lane_coeff = np.mean(lane_container)
    #     train_history.append(loss)
    #     train_dice.append(dice_coeff)
    #     train_lane.append(lane_coeff)
    #     lr_history.append(optimizer.param_groups[0]['lr'])
        
    #     print("="*50)
    #     print(f"Train Epoch: {epoch+1}, Dice Score: {dice_coeff:.5f}, Lane Score: {lane_coeff:.5f}, Loss: {loss:.5f}, LR: {optimizer.param_groups[0]['lr']}")
    #     print("="*50)
        
    #     if opts.rank == 0:
    #         train_logger.info('=' * 50 + f"\nTrain Epoch: {epoch+1}, Dice Score: {dice_coeff:.3f}, Lane Score: {lane_coeff:.3f}, Loss: {loss:.3f}, LR: {optimizer.param_groups[0]['lr']}\n" + '=' * 50)
        
    #     if opts.rank==0: # only rank 0 will run validation
    #         convnext_unet.eval()
    #         running_loss = 0.0
    #         running_lane = 0.0
    #         loss_container = []
    #         dice_container = []
    #         lane_container = []
            
    #         val_logger.info(f'epoch: {epoch+1} start!')
    #         with torch.no_grad():
    #             
        
    #         loss = np.mean(loss_container)
    #         dice_coeff = np.mean(dice_container)
    #         lane_coeff = np.mean(lane_container)
    #         val_history.append(loss)
    #         val_dice.append(dice_coeff)
    #         val_lane.append(lane_coeff)
    #         print("="*50)
    #         print(f"Val Epoch: {epoch+1}, Dice Score: {dice_coeff:.5f}, Lane Score: {lane_coeff:.5f}, Loss: {loss:.5f}, LR: {optimizer.param_groups[0]['lr']}")
    #         print("="*50)
    #         val_logger.info('='*50 + f"\nVal Epoch: {epoch+1}, Dice Score: {dice_coeff:.5f}, Lane Score: {lane_coeff:.5f}, Loss: {loss:.5f}, LR: {optimizer.param_groups[0]['lr']}\n" + '='*50)
        
    #     # cosine_warmup_restart uses timm library whoose schedulers are updated to epoch-wise
    #     if scheduler_type in ['cosine_warmup_restart']:    
    #         # update scheduler
    #         scheduler.step(epoch)
        
    #     if opts.rank == 0: # only rank 0 will save the model and plots
    #         checkdir(opts.save_path+'/weights')
    #         if ((epoch+1) % opts.save_step) == 0 or (epoch+1)==configs['epoch']:
    #             print("+"*50)
    #             print(f"Saving Model to {opts.save_path}/weights...")
    #             print("+"*50)
                
    #             currtime = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y_%m_%d_%H%M%S")
    #             torch.save({
    #                 'epoch': epoch,
    #                 'model_state_dict' : convnext_unet.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'scheduler_state_dict' : scheduler.state_dict(),
    #                 'consumed_batch': (epoch+1) * len(train_dataloader) # number of total consumed iteration (number of total consumed batches during training)
    #             }, f'{opts.save_path}/weights/{currtime}_epoch{epoch+1}.pth') # epochs are saved as 1-index ordering (index 0 means initial state)
                
    #         if ((epoch+1) % opts.save_plot_step) == 0 or (epoch+1)==configs['epoch']:
    #             save_plots(train_history, val_history, train_dice, val_dice, train_lane, val_lane, lr_history, epoch+1, opts.save_plot_path) # epochs as 1-index ordering
