import os, time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils
import global_constants as settings
from log_utils import log
from monodepth_model import MonodepthModel
import subprocess
import matplotlib.pyplot as plt


def train(train_image0_path,
          train_image1_path,
          train_camera_path,
          # Batch settings
          n_batch=settings.N_BATCH,
          n_height=settings.N_HEIGHT,
          n_width=settings.N_WIDTH,
          encoder_type=settings.ENCODER_TYPE,
          decoder_type=settings.DECODER_TYPE,
          activation_func=settings.ACTIVATION_FUNC,
          n_pyramid=settings.N_PYRAMID,
          # Training settings
          n_epoch=settings.N_EPOCH,
          learning_rates=settings.LEARNING_RATES,
          learning_schedule=settings.LEARNING_SCHEDULE,
          use_augment=settings.USE_AUGMENT,
          w_color=settings.W_COLOR,
          w_ssim=settings.W_SSIM,
          w_smoothness=settings.W_SMOOTHNESS,
          w_left_right=settings.W_LEFT_RIGHT,
          # Depth range settings
          scale_factor=settings.SCALE_FACTOR,
          # Checkpoint settings
          n_summary=settings.N_SUMMARY,
          n_checkpoint=settings.N_CHECKPOINT,
          checkpoint_path=settings.CHECKPOINT_PATH,
          # Hardware settings
          device=settings.DEVICE,
          n_thread=settings.N_THREAD,
          depth_model_restore_path0=settings.DEPTH_MODEL_RESTORE_PATH0,
          depth_model_restore_path1=settings.DEPTH_MODEL_RESTORE_PATH1):

    if device == settings.CUDA or device == settings.GPU:
        device = torch.device(settings.CUDA)
    else:
        device = torch.device(settings.CPU)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and event paths
    encoder_checkpoint_path = os.path.join(checkpoint_path, 'encoder-{}.pth')
    decoder_checkpoint_path = os.path.join(checkpoint_path, 'decoder-{}.pth')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    # Read paths for training
    train_image0_paths = data_utils.read_paths(train_image0_path)
    train_image1_paths = data_utils.read_paths(train_image1_path)
    train_camera_paths = data_utils.read_paths(train_camera_path)

    # Read paths for validation set
    pathSubString = train_image0_path.split("kitti_eigen", 1)[0]
    val_image0_paths = data_utils.read_paths(pathSubString + "kitti_eigen_val_image0.txt")
    val_image1_paths = data_utils.read_paths(pathSubString + "kitti_eigen_val_image1.txt")
    val_camera_paths = data_utils.read_paths(pathSubString + "kitti_eigen_val_camera.txt")

    assert len(train_image0_paths) == len(train_image1_paths)
    assert len(train_image0_paths) == len(train_camera_paths)

    assert len(val_image0_paths) == len(val_image1_paths)
    assert len(val_image0_paths) == len(val_camera_paths)

    n_train_sample = len(train_image0_paths)
    n_train_step = n_epoch*np.ceil(n_train_sample/n_batch).astype(np.int32)

    n_val_sample = len(val_image0_paths)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.ImagePairCameraDataset(
            train_image0_paths,
            train_image1_paths,
            train_camera_paths,
            shape=(n_height, n_width),
            augment=use_augment),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=False)

    val_dataloader = torch.utils.data.DataLoader(
        datasets.ImagePairCameraDataset(
            val_image0_paths,
            val_image1_paths,
            val_camera_paths,
            shape=(n_height, n_width),
            augment=use_augment),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=False)

    # Build network
    model = MonodepthModel(
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        activation_func=activation_func,
        n_pyramid=n_pyramid,
        scale_factor=scale_factor,
        device=device)
    train_summary = SummaryWriter(event_path)
    parameters = model.parameters()
    n_param = sum(p.numel() for p in parameters)


    #load saved model params
    saved_epoch = 0
    if depth_model_restore_path0 != "" and depth_model_restore_path1 != "":
        saved_epoch = model.restore_model(
                encoder_restore_path=depth_model_restore_path0,
                decoder_restore_path=depth_model_restore_path1)

        log('Model Loaded at epoch =%d' % saved_epoch, log_path)






    # Start training
    model.train()

    log('Network settings:', log_path)
    log('n_batch=%d  n_height=%d  n_width=%d  n_param=%d' %
        (n_batch, n_height, n_width, n_param), log_path)
    log('encoder_type=%s  decoder_type=%s  activation_func=%s  n_pyramid=%d' %
        (encoder_type, decoder_type, activation_func, n_pyramid), log_path)
    log('Training settings:', log_path)
    log('n_sample=%d  n_epoch=%d  n_step=%d' %
        (n_train_sample, n_epoch, n_train_step), log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}:{}'.format(l*(n_train_sample//n_batch), v, log_path)
        for l, v in zip([0] + learning_schedule, learning_rates)), log_path)
    log('use_augment=%s' % use_augment, log_path)
    log('w_color=%.2f  w_ssim=%.2f  w_smoothness=%.2f  w_left_right=%.2f' %
        (w_color, w_ssim, w_smoothness, w_left_right), log_path)
    log('Depth range settings:', log_path)
    log('scale_factor=%.2f' %
        (scale_factor), log_path)
    log('Checkpoint settings:', log_path)
    log('depth_model_checkpoint_path=%s' % checkpoint_path, log_path)
    

    learning_schedule.append(n_epoch)
    schedule_pos = 0
    train_step = saved_epoch*np.ceil(n_train_sample/n_batch).astype(np.int32)
    time_start = time.time()
    log('Begin training...', log_path)



    #Validation Set loss
    running_val_loss = 0
    for val_image0, val_image1, val_camera in val_dataloader:
        # Fetch data
        if device.type == settings.CUDA:
            val_image0 = val_image0.cuda()
            val_image1 = val_image1.cuda()
            val_camera = val_camera.cuda()

        # Forward through the network
        model.forward(val_image0, val_camera)

        # Compute loss function
        val_loss = model.compute_loss(val_image0, val_image1,
            w_color=w_color,
            w_ssim=w_ssim,
            w_smoothness=w_smoothness,
            w_left_right=w_left_right)

        running_val_loss += val_loss.item()

    running_val_loss /= (n_val_sample / n_batch)
    print("Val Loss: " + str(val_loss.item()))
    train_summary.add_scalar('val_loss', running_val_loss, global_step=train_step)

    #Print images
    plot_img = np.transpose(np.squeeze(val_image0[0].detach().cpu().numpy()), (1, 2, 0))
    plt.imshow(plot_img)
    plt.show()
    plot_img = np.transpose(np.squeeze(val_image1[0].detach().cpu().numpy()), (1, 2, 0))
    plt.imshow(plot_img)
    plt.show()
    plot_img = np.transpose(np.squeeze(model.image1w[0].detach().cpu().numpy()), (1, 2, 0))
    plt.imshow(plot_img)
    plt.show()
    plot_img = np.squeeze(model.disparity0[0].detach().cpu().numpy())
    plt.imshow(plot_img)
    plt.show()

    scaler = torch.cuda.amp.GradScaler() 

    for epoch in range(saved_epoch+1, n_epoch+1):
        # Set learning rate schedule
        while epoch > learning_schedule[schedule_pos]:
            schedule_pos = schedule_pos + 1
        learning_rate = learning_rates[schedule_pos]
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        running_train_loss = 0.0
        for train_image0, train_image1, train_camera in train_dataloader:
            train_step = train_step + 1
            # Fetch data
            if device.type == settings.CUDA:
                train_image0 = train_image0.cuda()
                train_image1 = train_image1.cuda()
                train_camera = train_camera.cuda()

            #Generate Pertubation
            epsilon = 8/255.
            alpha = 10/255.

            with torch.cuda.amp.autocast():
                delta = torch.zeros_like(train_image0).uniform_(-epsilon, epsilon).cuda()
                delta.requires_grad = True
                delta_image0 = train_image0 + delta
                model.forward(delta_image0, train_camera)
                loss = model.compute_loss(delta_image0, train_image1,
                    w_color=w_color,
                    w_ssim=w_ssim,
                    w_smoothness=w_smoothness,
                    w_left_right=w_left_right)

            scaler.scale(loss).backward()

            with torch.cuda.amp.autocast():
                delta_grad = delta.grad.detach()
                delta.data = torch.clamp(delta + alpha * torch.sign(delta_grad), -epsilon, epsilon)
                delta = delta.detach()
                pert_image0 = torch.clamp(train_image0 + delta, 0, 1)


                # Perturbed Feed-Forward through the network
                model.forward(pert_image0, train_camera)

                # Compute loss function
                loss_pert = model.compute_loss(pert_image0, train_image1,
                    w_color=w_color,
                    w_ssim=w_ssim,
                    w_smoothness=w_smoothness,
                    w_left_right=w_left_right)

                #Normal feed-forward
                model.forward(train_image0, train_camera)

                # Compute loss function
                loss_normal = model.compute_loss(train_image0, train_image1,
                    w_color=w_color,
                    w_ssim=w_ssim,
                    w_smoothness=w_smoothness,
                    w_left_right=w_left_right)

                loss = 0.5*loss_pert + 0.5*loss_normal

                running_train_loss += loss.item()

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print("\rTrain Step: " + str(train_step) + "  Loss: " + str(loss.item()), end="")

            if (train_step % n_summary) == 0:

                model.log_summary(
                    summary_writer=train_summary,
                    step=train_step)
                
                #Validation Set loss
                running_val_loss = 0
                for val_image0, val_image1, val_camera in val_dataloader:
                    # Fetch data
                    if device.type == settings.CUDA:
                        val_image0 = val_image0.cuda()
                        val_image1 = val_image1.cuda()
                        val_camera = val_camera.cuda()

                    with torch.cuda.amp.autocast():
                        # Forward through the network
                        model.forward(val_image0, val_camera)

                        # Compute loss function
                        val_loss = model.compute_loss(val_image0, val_image1,
                            w_color=w_color,
                            w_ssim=w_ssim,
                            w_smoothness=w_smoothness,
                            w_left_right=w_left_right)

                    running_val_loss += val_loss.item()

                running_val_loss /= (n_val_sample / n_batch)
                print("\n")
                print("Val Run Loss: " + str(running_val_loss))
                train_summary.add_scalar('val_loss', running_val_loss, global_step=train_step)


        #After epoch finish add running loss
        running_train_loss /= (n_train_sample / n_batch)
        print("Train Run Loss: " + str(running_train_loss))
        train_summary.add_scalar('train_run_loss', running_train_loss, global_step=train_step)

        if (epoch % 2) == 0:
            #Print images
            plot_img = np.transpose(np.squeeze(pert_image0[0].detach().cpu().numpy()), (1, 2, 0))
            plt.imshow(plot_img)
            plt.show()
            plot_img = np.transpose(np.squeeze(train_image1[0].detach().cpu().numpy()), (1, 2, 0))
            plt.imshow(plot_img)
            plt.show()
            plot_img = np.transpose(np.squeeze(model.image1w[0].detach().cpu().numpy()), (1, 2, 0))
            plt.imshow(plot_img)
            plt.show()
            plot_img = np.squeeze(model.disparity0[0].detach().cpu().numpy()).astype('float32')
            plt.imshow(plot_img)
            plt.show()




        # Log results and save checkpoints
        if (epoch % 2) == 0:
            time_elapse = (time.time() - time_start) / 3600
            time_remain = (n_train_step - train_step) * time_elapse / train_step
            log('Step={:6}/{}  Loss={:.5f} Val_Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                train_step, n_train_step, loss.item(), running_val_loss, time_elapse, time_remain), log_path)

            # Save checkpoints
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, encoder_checkpoint_path.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, decoder_checkpoint_path.format(epoch))
            
            # #Copy saved network params
            pathSubString = checkpoint_path.split("local/", 1)[1]
            command = "rclone copy " + checkpoint_path + " onedrive:'MSc Project/" + pathSubString + "'"
            print(command + "\n")
            direct_output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

            # Delete local
            command = "rm " + checkpoint_path + "/encoder-"+str(epoch)+".pth" + " " + checkpoint_path + "/decoder-"+str(epoch)+".pth"
            print(command + "\n")
            direct_output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)




    # Save checkpoints and close summary
    train_summary.close()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, encoder_checkpoint_path.format(epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, decoder_checkpoint_path.format(epoch))

    # #Copy saved network params
    pathSubString = checkpoint_path.split("local/", 1)[1]
    command = "rclone copy " + checkpoint_path + " onedrive:'MSc Project/" + pathSubString + "'"
    print(command + "\n")
    direct_output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

    # Delete local
    command = "rm " + checkpoint_path + "/encoder-"+str(epoch)+".pth" + " " + checkpoint_path + "/decoder-"+str(epoch)+".pth"
    print(command + "\n")
    direct_output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
