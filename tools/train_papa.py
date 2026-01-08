import os
import copy
import yaml
import shutil
import argparse
import torch
import torchvision
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
import numpy as np
from tools.train_classifier_mnist_digit import Net
from dataset.mnist_dataset import GeneratedDataset

from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def sample(model, base_model, scheduler, train_config, model_config, diffusion_config):
    r"""
    Samples a number of images defined by 'num_samples' in config for training.
    Samples stepwise by going backward one timestep at a time.
    """
    device = torch.cuda.current_device()
    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps'])), disable=True):
        # Get prediction of noise
        if i >= train_config['epapa_k']:
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        else:
            noise_pred = base_model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
    return xt


def sample_large_batch(model, base_model, scheduler, train_config, model_config, diffusion_config, num_samples=100):
    r"""
    Samples a large number of images for evaluation.
    Samples stepwise by going backward one timestep at a time.
    """
    device = torch.cuda.current_device()
    xt = torch.randn((num_samples,
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps'])), disable=True):
        # Get prediction of noise
        if i >= train_config['epapa_k']:
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        else:
            noise_pred = base_model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

    return xt


def save_image_collage(samples, train_config, suffix=""):
    ims = torch.clamp(samples, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, nrow=train_config['num_grid_rows'])
    img = torchvision.transforms.ToPILImage()(grid)
    if not os.path.exists(os.path.join(train_config['task_name'], train_config['sample_path'])):
        os.mkdir(os.path.join(train_config['task_name'], train_config['sample_path']))
    img.save(os.path.join(train_config['task_name'], train_config['sample_path'], 'x{}.png'.format(suffix)))
    img.close()


def save_samples(samples, train_config, suffix='', itr=0, rank=0):
    for (index, sample) in enumerate(samples):
        img = torch.clamp(sample, -1., 1.).detach().cpu()
        img = (img + 1) / 2
        img = torchvision.transforms.ToPILImage()(img)
        save_dir = os.path.join(train_config['task_name'], train_config['sample_path'], 'evaluation', 'epoch_{}'.format(suffix))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        img.save(os.path.join(save_dir, 'x{}_{}_{}.png'.format(index, itr, rank)))
        img.close()


if __name__ == '__main__':
    
    #####################################
    # DDP setup
    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']

    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    print('device', device)

    dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)
    print("Using dist.init_process_group. world_size ", world_size, 'World Rank', world_rank, 'Local Rank', local_rank, flush=True)
    #####################################
    
    # parsing config file
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default_epapa.yaml', type=str)
    args = parser.parse_args()

    # loading the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    #####################################
    
    # copy the config to output dir for experiment tracking
    if world_rank == 0:
        if not os.path.exists(os.path.join(train_config['task_name'], train_config['sample_path'])):
            os.mkdir(os.path.join(train_config['task_name'], train_config['sample_path']))
        shutil.copy(args.config_path, os.path.join(train_config['task_name'], train_config['sample_path'], os.path.basename(args.config_path)))

    # Load model with checkpoint
    base_model = Unet(model_config).to(device)
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                              train_config['base_ckpt_name']), map_location=map_location))
    model = copy.deepcopy(base_model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    mnist_model = Net().to(device)
    mnist_model.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['feedback_model']), map_location=map_location))
        
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Specify training parameters
    num_epochs = train_config['num_epochs']
    pref_loss_coef = train_config['pref_loss_coef']
    model_loss_coef = train_config['model_loss_coef']
    
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    model_criterion = torch.nn.MSELoss()
    noise_criterion = torch.nn.MSELoss(reduction='none')

    # using initial model generate and save samples for evaluation
    for i in range (train_config['sampling_epoch']):
        with torch.no_grad():
            samples = sample_large_batch(model, base_model, scheduler, train_config, model_config, diffusion_config, num_samples=train_config['sampling_size'])
        save_samples(samples, train_config, suffix=0, itr=i, rank=world_rank)

    # Run training
    for epoch_idx in range(num_epochs):

        if world_rank == 0:
            base_model.eval()    
            with torch.no_grad():
                samples = sample(model, base_model, scheduler, train_config, model_config, diffusion_config)

            # Save samples
            save_image_collage(samples, train_config, suffix=epoch_idx+1)
            
            # Receive automated feedback
            mnist_model.eval()
            with torch.no_grad():
                label_prob  = mnist_model(samples)
                label_pred = label_prob.argmax(dim=1, keepdim=True)

            if train_config['full_exploitation']:
                # uses both preference (positive) and non-preference (negative) data
                feedback_list = [1.0 if l in train_config['preference'] else -1.0 for l in label_pred]

                # for non-binary preference, define get_pref_weightage function to obtain preference weightage.
                # feedback_list = [get_pref_weightage(l) if l in train_config['preference'] else -1.0 for l in label_pred]
            else:
                # uses only preference (positive) data
                feedback_list = [1.0 if l in train_config['preference'] else 0.0 for l in label_pred]
            feedback = torch.tensor(feedback_list).to(device)   
        else:
            samples = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
            feedback = torch.randn(train_config['num_samples']).to(device)
            
        dist.barrier()
        dist.broadcast(feedback, src=0)
        dist.broadcast(samples, src=0)
        dist.barrier()
        
        # create new dataloader with generated samples
        generated_dataset = GeneratedDataset(samples.cpu().numpy(), feedback.cpu().tolist())
        data_loader = DataLoader(generated_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=1)

        # finetuing for preference alignment
        losses = []
        model.train()
        for im, label in tqdm(data_loader, disable=True):
            optimizer.zero_grad()
            im = im.float().to(device)
            label = label.float().to(device)

            # sample random noise
            noise = torch.randn_like(im).to(device)
            
            # calculate loss terms
            summed_pref_loss = torch.zeros(im.shape[0],).to(device)
            
            if train_config['algo']=='epapa':
                assert ((diffusion_config['num_timesteps'] - train_config['epapa_k']) % world_size == 0), "total effective timesteps must be a multiple of total number of ranks"
                tsteps_per_rank = int((diffusion_config['num_timesteps'] - train_config['epapa_k'])/world_size)
                tsteps = torch.linspace((world_rank*tsteps_per_rank)+train_config['epapa_k'], (world_rank*tsteps_per_rank)+train_config['epapa_k']+(tsteps_per_rank-1), tsteps_per_rank).to(device)
            else:
                assert (diffusion_config['num_timesteps'] % world_size == 0), "total timesteps must be a multiple of total number of ranks"
                tsteps_per_rank = int(diffusion_config['num_timesteps']/world_size)
                tsteps = torch.linspace(world_rank*tsteps_per_rank, (world_rank*tsteps_per_rank)+(tsteps_per_rank-1), tsteps_per_rank).to(device)

            for i in tsteps:
                t = torch.full((im.shape[0],), int(i)).to(device)
            
                # Add noise to images according to timestep
                noisy_im = scheduler.add_noise(im, noise, t)

                # predict noise and calculate loss
                noise_pred = model(noisy_im, t)
                pred_error = noise_criterion(noise_pred, noise).mean(dim=[1,2,3])
                
                t = t.to('cpu')
                scaling_factor = ((1. - scheduler.alphas[t])/(scheduler.alphas[t] * (1.- scheduler.alpha_cum_prod[t-1]))).to(device)
                t = t.to(device)

                noise_pred_loss = scaling_factor * pred_error
                summed_pref_loss+=noise_pred_loss

            # calculate model similarity loss, the QPDE in the paper
            if model_loss_coef > 0 and (label==1.0):
                with torch.no_grad():
                    noise_pred_base = base_model(noisy_im, t)
                model_loss = model_criterion(noise_pred, noise_pred_base)
            else:
                model_loss = 0.0

            # determine the final loss: add the preference loss (exploitation) with model similarity loss (exploration)
            final_loss = ((label * summed_pref_loss * pref_loss_coef) + (model_loss * model_loss_coef)).mean()
            losses.append(final_loss.item())

            # take the optimization step
            final_loss.backward()
            optimizer.step()

        if world_rank == 0:
            print('Rank', world_rank, 'Epoch: ', epoch_idx, 'Loss: ', format(np.mean(losses), '.10f'))
        
        # generate and save samples for evaluation
        for i in range (train_config['sampling_epoch']):
            with torch.no_grad():
                samples = sample_large_batch(model, base_model, scheduler, train_config, model_config, diffusion_config, num_samples=train_config['sampling_size'])
            save_samples(samples, train_config, suffix=epoch_idx+1, itr=i, rank=world_rank)


    if world_rank == 0:
        torch.save(model.state_dict(), os.path.join(train_config['task_name'], train_config['sample_path'], train_config['ckpt_name']))

        with torch.no_grad():
            samples = sample_large_batch(model, base_model, scheduler, train_config, model_config, diffusion_config, num_samples=100)
    
        # Save 100 samples from final model
        save_image_collage(samples, train_config, suffix='_final')
        
    dist.barrier()
    
    dist.destroy_process_group()