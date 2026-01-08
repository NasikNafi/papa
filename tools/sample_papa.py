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
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
import numpy as np

from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def sample_large_batch(model, base_model, scheduler, train_config, model_config, diffusion_config, num_samples=100):
    r"""
    Samples a large number of images for evaluation.
    Samples stepwise by going backward one timestep at a time.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    img.save(os.path.join(train_config['task_name'], train_config['sample_path'], 'images_{}.png'.format(suffix)))
    img.close()


def infer(args):
    
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

    # Read the config file
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

    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the base model from checkpoint
    base_model = Unet(model_config).to(device)
    base_model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['base_ckpt_name']), map_location=map_location))
    base_model.eval()

    # Load the finetuned model from checkpoint
    model = copy.deepcopy(base_model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['sample_path'],
                                                  train_config['ckpt_name']), map_location=map_location))
    model.eval()
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # generate and save samples using the finetuned model 
    with torch.no_grad():
        samples = sample_large_batch(model, base_model, scheduler, train_config, model_config, diffusion_config, num_samples=train_config['num_samples'])
    save_image_collage(samples, train_config, suffix=train_config['algo'])
    print("Successfully generated samples using the finetuned model!")

if __name__ == '__main__':

    # parsing config file
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default_epapa.yaml', type=str)
    args = parser.parse_args()

    infer(args)

    # calling signature
    # python -m tools.sample_4vis --config config/default_epapa.yaml
