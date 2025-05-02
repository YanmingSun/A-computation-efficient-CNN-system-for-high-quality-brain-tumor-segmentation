import os
import time
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.utils.data.distributed
import argparse

import BratsName
import pre_processing
import load_training_data_NET_ET
import Network_ThreePathways
# import Simple_UNet
import Testing_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter

# from torchsummary import summary
# from thop import profile
# from torchstat import stat


parser = argparse.ArgumentParser(description='Brain tumor segmentation models, distributed data parallel test')
parser.add_argument('--lr', default=0.01, help='')
parser.add_argument('--batch_size', type=int, default=50, help='')
parser.add_argument('--max_epochs', type=int, default=50, help='')
parser.add_argument('--num_workers', type=int, default=8, help='')

parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='gloo', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')


def main():
    print("Starting...")

    branch_name = 'SEG_2CV1'
    roi_training_path = '/scratch/sym/BrainTumorSegmentation/Code/Coarse_Fine_Seg/SEG_2C/TestTrainingSet_ROI_2D_ROI/'
    roi_test_path = '/scratch/sym/BrainTumorSegmentation/Code/Coarse_Fine_Seg/SEG_2C/Test_ROI_2D_ROI/'

    data_path_id = BratsName.Brats_Path_ID()
    training_path = data_path_id.path_training_cases
    validation_path = data_path_id.path_validation_cases
    brats_training_ID = data_path_id.training_cases
    brats_validation_ID = data_path_id.validation_cases

    checkpoint_folder = './CheckPoint_' + branch_name + '/'
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    checkpoint_path = os.path.join(checkpoint_folder, 'model_et')

    num_cases = len(brats_training_ID)
    print('number of cases: ', num_cases)

    training_images = pre_processing.BratsTrainingSet(training_path, roi_training_path, brats_training_ID[0:int(num_cases * 0.75)])
    print('number of slices: ', training_images.num_total_image)

    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    # This next line is the key to getting DistributedDataParallel working on SLURM:
    # SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the
    # current process inside a node and is also 0 or 1 in this example.

    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank

    current_device = local_rank

    torch.cuda.set_device(current_device)

    # this block initializes a process group and initiate communications
    # between all processes running on all nodes

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    # init the process group
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size,
                            rank=rank)

    print('dist.init_process_group ', args.dist_backend, args.init_method, args.world_size, rank)

    print("process group ready!")

    print('From Rank: {}, ==> Making model..'.format(rank))

    net = Network_ThreePathways.TC_Pathway()
    net.cuda()

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[current_device])

    print('From Rank: {}, ==> Preparing data..'.format(rank))

    train_data = load_training_data_NET_ET.TorchDataset_NET_ET(image_x=training_images.image_x, image_seg=training_images.image_seg,
                                                 patch_map=training_images.patch_map, repeat=1)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers,
                              sampler=train_sampler)

    print('DataLoader: ', args.num_workers)

    criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(
        training_images.num_total_image / args.batch_size * args.max_epochs / args.world_size) + 100, eta_min=0.000001)
    print('world size', args.world_size)

    for epoch in range(args.max_epochs):
        train_sampler.set_epoch(epoch)

        train(epoch, net, criterion, optimizer, scheduler, train_loader, rank)

        if args.max_epochs - epoch <= 5:
            torch.save(net.module.state_dict(), checkpoint_path + str(epoch))


def train(epoch, net, criterion, optimizer, scheduler, train_loader, train_rank):
    epoch_start = time.time()

    for step, (batch_image_tc, batch_label) in enumerate(train_loader):
        # batch_image = Variable(batch_image, requires_grad=False)
        # batch_label = Variable(batch_label, requires_grad=False)
        batch_image_tc = batch_image_tc.cuda()
        batch_label = batch_label.cuda()

        output_tc = net(batch_image_tc)

        loss_tc = criterion(output_tc, batch_label[:, :, :])

        lr_present = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()
        loss_tc.backward()
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            print("From Rank: {}, epoch: {}, step: {}, loss: {}, lr: {}".format(train_rank, epoch, step,
                                                                              loss_tc, lr_present))

    duration = time.time() - epoch_start
    print("From Rank: {}, epoch: {}, epoch time: {}".format(train_rank, epoch, duration))


if __name__ == '__main__':
    main()
