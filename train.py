import os
import time
from options.train_options import TrainOptions
from models import create_model
from datasets.dataset import getDataset
from tqdm import tqdm
from util.save_image import save_image
import torch
import numpy as np
import random

def init_seeds(RANDOM_SEED=1, no=1):
    RANDOM_SEED += no
    print("local_rank = {}, seed = {}".format(no, RANDOM_SEED))
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    init_seeds(no=1)
    opt = TrainOptions().parse()
    if os.path.isdir(opt.checkpoints_dir + opt.name) == False:
        os.makedirs(opt.checkpoints_dir + opt.name)
    # get training options
    train_dataset, val_dataset = getDataset()
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)

    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.val_interval, shuffle=False, num_workers=2,
                                                   pin_memory=True)
    dataset_size = len(train_dataset)

    model = create_model(opt)
    print('The number of training images = %d' % dataset_size)

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        pbar = tqdm(data_loader)
        for i, data in enumerate(pbar):  # inner loop within one epoch
            data = data.to("cuda")
            epoch_iter += opt.batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()
            model.set_input(data)
            model.optimize_parameters()
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / opt.batch_size * 0.005 + 0.995 * optimize_time
            losses = model.get_current_losses()
            pbar.set_postfix(loss = losses)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks(epoch)
            if os.path.isdir(opt.image_result_dir + opt.name+'/' + str(epoch)) == False:
                os.makedirs(opt.image_result_dir + opt.name+'/' + str(epoch))
            for test_i,valdata in enumerate(val_data_loader):
                valdata = valdata[0:1,:,:,:]
                model.set_input(valdata,train=False)
                model.test()  # run inference
                visuals = model.get_current_visuals()
                image = torch.cat((visuals['input_hdr'], visuals['output_hdr']),dim=-1)
                save_image(image.detach().cpu(),
                           opt.image_result_dir + opt.name+'/' + str(epoch) + '/' + str(test_i) + '.PNG')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
