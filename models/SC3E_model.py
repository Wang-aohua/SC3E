import copy
import numpy as np
import torch
from torch import nn
from models.networks import PatchSampleF,init_net
from util.data_process import crop,pre_process
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss,NCELoss
import util.util as util
import random
import torch.nn.functional as F

class SC3EModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--dr_range', type=str, default='40-100', help='DR range of training images')
        parser.add_argument('--Le_layers', type=str, default='4,8,12,16', help='compute L_E loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netM_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')

        parser.set_defaults(pool_size=0)

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt=opt
        self.loss_names = ['L']
        self.visual_names = ['input_hdr', 'output_hdr', 'input_ldr', 'output_ldr']
        self.nce_layers = [0]
        for i in self.opt.Le_layers.split(','):
            self.nce_layers.append(int(i))
        self.lowest_DR = int(opt.dr_range.split('-')[0])
        self.highest_DR = int(opt.dr_range.split('-')[1])


        if self.isTrain:
            self.model_names = ['N', 'M']
        else:
            self.model_names = ['N']

        self.netN = networks.define_N(opt.input_nc, opt.output_nc, opt.ngf, opt.normN, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netM = PatchSampleF(use_mlp=True, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, nc=opt.netM_nc)
        self.netM = init_net(self.netM, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.mse = torch.nn.MSELoss()
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.featureNCE = NCELoss(self.opt).to(self.device)
            self.optimizer_N = torch.optim.Adam(self.netN.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_N)

    def data_dependent_initialize(self, data, train=True):
        bs_per_gpu = data.size(0) // max(len(self.opt.gpu_ids), 1)
        if train:
            self.set_input(data)
        else:
            self.set_input(data,train=False)
        self.input_hdr = self.input_hdr[:bs_per_gpu]
        self.input_ldr = self.input_ldr[:bs_per_gpu]
        self.forward()
        if self.opt.isTrain:
            self.compute_loss().backward()
            self.optimizer_M = torch.optim.Adam(self.netM.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_M)

    def optimize_parameters(self):
        self.forward()
        self.optimizer_N.zero_grad()
        self.optimizer_M.zero_grad()
        self.loss_L = self.compute_loss()
        self.loss_L.backward()
        self.optimizer_N.step()
        self.optimizer_M.step()

    def set_input(self, input,train=True):
        if train:
            t = np.random.randint(0, (self.highest_DR-self.lowest_DR)/5+1, [1])
            DR = self.lowest_DR + t * 5
        else:
            DR = 60
        input_hdr = pre_process(input[:, 0:1, :, :], DR)
        input_ldr = pre_process(input[:, 0:1, :, :], int(DR*2/3), low=train)

        if train and self.opt.patch_size > 0:
            data = crop(torch.cat((input_hdr,input_ldr),dim=1),self.opt.patch_size)
            input_hdr = data[:,0:1,:,:]
            input_ldr = data[:,1:2,:,:]

        self.input_hdr = input_hdr.to(self.device)
        self.input_ldr = input_ldr.to(self.device)

    def forward(self):
        self.output = torch.cat((self.input_hdr, self.input_ldr), dim=0)
        self.output = self.netN(self.output)
        self.output = torch.clip(self.output,max=1,min=0)
        self.output_hdr = self.output[:self.input_hdr.size(0)]
        self.output_ldr = self.output[self.input_hdr.size(0):]

    def compute_TV_loss(self,):
        k_x = torch.Tensor([[[[-1, 1]]]])
        self.weight_x = nn.Parameter(data=k_x, requires_grad=False).cuda()
        k_y = torch.Tensor([[[[-1], [1]]]])
        self.weight_y = nn.Parameter(data=k_y, requires_grad=False).cuda()
        tv_x = F.conv2d(self.output_hdr, self.weight_x, groups=1, stride=1)
        tv_y = F.conv2d(self.output_hdr, self.weight_y, groups=1, stride=1)
        loss_tv = torch.mean(torch.pow(tv_x, 2)) + torch.mean(torch.pow(tv_y, 2))
        return loss_tv


    def compute_loss(self):
        loss_c_and_e = self.calculate_NCE_loss(self.input_hdr, self.input_ldr, self.output_hdr)
        loss_tv = self.compute_TV_loss()
        loss_mse = self.mse(self.input_ldr, self.output_ldr)

        return loss_c_and_e + loss_mse + 0.1*loss_tv

    def calculate_NCE_loss(self, input_hdr, input_ldr, output_hdr):
        n_layers = len(self.nce_layers)
        feat_v = self.netN(output_hdr, self.nce_layers, encode_only=True)

        split_layer = 1

        feat_pos = self.netN(input_hdr, self.nce_layers, encode_only=True)
        feat_pos2 = self.netN(input_ldr, self.nce_layers, encode_only=True)
        feat_neg = copy.copy(feat_v)
        feat_neg_pool, sample_ids = self.netM(feat_neg, self.opt.num_patches, None)
        feat_pos[split_layer:n_layers] = feat_pos2[split_layer:n_layers]
        feat_pos_pool, sample_ids = self.netM(feat_pos, self.opt.num_patches, None)
        feat_v_pool, _ = self.netM(feat_v, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        layer = 0

        for f_v, f_pos, f_neg, crit, nce_layer in zip(feat_v_pool, feat_pos_pool, feat_neg_pool, self.criterionNCE,
                                                      self.nce_layers):
            if layer < split_layer:
                loss = crit(f_v, f_pos, f_neg, nce_T=self.opt.nce_T)
            else:
                loss = crit(f_v, f_pos, f_neg, nce_T=self.opt.nce_T)
            total_nce_loss += loss.mean()
            layer += 1

        return total_nce_loss / n_layers

    def samplePatch(self,data,loc=None):
        cropsize = [int(data.shape[-1]/8),int(data.shape[-1]/8)]
        if loc ==None:
            w1 = random.randint(0, data.shape[-2] - cropsize[-2])
            h1 = random.randint(0, data.shape[-1] - cropsize[-1])
        else:
            w1 = loc[0]
            h1 = loc[1]
        w2 = w1 + cropsize[-2]
        h2 = h1 + cropsize[-1]
        data = data[:, :, w1:w2, h1:h2]
        data = data.permute(0, 2, 3, 1).flatten(1, 2)
        data = data.flatten(1, 2)
        return data,[w1,h1]