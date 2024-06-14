import os
from options.test_options import TestOptions
from models import create_model
import scipy.io as scio
import torch
from datasets.dataset import getTestDataset
from util.eval_contrast import eval_contrast


if __name__ == '__main__':
    opt = TestOptions().parse()
    test_dataset = getTestDataset()
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                   pin_memory=True)
    dataset_size = len(test_dataset)
    model = create_model(opt)
    for i, testdata in enumerate(test_data_loader):
        if i == 0:
            model.data_dependent_initialize(testdata,train=False)
            model.setup(opt)
            model.parallelize()
            model.eval()
        i_roi=1
        while os.path.exists(opt.roi_path+str(i)+"_roi_"+str(i_roi)+"_roi.mat"):
            roi = scio.loadmat(opt.roi_path+str(i)+"_roi_"+str(i_roi)+"_roi.mat")
            roi2 = scio.loadmat(opt.roi_path+str(i)+"_roi_"+str(i_roi)+"_back.mat")
            model.set_input(testdata,train=False)
            model.test()
            visuals = model.get_current_visuals()
            cnr, cr, gCNR = eval_contrast(visuals['input_hdr'], roi, roi2)
            print("input image:{}, {}-th roi\nCR:{}\tCNR:{}\tgCNR:{}".format(i,i_roi,cr,cnr,gCNR))
            cnr, cr, gCNR = eval_contrast(visuals['output_hdr'], roi, roi2)
            print("output image:{}, {}-th roi\nCR:{}\tCNR:{}\tgCNR:{}".format(i, i_roi, cr, cnr, gCNR))
            print("\n")
            i_roi+=1