import os
import matplotlib.pyplot as plt
from options.test_options import TestOptions
from util.data_process import pre_process
from datasets.dataset import getTestDataset
from roipoly import RoiPoly, MultiRoi
from scipy.io import savemat
import torch

if __name__ == '__main__':
    opt = TestOptions().parse()
    draw_i = 0
    testSet = getTestDataset()
    assert 0 <= draw_i < testSet.__len__()
    testdata = testSet.__getitem__(draw_i)
    testdata = torch.tensor(testdata).unsqueeze(dim=0)
    image = pre_process(testdata,DR=60)
    image = image.squeeze().cpu()
    plt.imshow(image, cmap='gray')
    multiroi_named = MultiRoi(roi_names=['ROI', 'Background'], color_cycle=['r', 'b', 'b'])

    # Draw all ROIs
    plt.imshow(image, cmap="gray")
    roi_names = []
    masks = []
    for name, roi in multiroi_named.rois.items():
        roi.display_roi()
        roi_names.append(name)
        mask = roi.get_mask(image)
        masks.append(mask)
    plt.show()
    if draw_i >= 0:
        i = 1
        while os.path.exists(opt.roi_path + str(draw_i) + "_roi_" + str(i) + "_1.mat"):
            i += 1
        savemat(opt.roi_path + str(draw_i) + "_roi_" + str(i) + "_roi.mat", {'mask':masks[0].astype(int)})
        savemat(opt.roi_path + str(draw_i) + "_roi_" + str(i) + "_back.mat", {'mask':masks[1].astype(int)})
