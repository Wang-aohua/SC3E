import numpy as np
import math
from scipy import stats

def eval_contrast(x,roi,roi2,numbin=100):
    x=np.array(x.squeeze().cpu())
    roi = np.array(roi['mask'])
    roi2 = np.array(roi2['mask'])
    roi_=[]
    back_ = []
    for i in range(x.shape[-2]):
        for j in range(x.shape[-1]):
            if roi[i][j] == 1:
                roi_.append(x[i][j])
            if roi2[i][j] == 1:
                back_.append(x[i][j])
    TheMean = abs(np.mean(roi_) - np.mean(back_))
    TheVar = math.sqrt((np.var(roi_) + np.var(back_))/2)

    res_roi = stats.relfreq(roi_, numbin)
    pdf_value_roi = res_roi.frequency
    res_back = stats.relfreq(back_, numbin)
    pdf_value_back = res_back.frequency
    ovl = 0
    for i in range(len(pdf_value_roi)):
        ovl += min(pdf_value_roi[i], pdf_value_back[i])
    gCNR = 1 - ovl

    cnr = 20 * math.log10(TheMean / TheVar)
    cr = -20 * math.log10(np.mean(roi_)/np.mean(back_))

    return cnr,cr,gCNR