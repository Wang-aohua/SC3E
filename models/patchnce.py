from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_v, feat_pos,feat_neg,nce_T = None): # B*N C1
        num_patches = feat_v.shape[0]   # B*N
        dim = feat_v.shape[1]   # C1
        feat_pos = feat_pos.detach()
        feat_neg = feat_neg.detach()
        # pos logit
        l_pos = torch.bmm(
            feat_v.view(num_patches, 1, -1), feat_pos.view(num_patches, -1, 1))      # B*N 1 C1和B*N C1 1得到B*N 1 1，代表共采样B*N个patch，每个patch的正例的相似度
        l_pos = l_pos.view(num_patches, 1)  # B*N 1，代表共采样B*N个patch，每个patch的正例的相似度
        '''a = torch.exp(l_pos)
        loss = -torch.log(a/(a+1))
        return loss'''
        # l_pos2 = torch.cosine_similarity(feat_v, feat_pos, dim=1)
        # l_pos2 = l_pos2.view(num_patches, 1)
        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_v = feat_v.view(batch_dim_for_bmm, -1, dim)    # 还原为B N C1
        feat_neg = feat_neg.view(batch_dim_for_bmm, -1, dim)    # 还原为B N C1
        npatches = feat_v.size(1)
        l_neg_curbatch = torch.bmm(feat_v, feat_neg.transpose(2, 1))    #B N C1和B C1 N得到B N N,代表2张图N个patch，每个patch有N个负例（实际上是N-1）
        # l_neg_curbatch2 = torch.cosine_similarity(feat_v, feat_neg.transpose(2, 1))
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_v.device, dtype=self.mask_dtype)[None, :, :] #计算负例时的mask，排除掉N个patch里的一个正例
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)   # B*C1 N,共B*C1个patch，每个patch有N个负例（其中有一个正例被mask掉了）
        if nce_T is None:
            out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        else:
            out = torch.cat((l_pos, l_neg), dim=1) / nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_v.device))

        return loss

class NCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, v, v_pos,v_neg,nce_T = None):
        num_patches = v.shape[0]
        dim = v.shape[1]
        v = v.detach()

        # pos logit
        l_pos = torch.bmm(
            v.view(num_patches, 1, -1), v_pos.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        l_neg = torch.bmm(
            v.view(num_patches, 1, -1), v_neg.view(num_patches, -1, 1))
        l_neg = l_neg.view(num_patches, 1)


        if nce_T == None:
            out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        else:
            out = torch.cat((l_pos, l_neg), dim=1) / nce_T

        loss = self.cross_entropy_loss(out, torch.cat((torch.ones(1, dtype=torch.long,
                                                        device=v.device),torch.zeros(out.size(0)-1, dtype=torch.long,
                                                        device=v.device))))

        return loss
