import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CenterNetLoss(nn.Module):
    def __init__(self, nClass=20, **kwargs):
        """Constructor of class CenterNetLoss

        Args:
            nClass (int): number of classes of VOC
        """
        super(CenterNetLoss, self).__init__()
        self.focal_alpha = 2.0 if 'alpha' not in kwargs else kwargs['alpha']  # focal loss hyper param
        self.focal_beta = 4.0 if 'beta' not in kwargs else kwargs['beta']  # focal loss hyper param
        self.lambda_size = 0.1 if 'lambda_size' not in kwargs else kwargs['lambda_size']
        self.lambda_offset = 1.0 if 'lambda_offset' not in kwargs else kwargs['lambda_offset']
        self.nClass = nClass
        self.debug = False if 'debug' not in kwargs else kwargs['debug']
        self.device_ = device if not self.debug else 'cpu'

    def focal_loss(self, input, target, mask_true_center):
        """Focal loss

        Args:
            input (torch.Tensor): predicted heat map, shape (N, nClass, H, W)
            target (torch.Tensor): target heat map, shape (N, nClass, H, W)
            mask_true_center (torch.Tensor): bool tensor denotes location of true center in target,
                (N, nClass, H, W)
        """
        assert 0.0 <= input.min().item() <= 1.0, "Remember to activate heatmap with sigmoid"
        assert 0.0 <= input.max().item() <= 1.0, "Remember to activate heatmap with sigmoid"

        nCenters = mask_true_center.sum()  # scalar, number of centers (i.e. objects) in this mini-batch

        pred_center_prob = input[mask_true_center]  # (nCenters)
        # Note: when slice the 4D tensor `input` with another 4D tensor - `mask_true_center`,
        # the result is a 1D tensor
        loss_center = (1.0 - pred_center_prob)**self.focal_alpha * torch.log(pred_center_prob)  # (nCenters)

        mask_non_center = torch.logical_not(mask_true_center)  # (N, nClass, H, W)
        #  similar to computing loss_center, use mask_non_center to extract probability of non center pixel
        # (cont): from input (Y^ in Eq.6) and from target (Y in Eq.6). Then use them to compute loss
        # (cont): for non center pixels (2nd case of Eq. 6)
        pred_noncenter_prob = input[mask_non_center]  # shape (N*nClass*H*W - nCenters)
        target_noncenter_prob = target[mask_non_center]  # shape (N*nClass*H*W - nCenters)
        loss_noncenter = (1-target_noncenter_prob)**self.focal_beta * (pred_noncenter_prob)**self.focal_alpha * torch.log(1.0 - pred_noncenter_prob) # shape (N*nClass*H*W - nCenters) -

        focal_loss = (loss_center.sum() + loss_noncenter.sum()) / (-1 * nCenters)  # sum all elements in loss_center & in loss_noncenter, then divide by nCenters
        return focal_loss

    def forward(self, input, target):
        """Main function of loss. This one is called by '()' operator

        Args:
            input (torch.Tensor): model's prediction, shape (N, nClass+4, H, W)
            target (torch.Tensor): model's label, shape (N, nClass+4, H, W)
        """
        mask_true_center = target[:, :self.nClass] > 0.999  # bool tensor of shape (N, nClass, H, W)
        # Note: `mask_true_center` is a batch of N masks (binary grids). Each mask has nClass channels denote
        # location of centers of different classes in each image.
        focal_loss = self.focal_loss(input[:, :self.nClass], target[:, :self.nClass], mask_true_center)

        # regression loss including offset loss & size loss
        size_loss = torch.tensor([0.0], dtype=torch.float).to(self.device_)
        offset_loss = torch.tensor([0.0], dtype=torch.float).to(self.device_)
        for b_idx in range(input.size(0)):  # iterate through each image in the mini-batch
            collapsed_mask_true_center = torch.zeros(input.size(2), input.size(3),
                                                     dtype=torch.bool).to(self.device_)  # shape (H, W)
            # : for each nClass-channel heat map in `mask_true_center`, convert it into a single-channel mask
            #  cont: by accumulating True value across the channel dimension.
            #  cont: Such accumulation can be done by OR operator. Helpful function: torch.logical_or()
            #  cont: The result is stored in `collapsed_mask_true_center`
            for ch_idx in range(self.nClass):
                collapsed_mask_true_center = torch.logical_or(collapsed_mask_true_center, input[b_idx][ch_idx])

            # extract predicted regression values (size & offset) using collapsed_mask_true_center
            pred_regress = input[b_idx, self.nClass:, collapsed_mask_true_center]  # (4, nCenters_for_this_image)
            # similar to pred_regress, use collapsed_mask_true_center to extract target regression values
            target_regress = target[b_idx, self.nClass:, collapsed_mask_true_center]  # shape (4, nCenters_for_this_image) -

            size_loss += torch.abs(pred_regress[:2] - target_regress[:2]).sum()
            # similar to size_loss, compute offset loss using pred_regress and target_regress
            offset_loss += torch.abs(pred_regress[2:] - target_regress[2:]).sum()  #

        # normalize size_loss & offset_loss with total number of centers in the mini-batch
        nCenters = collapsed_mask_true_center.sum() # compute the number of centers present in this mini-batch
        size_loss /= nCenters
        offset_loss /= nCenters
        if self.debug:
            return focal_loss, size_loss, offset_loss
        else:
            return focal_loss + self.lambda_size * size_loss + self.lambda_offset * offset_loss

