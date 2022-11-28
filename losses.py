from enum import IntEnum, auto
from functools import partial
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logit
from torch.nn.modules.loss import _Loss


class DistanceTransformNormalize(IntEnum):
    NONE = auto()
    L2 = auto()
    ZERO_MEAN_L2 = auto()
    MINMAX = auto()


def sigmoid_focal_loss_with_logits(
        output: torch.Tensor,
        target: torch.Tensor,
        offset: float = 0.,
        coeff: float = 1.0,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        sum_reduce_dim: Optional[Sequence] = None,
        normalized: bool = False,
        eps: float = 1e-6,
) -> torch.Tensor:
    target = target.type(output.type())

    # logits = target * output + (1 - target) * (- output)
    logits = (2 * target - 1) * output
    focal_term = torch.sigmoid(-coeff * (logits - offset))
    bce = -F.logsigmoid(logits)

    if alpha is not None:
        # if alpha is a tensor, assume the second dimension
        # of output/target is for classes
        if torch.is_tensor(alpha):
            alpha = alpha[None, -1]
        focal_term = focal_term * (alpha * target + (1 - alpha) * (1 - target))

    loss = focal_term * bce

    if sum_reduce_dim is not None:
        loss = loss.sum(dim=sum_reduce_dim, keepdim=True)
        if normalized:
            norm_factor =\
                focal_term.sum(dim=sum_reduce_dim,
                               keepdim=True).clamp_min(eps)
            loss /= norm_factor
        loss = loss.mean()
    else:
        loss = loss.mean()
        if normalized:
            norm_factor = focal_term.sum().clamp_min(eps)
            loss /= norm_factor

    return loss


class SigmoidBinaryFocalLoss(_Loss):
    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        thres: float = 0.9,
        coeff: Optional[float] = None,
        sum_reduce_dim: Optional[tuple] = None,
        normalized: bool = False,
    ):
        """Compute modified version of binary focal loss

        Parameters
        ----------
        alpha : Optional[Union[float, torch.Tensor]], optional
            weight for positive class of focal term.
            should be a value between 0 and 1, by default None
        thres : float, optional
            probability for focal terms to suppressed to near zero,
            by default 0.9
        coeff : Optional[float], optional
            sensitivity of suppressing focal terms around thres.
            determined autoatically if None is given, by default None
        sum_reduce_dim : Optional[tuple], optional
            dimensions to normalize forcal terms over.
            a tuple of dimensions excluding batch and channel are a likely
            choice if normalized is True, by default None
        normalized : bool, optional
            whether to normalize focal terms or not, by default False
        """
        super().__init__()

        if coeff is None and thres == 0.5:
            coeff = 1.
        elif coeff is None:
            width_rate = 0.8
            if 0. < thres < 0.5:
                width = width_rate * thres
                intersection = thres - width
            elif 0.5 < thres < 1.0:
                width = width_rate * (1 - thres)
                intersection = thres + width

            intersection_logit = logit(intersection)
            coeff = (intersection_logit /
                     (intersection_logit - logit(thres)))

        offset = logit(thres)

        self.focal_loss_fn = partial(
            sigmoid_focal_loss_with_logits,
            alpha=alpha,
            offset=offset,
            coeff=coeff,
            sum_reduce_dim=sum_reduce_dim,
            normalized=normalized
        )

    def forward(self, label_input, label_target):
        loss = self.focal_loss_fn(label_input.squeeze(), label_target)
        return loss


class CorrelationCoefficientLoss(_Loss):
    def __init__(self, from_logits: bool = True):
        """Compute correlation coefficient loss

        Parameters
        ----------
        from_logits : bool, optional
            whether output values are given by logits, by default True
        """
        super().__init__()
        self.from_logits = from_logits
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def _single_class(self, output, target):
        output = output.squeeze()
        if self.from_logits:
            output = torch.sigmoid(output)

        target = target.type(output.type())

        output = output.view(output.shape[0], -1)
        target = target.view(target.shape[0], -1)
        mcc = self.cos(output - output.mean(dim=1, keepdim=True),
                       target - target.mean(dim=1, keepdim=True))
        loss = (1. - mcc.mean()) / 2.
        return loss

    def forward(self, output, target):
        """calculate loss value

        Parameters
        ----------
        output :torch.Tensor
            model output with dimensions [B, H, W] or [B, C, H, W]
        target : torch.Tensor
            ground truths with the same dimensions of output

        Returns
        -------
        torch.Tensor
            loss value

        Raises
        ------
        ValueError
            the case output dimension is neither 3 nor 4
        """
        if output.dim() == 3:
            output = output[:, None, :, :]
            if target.dim() == 3:
                target = target[:, None, :, :]
            nclasses = 1
        elif output.dim() == 4:
            nclasses = output.shape[1]
        else:  # errornuous case
            raise ValueError("the tensor net_output need to have"
                             f"3 or 4 dimesion, not {output.dim()}")

        scores = torch.tensor(tuple(
            self._single_class(output[:, i, :, :],
                               target[:, i, :, :])
            for i in range(nclasses)
        ), device=output.device)
        return scores.mean()


def compute_edts(
    segmentation: torch.Tensor,
    normalize=DistanceTransformNormalize.NONE,
    log_scale: bool = False,
    margin: float = 0.
):
    """compute weight maps using distance transformation

    Parameters
    ----------
    segmentation : torch.Tensor
        ground truths
    normalize : _type_, optional
        the way to normalize weight maps,
        by default DistanceTransformNormalize.NONE
    log_scale : bool, optional
        wheter to apply log over distance transformation, by default False
    margin : float, optional
        margin to separate positive and negative weights, by default 0.

    Returns
    -------
    torch.Tensor
        computed weight maps
    """
    masks = segmentation.to('cpu').detach().numpy()
    res = np.zeros_like(masks, dtype=np.float32)
    for i in range(masks.shape[0]):
        posmask = masks[i] > 0
        posratio = posmask.sum() / posmask.size
        negmask = ~posmask
        negratio = 1. - posratio

        if posratio == 0. or negratio == 0.:
            continue

        posbound = cv2.distanceTransform(
            posmask.astype(np.uint8),
            distanceType=cv2.DIST_L2,
            maskSize=cv2.DIST_MASK_PRECISE
        )
        negbound = cv2.distanceTransform(
            negmask.astype(np.uint8),
            distanceType=cv2.DIST_L2,
            maskSize=cv2.DIST_MASK_PRECISE
        )
        if log_scale:
            eps = 1e-6
            posbound[posmask] = np.log(posbound[posmask] + eps)
            negbound[negmask] = np.log(negbound[negmask] + eps)

        posbound[posmask] += margin
        negbound[negmask] += margin

        if normalize == DistanceTransformNormalize.MINMAX:
            posmax = posbound.max()
            if posmax != 0.:
                posbound[posmask] /= posmax
            negmax = negbound.max()
            if negmax != 0.:
                negbound[negmask] /= negmax
        elif normalize == DistanceTransformNormalize.ZERO_MEAN_L2:
            psum = posbound[posmask].sum()
            nsum = negbound[negmask].sum()
            posbound[posmask] *= nsum
            negbound[negmask] *= psum

        res[i][posmask] = posbound[posmask]
        res[i][negmask] = -negbound[negmask]

        if (normalize == DistanceTransformNormalize.L2
                or normalize == DistanceTransformNormalize.ZERO_MEAN_L2):
            res[i] /= np.sqrt(np.sum(np.square(res[i])))

    return torch.from_numpy(res)


def single_class_modified_BD_loss(
    net_output: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = False,
    log_scale: bool = False,
    normalize:
        DistanceTransformNormalize = DistanceTransformNormalize.ZERO_MEAN_L2,
    cosine_similarity: bool = False,
    margin: float = 0.,
    eps: float = 1e-6,
) -> torch.Tensor:
    net_output = net_output.squeeze()
    if from_logits:
        net_output = torch.sigmoid(net_output)
    bound = compute_edts(target,
                         normalize=normalize,
                         log_scale=log_scale,
                         margin=margin)
    bound = bound.to(net_output.device)

    net_output = net_output.view(net_output.shape[0], -1)
    bound = bound.view(bound.shape[0], -1)
    mean_centered = net_output - net_output.mean(dim=1, keepdim=True)
    if cosine_similarity:
        score = nn.CosineSimilarity(dim=1, eps=eps)(mean_centered, bound)
        return (1. - score.mean()) / 2.
    else:
        score = torch.mul(mean_centered, bound).mean()
        return -score


class ModifiedBDLoss(_Loss):
    def __init__(self,
                 from_logits: bool = True,
                 weighted: bool = True,
                 cosine_similarity: bool = False,
                 log_scale: bool = False,
                 margin: float = 0.,
                 eps: float = 1e-6):
        """Compute modified version of BD loss

        Parameters
        ----------
        from_logits : bool, optional
            whether output values are given by logits, by default True
        weighted : bool, optional
            whether weight maps are weighted by the portion of positive
            and negative pixels, by default True
        cosine_similarity : bool, optional
            whether loss is calculated with cosine similarity score,
            by default False
        log_scale : bool, optional
            wheter to apply log over distance transformation, by default False
        margin : float, optional
            margin to separate positive and negative weights, by default 0.
        eps : float, optional
            a small value used to stabilize calculation, by default 1e-6
        """
        super().__init__()
        self.from_logits = from_logits
        if weighted:
            self.normalize = DistanceTransformNormalize.ZERO_MEAN_L2
        else:
            self.normalize = DistanceTransformNormalize.NONE
        self.cosine_similarity = cosine_similarity
        self.log_scale = log_scale
        self.margin = margin
        self.eps = eps

    def forward(self, net_output, target):
        """
        net_output: [B, H, W] or [B, C, H, W]
        target: ground truth, shape: [B, H, W] or [B, C, H, W]
        """
        if net_output.dim() == 3:
            net_output = net_output[:, None, :, :]
            if target.dim() == 3:
                target = target[:, None, :, :]
            nclasses = 1
        elif net_output.dim() == 4:
            nclasses = net_output.shape[1]
        else:  # errornuous case
            raise ValueError("the tensor net_output need to have"
                             f"3 or 4 dimesions, not {net_output.dim()}")

        scores = torch.tensor(tuple(
            single_class_modified_BD_loss(
                net_output[:, i, :, :], target[:, i, :, :],
                from_logits=self.from_logits,
                log_scale=self.log_scale,
                normalize=self.normalize,
                cosine_similarity=self.cosine_similarity,
                margin=self.margin,
                eps=self.eps
            )
            for i in range(nclasses)
        ), device=net_output.device)
        return scores.mean()


def single_class_BD_focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    offset: float = 0.,
    coeff: float = 1.0,
    sum_reduce_dim: Optional[Sequence] = (-2, -1),
    normalized: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    output = output.squeeze()
    target = target.type(output.type())

    # logits = target * output + (1 - target) * (- output)
    logits = (2 * target - 1) * output
    focal_term = torch.sigmoid(-coeff * (logits - offset))
    bce = -F.logsigmoid(logits)

    bound = compute_edts(target,
                         normalize=DistanceTransformNormalize.MINMAX,
                         log_scale=True,
                         margin=0.)
    bound = bound.to(output.device)
    with torch.no_grad():
        bound = torch.abs(bound)

    focal_term = focal_term * bound
    loss = focal_term * bce

    if sum_reduce_dim is not None:
        loss = loss.sum(dim=sum_reduce_dim, keepdim=True)
        if normalized:
            norm_factor =\
                focal_term.sum(dim=sum_reduce_dim,
                               keepdim=True).clamp_min(eps)
            loss /= norm_factor
        loss = loss.mean()
    else:
        loss = loss.mean()
        if normalized:
            norm_factor = focal_term.sum().clamp_min(eps)
            loss /= norm_factor

    return loss


class BDBinaryFocalLoss(_Loss):
    def __init__(
        self,
        thres: float = 0.9,
        coeff: Optional[float] = None,
        normalized=False,
    ):
        """Compute BD bianry focal loss

        Parameters
        ----------
        thres : float, optional
            probability for focal terms to suppressed to near zero,
            by default 0.9
        coeff : Optional[float], optional
            sensitivity of suppressing focal terms around thres.
            determined autoatically if None is given, by default None
        normalized : bool, optional
            whether to normalize focal terms or not, by default False
        """
        super().__init__()

        if coeff is None and thres == 0.5:
            coeff = 1.
        elif coeff is None:
            width_rate = 0.8
            if 0. < thres < 0.5:
                width = width_rate * thres
                intersection = thres - width
            elif 0.5 < thres < 1.0:
                width = width_rate * (1 - thres)
                intersection = thres + width

            intersection_logit = logit(intersection)
            coeff = (intersection_logit /
                     (intersection_logit - logit(thres)))

        offset = logit(thres)

        self.bd_focal_loss_fn = partial(
            single_class_BD_focal_loss_with_logits,
            offset=offset,
            coeff=coeff,
            sum_reduce_dim=(-2, -1),
            normalized=normalized
        )

    def forward(self, net_output, target):
        """
        net_output: [B, H, W] or [B, C, H, W]
        target: ground truth, shape: [B, H, W] or [B, C, H, W]
        """
        if net_output.dim() == 3:
            net_output = net_output[:, None, :, :]
            if target.dim() == 3:
                target = target[:, None, :, :]
            nclasses = 1
        elif net_output.dim() == 4:
            nclasses = net_output.shape[1]
        else:  # errornuous case
            raise ValueError("the tensor net_output need to have"
                             f"3 or 4 dimesions, not {net_output.dim()}")

        scores = torch.tensor(tuple(
            self.bd_focal_loss_fn(net_output[:, i, :, :], target[:, i, :, :])
            for i in range(nclasses)
        ), device=net_output.device)
        return scores.mean()
