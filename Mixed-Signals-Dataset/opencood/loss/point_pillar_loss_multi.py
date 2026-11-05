import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.loss.point_pillar_loss import PointPillarLoss


class PointPillarLossMulti(PointPillarLoss):
    def __init__(self, args):
        super().__init__(args)
        self.num_classes = args['num_classes']
        self.name_classes = args['name_classes']
        self.list_loss_dict = list()

    def forward(self, output_dict, target_dict) -> torch.Tensor:
        rm = output_dict['rm']  # (B, num_class, 7*num_anchors, h, w)
        psm = output_dict['psm']  # (B, num_class, num_anchors, h, w)

        targets = target_dict['targets']  # (B, h, w, 7*num_anchors, num_class)
        box_cls_labels = target_dict['pos_equal_one']  # (B, h, w, 2, num_class)

        total_loss = torch.tensor(0.0, requires_grad=True).float().cuda()
        for cls_idx in range(self.num_classes):
            # positives = box_cls_labels[..., cls_idx] > 0
            cls_total_loss, cls_display_loss_dict = self.forward_1cls(
                rm[:, cls_idx], 
                psm[:, cls_idx], 
                targets[..., cls_idx], 
                box_cls_labels[..., cls_idx]
            )
            
            total_loss = total_loss + cls_total_loss
            
            self.list_loss_dict.append(cls_display_loss_dict)

        return total_loss

    def forward_1cls(self, 
                     rm: torch.Tensor, 
                     psm: torch.Tensor, 
                     targets: torch.Tensor, 
                     box_cls_labels: torch.Tensor) -> torch.Tensor:
        """
        forward of PointPillarLoss
        
        Parameters:
        rm : 
            (B, 7 * num_anchor, h, w) - predicted regression map

        psm :
            (B, num_anchor, h, w) - predicted LOGITS map (unnormalized logits)
        
        targets: 
            (B, h, w, 7*num_anchor)

        box_cls_labels:
            (B, h, w, num_anchor)
        """
        # TODO: zero out loss of pixels that is outside of labeled regions

        cls_preds = psm.permute(0, 2, 3, 1).contiguous()

        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), 2,
            dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.cls_loss_func(cls_preds,
                                          one_hot_targets,
                                          weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        if torch.any(positives):
            rm = rm.permute(0, 2, 3, 1).contiguous()
            rm = rm.view(rm.size(0), -1, 7)
            targets = targets.view(targets.size(0), -1, 7)
            box_preds_sin, reg_targets_sin = self.add_sin_difference(rm,
                                                                    targets)
            loc_loss_src =\
                self.reg_loss_func(box_preds_sin,
                                reg_targets_sin,
                                weights=reg_weights)
            reg_loss = loc_loss_src.sum() / rm.shape[0]
            reg_loss *= self.reg_coe
        else:
            reg_loss = torch.tensor(0.0, requires_grad=True).float().cuda()

        total_loss = reg_loss + conf_loss
        display_loss_dict = {
            'total_loss': total_loss.item(),
            'reg_loss': reg_loss.item(),
            'conf_loss': conf_loss.item()
        }

        return total_loss, display_loss_dict
    
    def logging(self, epoch, batch_id, batch_len, writer=None, pbar=None, time_infos: dict = None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = 0
        reg_loss = 0
        conf_loss = 0
        for loss_dict in self.list_loss_dict:
            total_loss += loss_dict['total_loss']
            reg_loss += loss_dict['reg_loss']
            conf_loss += loss_dict['conf_loss']
        
        if writer is not None:
            writer.add_scalar('Regression_loss', reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss', conf_loss,
                            epoch*batch_len + batch_id)

        disp_string = "[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f" \
            " || Loc Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss, conf_loss, reg_loss)
        
        for cls_name, loss_dict in zip(self.name_classes, self.list_loss_dict):
            # disp_string += " | %s: conf %.3f ; loc %.3f" % \
            #     (cls_name, loss_dict['conf_loss'], loss_dict['reg_loss'])
            if writer is not None:
                writer.add_scalar(f'{cls_name}_reg_loss', loss_dict['reg_loss'],
                                epoch*batch_len + batch_id)
            
                writer.add_scalar(f'{cls_name}_conf_loss', loss_dict['conf_loss'],
                                epoch*batch_len + batch_id)
        
        if isinstance(time_infos, dict):
            disp_string += " | data_t %.3f ; forw_t %.3f ; back_t %.3f" \
                % (time_infos['data_time'], time_infos['forward_time'], time_infos['backward_time'])
            
        if pbar is None:
            print(disp_string)
        else:
            pbar.set_description(disp_string)

        self.list_loss_dict = list()
