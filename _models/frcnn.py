from torchvision.models.detection.rpn import concat_box_prediction_layers, permute_and_flatten, RegionProposalNetwork, RPNHead
from torchvision.models.detection.roi_heads import fastrcnn_loss, RoIHeads
from torchvision.models.detection.faster_rcnn import _default_anchorgen, TwoMLPHead, FastRCNNPredictor

import torch.nn.functional as F
import torch
from torchvision.ops import MultiScaleRoIAlign

def add_hooks_frcnn(model):
    def get_features(name, model):
        def hook(layer, input, output):
            model.features.setdefault(name,[])
            model.features[name].append(output)
        return hook
    
    model.rpn.head.cls_logits.register_forward_hook(get_features('rpn_cls_logits',model))
    model.rpn.head.bbox_pred.register_forward_hook(get_features('rpn_bbox_pred',model))
    model.rpn.anchor_generator.register_forward_hook(get_features('anchors',model))
    model.roi_heads.box_predictor.cls_score.register_forward_hook(get_features('roi_cls_score',model))
    model.roi_heads.box_predictor.bbox_pred.register_forward_hook(get_features('roi_bbox_pred',model))
    return model
    
def loss_atk_frcnn(model, targets, args, detected):
    losses = dict()
    #rpn
    rpn = RPNLoss()
    layers = ['rpn_cls_logits', 'rpn_bbox_pred', 'anchors']
    feats = dict()
    for l in layers:
        if torch.is_tensor(model.features[l][0]):
            _list = [f[detected] for f in model.features[l]]
        else:
            _list = [model.features[l][0][i] for i in detected]
        feats[l] = _list
        
    boxes, losses_rpn = rpn.compute_loss_atk(feats['rpn_cls_logits'], 
                                             feats['rpn_bbox_pred'],  
                                             feats['anchors'], targets, args)
    
    #roi heads
    feats = dict()
    per = len(model.features['roi_cls_score'][0])/len(targets)
    f = model.features['roi_cls_score'][0].split(int(per))
    _l = torch.concat([f[i] for i in detected],dim=0)
    _list = [_l]
    feats['roi_cls_score'] = _list 
    
    per = len(model.features['roi_bbox_pred'][0])/len(targets)
    f = model.features['roi_bbox_pred'][0].split(int(per))
    _l = torch.concat([f[i] for i in detected],dim=0)
    _list = [_l]
    feats['roi_bbox_pred'] = _list 
        
    proposals = boxes
    roi = RoiLoss()
    losses_roi = roi.compute_loss_atk(feats['roi_cls_score'][0], 
                                      feats['roi_bbox_pred'][0], proposals, targets)
    losses.update(losses_rpn)
    losses.update(losses_roi)
    
    return losses

class RPNLoss(RegionProposalNetwork):
    def __init__(self, *args,**kwargs) -> None:
        rpn_anchor_generator=_default_anchorgen()
        rpn_head=RPNHead(1, 1) #dummy
        rpn_pre_nms_top_n_train=2000
        rpn_pre_nms_top_n_test=1000
        rpn_post_nms_top_n_train=2000
        rpn_post_nms_top_n_test=1000
        rpn_nms_thresh=0.7
        rpn_fg_iou_thresh=0.7
        rpn_bg_iou_thresh=0.3
        rpn_batch_size_per_image=256 #objectness score length per image
        rpn_positive_fraction=0.5
        rpn_score_thresh=0.0
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        super().__init__(rpn_anchor_generator,
                            rpn_head,
                            rpn_fg_iou_thresh,
                            rpn_bg_iou_thresh,
                            rpn_batch_size_per_image,
                            rpn_positive_fraction,
                            rpn_pre_nms_top_n,
                            rpn_post_nms_top_n,
                            rpn_nms_thresh,
                            rpn_score_thresh)
        
        
    def compute_loss_atk(self, objectness, pred_bbox_deltas, anchors, targets, args):
        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
        """

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        
        image_sizes = [args.display_res for i in range(num_images)]
        
        boxes, scores = self.filter_proposals(proposals, objectness, image_sizes, num_anchors_per_level)

        if targets is None:
            raise ValueError("targets should not be None")
        
        labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = self.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        losses = {'objectness': loss_objectness, 'rpn_box': loss_rpn_box_reg}
        
        return boxes, losses
    
    #TODO: individual 
    # def compute_loss(
    #     self, objectness, pred_bbox_deltas, labels, regression_targets):
    #     """
    #     Args:
    #         objectness (Tensor)
    #         pred_bbox_deltas (Tensor)
    #         labels (List[Tensor])
    #         regression_targets (List[Tensor])

    #     Returns:
    #         objectness_loss (Tensor)
    #         box_loss (Tensor)
    #     """
    #     pos_inds = []
    #     sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
    #     for pos_ind, neg_ind in zip(sampled_pos_inds, sampled_neg_inds):

    #     sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
    #     sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

    #     sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    #     objectness = objectness.flatten()

    #     labels = torch.cat(labels, dim=0)
    #     regression_targets = torch.cat(regression_targets, dim=0)

    #     box_loss = (
    #         F.smooth_l1_loss(
    #             pred_bbox_deltas[sampled_pos_inds],
    #             regression_targets[sampled_pos_inds],
    #             beta=1 / 9,
    #             reduction="sum",
    #         )
    #         / (sampled_inds.numel())
    #     )

    #     objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

    #     return objectness_loss, box_loss
    
    # def concat_box_prediction_layers(box_cls, box_regression):
    #     box_cls_flattened = []
    #     box_regression_flattened = []
    #     # for each feature level, permute the outputs to make them be in the
    #     # same format as the labels. Note that the labels are computed for
    #     # all feature levels concatenated, so we keep the same representation
    #     # for the objectness and the box_regression
    #     for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
    #         N, AxC, H, W = box_cls_per_level.shape
    #         Ax4 = box_regression_per_level.shape[1]
    #         A = Ax4 // 4
    #         C = AxC // A
    #         box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
    #         box_cls_flattened.append(box_cls_per_level)

    #         box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
    #         box_regression_flattened.append(box_regression_per_level)
    #     # concatenate on the first dimension (representing the feature levels), to
    #     # take into account the way the labels were generated (with all feature maps
    #     # being concatenated as well)
    #     #Woo. flatten(0,-2)
    #     box_cls = torch.cat(box_cls_flattened, dim=1).flatten(start_dim=1)
    #     box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    #     return box_cls, box_regression   
    
    
class RoiLoss(RoIHeads):
    def __init__(self):
        box_roi_pool=MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2),
        box_head=TwoMLPHead(10 * 10 ** 2, 10)
        box_predictor=FastRCNNPredictor(10, 10) #dummies
        box_score_thresh=0.05
        box_nms_thresh=0.5
        box_detections_per_img=100
        box_fg_iou_thresh=0.5
        box_bg_iou_thresh=0.5
        box_batch_size_per_image=512 #-- rpn_post_nms_top_n_test과 같은 값
        box_positive_fraction=0.25
        bbox_reg_weights=None
        super().__init__(box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img)
        
    def compute_loss_atk(
        self,
        class_logits, box_regression,
        proposals,  # type: List[Tensor]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
        ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
        features (List[Tensor])
        proposals (List[Tensor[N, 4]])
        image_shapes (List[Tuple[H, W]])
        targets (List[Dict])
        """
       
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        losses = {}

        if labels is None:
            raise ValueError("labels cannot be None")
        if regression_targets is None:
            raise ValueError("regression_targets cannot be None")
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        
        losses = {'box_cls': loss_classifier, 'box_reg': loss_box_reg}   
        return losses
    
if __name__ == "__main__":
    roi = RoiLoss()