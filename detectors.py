import torch
import torchvision

class FasterRCNNExtractor(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        # Backbone and RPN from the model
        self.backbone = model.backbone
        self.rpn = model.rpn
        
        # ROI pooling and box head (the fully connected layers)
        self.roi_pooler = model.roi_heads.box_roi_pool
        self.box_head = model.roi_heads.box_head
        
        # Output feature dimension from box_head
        self.head_out_channels = model.roi_heads.box_head.fc7.out_features

    def forward(self, images):
        features = self.backbone(images.tensors)
        proposals, _ = self.rpn(images, features)
        # Perform ROI Pooling
        pooled_feats = self.roi_pooler(features, proposals, images.image_sizes)
        # Forward pooled features through the box head (fc layers)
        roi_feats = self.box_head(pooled_feats)
        return roi_feats, proposals