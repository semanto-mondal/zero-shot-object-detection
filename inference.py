import torch
import torch.nn.functional as F
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from detectors import FasterRCNNExtractor


def load_detector(device):
    """Loads the Faster R-CNN feature extractor model."""
    model = FasterRCNNExtractor(pretrained=True).to(device)
    model.eval()
    return model


def prepare_image(image_tensor):
    """
    Applies the internal transformation used by Faster R-CNN.
    Converts raw image tensor to GeneralizedRCNNTransform format (ImageList).
    """
    transform = GeneralizedRCNNTransform(
        min_size=800, max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
    image_tensor = image_tensor.to(torch.float32)

    # Remove batch dimension: image_tensor shape (1, 3, H, W) -> (3, H, W)
    image_3d = image_tensor.squeeze(0)

    # Pass a list of 3D tensors to transform
    image_list, _ = transform([image_3d])  # (ImageList, targets)
    return image_list


def extract_roi_features(model, image_list):
    """
    Extracts ROI features from the input image list using Faster R-CNN's backbone + RPN + ROIAlign.
    """
    with torch.no_grad():
        roi_feats, proposals = model(image_list)
    return roi_feats, proposals


def match_regions_to_classes(region_feats, class_embeds, class_list, threshold=None):
    """
    Matches each region feature to the most similar class embedding using cosine similarity.
    Optionally filters with a similarity threshold.
    """
    similarities = F.cosine_similarity(region_feats.unsqueeze(1), class_embeds.unsqueeze(0), dim=2)  # (num_regions, num_classes)

    top_vals, top_indices = similarities.max(dim=1)

    if threshold is not None:
        pred_classes = [
            class_list[i] if score >= threshold else "unknown"
            for i, score in zip(top_indices, top_vals)
        ]
    else:
        pred_classes = [class_list[i] for i in top_indices]

    return pred_classes, top_vals


def extract_global_feature(model, image_list):
    """
    Extracts a global image feature using the backbone's output and average pooling.
    This represents the entire image instead of individual regions.
    """
    with torch.no_grad():
        features = model.backbone(image_list.tensors)  # dict of feature maps
        pooled = torch.nn.functional.adaptive_avg_pool2d(features['0'], (1, 1))  # (B, C, 1, 1)
        global_feat = pooled.view(pooled.size(0), -1)  # flatten: (B, C)
    return global_feat
