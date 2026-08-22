import torch
import torch.nn.functional as F


def create_static_mask(
    gt_bboxes_3d,
    bev_h,
    bev_w,
    bev_range,
    expand_pixels=2,
    device=None,
):
    """
    Creates a binary mask:
      1 = static/background
      0 = dynamic/object region

    Args:
        gt_bboxes_3d: list of GT 3D boxes for one sample
        bev_h, bev_w: BEV feature map size
        bev_range: [x_min, x_max, y_min, y_max]
        expand_pixels: enlarge object region slightly
        device: torch device
    Returns:
        mask: [1, 1, bev_h, bev_w]
    """
    if device is None:
        device = "cpu"

    mask = torch.ones(1, 1, bev_h, bev_w, device=device)

    x_min, x_max, y_min, y_max = bev_range

    if gt_bboxes_3d is None or len(gt_bboxes_3d) == 0:
        return mask

    for bbox in gt_bboxes_3d:
        # expected fields:
        # bbox.center = [x, y, z]
        # bbox.wlh    = [w, l, h]
        cx_world = float(bbox.center[0])
        cy_world = float(bbox.center[1])

        # world -> BEV pixel
        cx = int((cx_world - x_min) / (x_max - x_min) * bev_w)
        cy = int((cy_world - y_min) / (y_max - y_min) * bev_h)

        half_w = int((float(bbox.wlh[0]) / (x_max - x_min)) * bev_w / 2) + expand_pixels
        half_h = int((float(bbox.wlh[1]) / (y_max - y_min)) * bev_h / 2) + expand_pixels

        x0 = max(0, cx - half_w)
        x1 = min(bev_w, cx + half_w)
        y0 = max(0, cy - half_h)
        y1 = min(bev_h, cy + half_h)

        mask[:, :, y0:y1, x0:x1] = 0.0

    return mask


def composite_bev_target(prev_bev_aligned, bev_features, static_mask):
    """
    Build denoising target:
      static pixels  -> aligned previous BEV
      dynamic pixels -> current BEV
    """
    if prev_bev_aligned is None:
        return bev_features

    if static_mask.shape[0] == 1 and bev_features.shape[0] > 1:
        static_mask = static_mask.repeat(bev_features.shape[0], 1, 1, 1)

    return static_mask * prev_bev_aligned + (1.0 - static_mask) * bev_features


def cross_frame_diffusion_loss(
    predicted_x0,
    bev_features,
    prev_bev_aligned,
    gt_bboxes_3d,
    bev_range,
):
    """
    Drop-in replacement for:
        F.mse_loss(predicted_x0, bev_features)

    Args:
        predicted_x0:    [B, C, H, W]
        bev_features:    [B, C, H, W]
        prev_bev_aligned:[B, C, H, W] or None
        gt_bboxes_3d:    list length B, each item = boxes for one sample
        bev_range:       [x_min, x_max, y_min, y_max]

    Returns:
        loss_diffusion
        composite_target
        static_mask
    """
    B, C, H, W = bev_features.shape
    device = bev_features.device

    if prev_bev_aligned is None:
        target = bev_features
        static_mask = torch.ones(B, 1, H, W, device=device)
        return F.mse_loss(predicted_x0, target), target, static_mask

    masks = []
    for b in range(B):
        boxes_b = gt_bboxes_3d[b] if gt_bboxes_3d is not None else []
        mask_b = create_static_mask(
            gt_bboxes_3d=boxes_b,
            bev_h=H,
            bev_w=W,
            bev_range=bev_range,
            device=device,
        )
        masks.append(mask_b)

    static_mask = torch.cat(masks, dim=0)

    target = composite_bev_target(
        prev_bev_aligned=prev_bev_aligned,
        bev_features=bev_features,
        static_mask=static_mask,
    )

    loss = F.mse_loss(predicted_x0, target)
    return loss, target, static_mask