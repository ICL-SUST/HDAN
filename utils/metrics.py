import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion


def transformation(im1, im2, tid):
    im1 = im1 == tid
    im2 = im2 == tid
    im1 = np.asarray(im1).astype(np.float32)
    im2 = np.asarray(im2).astype(np.float32)
    return im1, im2


def dice(im1, im2, tid):
    im1 = im1 == tid
    im2 = im2 == tid
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    # im3=im1.astype(np.float32)
    # print(im3.shape)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc = 2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc


def extract_points(mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary mask (H, W) to a set of 2D points.
    Returns shape (N, 2), where each row is (y, x).
    """
    return np.argwhere(mask > 0)


def modified_hausdorff_distance(A: np.ndarray, B: np.ndarray, tid:int) -> float:
    """
    Compute Modified Hausdorff Distance (MHD) for batches of binary masks,
    and return the mean MHD across the entire batch and all channels.

    Parameters:
        A: ndarray of shape (B, L, H, W) - batch of masks
        B: ndarray of shape (B, L, H, W) - batch of masks (same shape as A)

    Returns:
        mean_mhd: float, mean MHD over all batches and channels
    """
    A, B = transformation(A, B, tid)
    A = np.asarray(A, dtype=np.uint8)
    B = np.asarray(B, dtype=np.uint8)

    assert A.shape == B.shape, "A and B must have the same shape"
    B_size, L, H, W = A.shape

    mhd_list = []

    for b in range(B_size):
        for l in range(L):
            pts_A = extract_points(A[b, l])
            pts_B = extract_points(B[b, l])

            if pts_A.size == 0 and pts_B.size == 0:
                mhd = 0.0
            elif pts_A.size == 0 or pts_B.size == 0:
                mhd = np.inf
            else:
                dist_matrix = cdist(pts_A, pts_B)
                min_A_to_B = np.mean(np.min(dist_matrix, axis=1))
                min_B_to_A = np.mean(np.min(dist_matrix, axis=0))
                mhd = max(min_A_to_B, min_B_to_A)

            mhd_list.append(mhd)

    mean_mhd = np.mean(mhd_list)
    return mean_mhd


def extract_surface_3d(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=structure)
    surface = mask ^ eroded
    return np.argwhere(surface)


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.sum(pred & gt)
    union = np.sum(pred) + np.sum(gt)
    return 1.0 if union == 0 else 2.0 * inter / union


def jaccard_index(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.sum(pred & gt)
    union = np.sum(pred | gt)
    return 1.0 if union == 0 else inter / union


def average_surface_distance_3d(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    surface_pred = extract_surface_3d(pred)
    surface_gt = extract_surface_3d(gt)

    if surface_pred.size == 0 or surface_gt.size == 0:
        return None

    surface_pred = surface_pred * spacing
    surface_gt = surface_gt * spacing

    dist_pred_to_gt = np.min(cdist(surface_pred, surface_gt), axis=1).mean()
    dist_gt_to_pred = np.min(cdist(surface_gt, surface_pred), axis=1).mean()
    return (dist_pred_to_gt + dist_gt_to_pred) / 2.0


def hd95_3d(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    surface_pred = extract_surface_3d(pred)
    surface_gt = extract_surface_3d(gt)

    if surface_pred.size == 0 or surface_gt.size == 0:
        return None

    surface_pred = surface_pred * spacing
    surface_gt = surface_gt * spacing

    d_pred_to_gt = np.min(cdist(surface_pred, surface_gt), axis=1)
    d_gt_to_pred = np.min(cdist(surface_gt, surface_pred), axis=1)

    return max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))


def ASD(pre_batch, gt_batch, tid):
    pred_batch, gt_batch = transformation(pre_batch, gt_batch, tid)
    pred_batch = np.asarray(pred_batch, dtype=bool)
    gt_batch = np.asarray(gt_batch, dtype=bool)
    assert pred_batch.shape == gt_batch.shape, "The predicted and true masks must have the same shape."
    B = pred_batch.shape[0]
    asd_list = []
    spacing = (1.0, 1.0, 1.0)
    for b in range(B):
        pred = pred_batch[b]
        gt = gt_batch[b]
        asd_val = average_surface_distance_3d(pred, gt, spacing)
        if asd_val is not None:
            asd_list.append(asd_val)
    return np.mean(asd_list)


def compute_segmentation_metrics(pred_batch: np.ndarray, gt_batch: np.ndarray,tid:int, spacing=(1.0, 1.0, 1.0), ):
    pred_batch,gt_batch=transformation(pred_batch,gt_batch,tid)
    pred_batch = np.asarray(pred_batch, dtype=bool)
    gt_batch = np.asarray(gt_batch, dtype=bool)
    assert pred_batch.shape == gt_batch.shape, "The predicted and true masks must have the same shape."
    B = pred_batch.shape[0]

    dice_list, jaccard_list, asd_list, hd95_list = [], [], [], []

    for b in range(B):
        pred = pred_batch[b]
        gt = gt_batch[b]

        dice_list.append(dice_coefficient(pred, gt))
        jaccard_list.append(jaccard_index(pred, gt))

        asd_val = average_surface_distance_3d(pred, gt, spacing)
        if asd_val is not None:
            asd_list.append(asd_val)

        hd95_val = hd95_3d(pred, gt, spacing)
        if hd95_val is not None:
            hd95_list.append(hd95_val)

    metrics = {
        "Dice": np.mean(dice_list),
        "Jaccard": np.mean(jaccard_list),
        "ASD": np.mean(asd_list) if len(asd_list) > 0 else 0.0,
        "HD95": np.mean(hd95_list) if len(hd95_list) > 0 else 0.0
    }

    return metrics


if __name__ == '__main__':
    A = np.zeros((2, 1, 5, 5), dtype=np.uint8)
    B = np.zeros((2, 1, 5, 5), dtype=np.uint8)

    # batch 1
    A[0, 0, 1, 1] = 1
    A[0, 0, 3, 3] = 1
    B[0, 0, 1, 2] = 1
    B[0, 0, 3, 4] = 1

    # batch 2
    A[1, 0, 2, 2] = 1
    B[1, 0, 2, 2] = 1

    mhd_mean = modified_hausdorff_distance(A, B)
    print(f"Mean Modified Hausdorff Distance: {mhd_mean}")

    pred = np.zeros((2, 5, 5, 5), dtype=np.uint8)
    gt = np.zeros((2, 5, 5, 5), dtype=np.uint8)

    pred[0, 1:4, 1:4, 1:4] = 1
    gt[0, 2:5, 2:5, 2:5] = 1

    spacing = (2.5, 1.0, 1.0)
    metrics = compute_segmentation_metrics(pred, gt, spacing)
    print(metrics)
    print("well done")


