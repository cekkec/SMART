# eval_tao.py

import math
import os
import glob
import numpy as np
from PIL import Image
from multiprocessing import Pool

def compute_iou(prediction, target):
    # we assume a single image here
    assert target.ndim == 2 or (target.ndim == 3 and target.shape[-1] == 1)
    I = np.logical_and(prediction == 1, target == 1).sum()
    U = np.logical_or(prediction == 1, target == 1).sum()

    if U == 0:
        iou = 1.0
    else:
        iou = float(I) / U
    return iou


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.
    """
    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can't convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    """
    import cv2
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(np.bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def seq_ious(seq, result_path, gt_path):
    ann_fns = sorted(glob.glob(os.path.join(gt_path, seq, "*.png")))
    res_fns = [x.replace(gt_path, result_path) for x in ann_fns]
    anns = [np.array(Image.open(ann_fn)) for ann_fn in ann_fns]
    ress = [np.array(Image.open(res_fn)) for res_fn in res_fns]

    started_ids = set()
    obj_ious = {}
    for ann, res in zip(anns, ress):
        frame_gt_ids = np.setdiff1d(np.unique(ann), [0])
        for id_ in started_ids:
            iou = compute_iou(res == id_, ann == id_)
            f = f_measure(res == id_, ann == id_)
            obj_ious[id_].append((iou, f))
        for id_ in frame_gt_ids:
            if id_ not in started_ids:
                started_ids.add(id_)
                obj_ious[id_] = []

    result_ious = []
    for obj, ious in obj_ious.items():
        if len(ious) > 0:
            result_ious.append(np.mean(ious, axis=0))
    # print(seq, len(result_ious), result_ious)
    return result_ious


def evaluate_tao_results(result_path, imgset_file, gt_path, parallel=True, processes=12):
    seqs = []
    with open(imgset_file) as f:
        for l in f:
            l = l.strip()
            seqs.append(l)

    if parallel:
        with Pool(processes=processes) as pool:
            obj_ious = []
            res = pool.starmap(seq_ious, [(seq, result_path, gt_path) for seq in seqs])
            for r in res:
                obj_ious += r
    else:
        obj_ious = []
        for seq in seqs:
            obj_ious += seq_ious(seq, result_path, gt_path)

    print()
    m = np.mean(obj_ious, axis=0)
    print(result_path)
    print("number of objects:", len(obj_ious))
    print("J-measure (IoU):", m[0])
    print("F-measure:", m[1])
    print("J&F measure:", np.mean(m))
    
    return {
        "result_path": result_path,
        "number_of_objects": len(obj_ious),
        "J-measure": m[0],
        "F-measure": m[1],
        "J&F measure": np.mean(m)
    }
