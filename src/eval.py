def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom