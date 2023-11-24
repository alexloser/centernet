# coding: utf-8
import tensorflow as tf


@tf.function
def focalLoss(gt, pred):
    """Return focal loss of predict heatmaps and ground-truth heatmaps"""
    positive = tf.cast(tf.equal(gt, 1), dtype=tf.float32)
    negtive = tf.cast(tf.less(gt, 1), dtype=tf.float32)
    negtive_scale = tf.math.pow(1 - gt, 4)
    # pred = tf.clip_by_value(pred, 1e-6, 1 - 1e-6)
    num_pos = tf.einsum("nijk->", positive)
    neg_loss = tf.math.log(1 - pred) * tf.math.pow(pred, 2) * negtive_scale * negtive
    neg_loss = tf.einsum("nijk->", neg_loss)
    if num_pos == 0:
        return -neg_loss
    else:
        pos_loss = tf.math.log(pred) * tf.math.pow(1 - pred, 2) * positive
        pos_loss = tf.einsum("nijk->", pos_loss)
        return -(pos_loss + neg_loss) / num_pos


@tf.function
def regL1Loss(y_true, y_pred, mask, index):
    """Return L1 regression loss of prediction and truth,
    if `pred` is predict sizes, `truth` should be ground truth sizes,
    if `pred` is predict regs, `truth` should be ground truth regs.
    `mask` is ground truth label built by dataholder object.
    """
    y_pred = tf.reshape(y_pred, shape=(y_pred.shape[0], -1, y_pred.shape[-1]))
    y_pred = tf.gather(y_pred, indices=tf.cast(index, dtype=tf.int32), batch_dims=1)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), tf.constant([1, 1, 2], dtype=tf.int32))
    loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    return loss / (tf.reduce_sum(mask) + 1e-4)


class CenterNetLoss:
    def __init__(self, num_classes, hmap_weight=1.0, reg_weight=1.0, size_weight=0.1):
        self.num_classes = num_classes
        self.hmap_weight = hmap_weight
        self.reg_weight = reg_weight
        self.size_weight = size_weight

    @tf.function
    def __call__(self, y_pred, y_true):
        true_hmaps, true_regs, true_sizes, masks, indices = y_true
        pred_hmaps, pred_regs, pred_sizes = tf.split(
            value=y_pred, num_or_size_splits=[self.num_classes, 2, 2], axis=-1
        )

        pred_hmaps = tf.clip_by_value(
            t=tf.sigmoid(pred_hmaps), clip_value_min=1e-4, clip_value_max=1.0 - 1e-4
        )

        hmap_loss = focalLoss(true_hmaps, pred_hmaps)
        reg_loss = regL1Loss(y_true=true_regs, y_pred=pred_regs, mask=masks, index=indices)
        size_loss = regL1Loss(y_true=true_sizes, y_pred=pred_sizes, mask=masks, index=indices)

        total_loss = (
            self.hmap_weight * hmap_loss + self.reg_weight * reg_loss + self.size_weight * size_loss
        )

        return hmap_loss, reg_loss, size_loss, total_loss
