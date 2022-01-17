# coding: utf-8
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from cnnkit.preprocess import sub_means, normalize_l1
from bitcv import resize2, letterbox_embed, letterbox_embed_optmized
from bitcv import read_image_zh, put_text, correct_box_l2o, Box
from pymagic import logI, logF, Timer


def gather_feat(feat, idx):
    feat = tf.reshape(feat, shape=(feat.shape[0], -1, feat.shape[-1]))
    idx = tf.cast(idx, dtype=tf.int32)
    feat = tf.gather(params=feat, indices=idx, batch_dims=1)
    return feat


def pool_nms(heatmap, pool_size=3):
    hmax = keras.layers.MaxPool2D(pool_size=pool_size, strides=1, padding="same")(heatmap)
    keep = tf.cast(tf.equal(heatmap, hmax), tf.float32)
    return hmax * keep


def calc_iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)


def non_max_suppression(detections, threhold):
    unique_class = np.unique(detections[:, -1])
    if len(unique_class) == 0:
        return None
    best_box = []
    # 对种类进行循环，筛选出一定区域内属于同一种类得分最大的框，对种类进行循环可以对每一个类分别进行非极大抑制。
    for c in unique_class:
        cls_mask = detections[:, -1] == c
        detection = detections[cls_mask]
        scores = detection[:, 4]
        # 根据得分对该种类进行从大到小排序。
        arg_sort = np.argsort(scores)[::-1]
        detection = detection[arg_sort]
        while np.shape(detection)[0] > 0:
            # 每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
            best_box.append(detection[0])
            if len(detection) == 1:
                break
            val = calc_iou(best_box[-1], detection[1:])
            detection = detection[1:][val < threhold]
    return np.array(best_box)


class CenterNetDecoder:
    def __init__(self,
                 input_shape,
                 max_boxes,
                 downsample_ratio,
                 score_threshold,
                 nms_threshold=0):
        self.K = max_boxes
        self.input_shape = np.array(input_shape, dtype=np.float32)[:2]
        self.dsr = downsample_ratio
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def __call__(self, image, pred, *args, **kwargs):
        raw_shape = np.array(image.shape, dtype=np.float32)[:2]
        num_classes = pred.shape[-1] - 4
        hmap, off, size = tf.split(value=pred, num_or_size_splits=[num_classes, 2, 2], axis=-1)
        hmap = tf.math.sigmoid(hmap)
        batch_size = hmap.shape[0]
        hmap = pool_nms(hmap)
        scores, inds, clses, ys, xs = self.topK(scores=hmap)

        if off is not None:
            off = gather_feat(feat=off, idx=inds)
            xs = tf.reshape(xs, shape=(batch_size, self.K, 1)) + off[:, :, 0:1]
            ys = tf.reshape(ys, shape=(batch_size, self.K, 1)) + off[:, :, 1:2]
        else:
            xs = tf.reshape(xs, shape=(batch_size, self.K, 1)) + 0.5
            ys = tf.reshape(ys, shape=(batch_size, self.K, 1)) + 0.5

        size = gather_feat(feat=size, idx=inds)
        clses = tf.cast(tf.reshape(clses, (batch_size, self.K, 1)), dtype=tf.float32)
        scores = tf.reshape(scores, (batch_size, self.K, 1))

        values = [
            xs - size[..., 0:1] / 2,
            ys - size[..., 1:2] / 2,
            xs + size[..., 0:1] / 2,
            ys + size[..., 1:2] / 2
        ]

        bboxes = tf.concat(values=values, axis=2)
        detections = tf.concat(values=[bboxes, scores, clses], axis=2)
        detections = self.map_to_original(raw_shape, detections)

        return detections

    def map_to_original(self, raw_shape, detections):
        bboxes, scores, clses = tf.split(value=detections, num_or_size_splits=[4, 1, 1], axis=2)
        bboxes, scores, clses = bboxes.numpy()[0], scores.numpy()[0], clses.numpy()[0]
        resize_ratio = raw_shape / self.input_shape

        bboxes[:, 0::2] = bboxes[:, 0::2] * self.dsr * resize_ratio[1]
        bboxes[:, 1::2] = bboxes[:, 1::2] * self.dsr * resize_ratio[0]
        bboxes[:, 0::2] = np.clip(a=bboxes[:, 0::2], a_min=0, a_max=raw_shape[1])
        bboxes[:, 1::2] = np.clip(a=bboxes[:, 1::2], a_min=0, a_max=raw_shape[0])
        score_mask = scores >= self.score_threshold

        bboxes = self.mask_reshape(bboxes, np.tile(score_mask,(1, 4)))
        scores = self.mask_reshape(scores,score_mask)
        clses = self.mask_reshape(clses, score_mask)
        detections = np.concatenate([bboxes, scores, clses], axis=-1)
        return detections

    def mask_reshape(self, a, mask):
        return a[mask].reshape(-1, a.shape[-1])

    def topK(self, scores):
        n, h, w, c = scores.shape
        scores = tf.reshape(scores, shape=(n, -1))
        topk_scores, topk_inds = tf.math.top_k(input=scores, k=self.K, sorted=True)
        topk_clses = topk_inds % c
        topk_xs = tf.cast(topk_inds // c % w, tf.float32)
        topk_ys = tf.cast(topk_inds // c // w, tf.float32)
        topk_inds = tf.cast(topk_ys * tf.cast(w, tf.float32) + topk_xs, tf.int32)
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


class CenterNetInfer:
    def __init__(self, model: keras.Model, param: object, **kwargs):
        self.model = model
        self.param = param
        if isinstance(self.model, tf.lite.Interpreter):
            self.model.allocate_tensors()
            output_details = self.model.get_output_details()
            self.output_tendor_id = int(output_details[0]["index"])
            self.input_tendor_id = int(self.model.get_input_details()[0]["index"])
        self.decode = CenterNetDecoder(input_shape=self.param.input_shape,
                                       max_boxes=self.param.max_boxes,
                                       downsample_ratio=self.param.downsample_ratio,
                                       score_threshold=self.param.score_threshold,
                                       nms_threshold=self.param.nms_threshold)

    def crop_and_resize(self, mat):
        return resize2(mat, self.param.input_size, self.param.input_size)

    def check_shapes(self, images: list):
        for img in images:
            if img.shape[0] < self.param.min_size[0] or img.shape[1] < self.param.min_size[1]:
                raise ValueError(F"Wrong image shape: {img.shape}")
            if img.shape[0] > img.shape[1]:
                raise ValueError(F"Wrong image shape: {img.shape}")

    def preprocess(self, image) -> np.ndarray:
        x = np.array([self.crop_and_resize(image)], dtype=np.float32)
        if self.param.channel_means and self.param.channel_means != [0, 0, 0]:
            x = sub_means(x, self.param.channel_means)
        return normalize_l1(x, maxval=255.0, minval=0.0, local=True)

    def _inference(self, image):
        elapsed = []
        timer = Timer()
        x = self.preprocess(image)
        elapsed.append(timer.seconds())
        timer.reset()
        if isinstance(self.model, tf.lite.Interpreter):
            self.model.set_tensor(self.input_tendor_id, x)
            self.model.invoke()
            pred = self.model.get_tensor(self.output_tendor_id)
        else:
            pred = self.model.predict(x, batch_size=1)
        elapsed.append(timer.seconds())
        timer.reset()
        detections = self.decode(image, pred)
        elapsed.append(timer.seconds())
        timer.reset()
        detections = non_max_suppression(detections, self.param.nms_threshold)
        elapsed.append(timer.seconds())
        logI("Elapsed Pre=%.3f Inf=%.3f Dec=%.3f NMS=%.3f" % (elapsed[0], elapsed[1], elapsed[2], elapsed[3]))
        return detections

    def inference(self, path, image=None):
        if image is None:
            image = read_image_zh(path)
        lettersize = max(image.shape[:2])
        if 0:
            resized = letterbox_embed(image, lettersize, lettersize)
        else:
            resized = letterbox_embed_optmized(image, lettersize)
        detections = self._inference(resized)
        if detections is None:
            return
        boxes = detections[:, 0:4]
        for box in boxes:
            xmin, ymin, xmax, ymax = correct_box_l2o(box[0], box[1], box[2], box[3], lettersize, image.shape[:2])
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, image.shape[1]-1)
            ymax = min(ymax, image.shape[0]-1)
            box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
        return detections

    def draw_boxes(self, image, detections, idx2class, fontscale=1):
        if detections is None:
            return image
        labeled = image.copy()
        boxes = detections[:, 0:4]
        scores = detections[:, 4]
        classes = detections[:, 5]
        num_boxes = boxes.shape[0]
        for i in range(num_boxes):
            box = boxes[i]
            info = "%s: %.2f" % (idx2class[int(classes[i])][0], scores[i])
            Box(int(box[0]), int(box[1]), int(box[2]), int(box[3])).draw(labeled, (0, 0, 255), 2)
            put_text(labeled, info, (int(box[0]) + 1, int(box[3]) - 2), (0, 255, 0), 1, fontscale)
        return labeled
