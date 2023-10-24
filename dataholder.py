# coding: utf-8
import os, psutil
import re
import numpy as np
from collections import defaultdict
from tfcnnkit.preprocess import sub_means, normalize_l1
from centernet.gauss import gaussianRadius, createGaussHeatmap
from bitcv import Box, read_image
from bitcv import letterbox_embed, points_enter_letterbox, gray_to_color
from pymagic import logger


def encodeLabel(label, max_boxes, heatmap_shape, downsample_ratio):
    hm = np.zeros(shape=heatmap_shape, dtype=np.float32)
    reg = np.zeros(shape=(max_boxes, 2), dtype=np.float32)
    wh = np.zeros(shape=(max_boxes, 2), dtype=np.float32)
    mask = np.zeros(shape=max_boxes, dtype=np.float32)
    idx = np.zeros(shape=max_boxes, dtype=np.float32)
    for j, item in enumerate(label):
        item[:4] = item[:4] / downsample_ratio
        xmin, ymin, xmax, ymax, category = item
        box_h, box_w = int(ymax - ymin), int(xmax - xmin)
        radius = gaussianRadius(box_h, box_w)
        radius = max(0, int(radius))
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        center = np.array([cx, cy], dtype=np.float32)
        center_int = center.astype(np.int32)
        createGaussHeatmap(hm[:, :, int(category)], center_int, radius)
        reg[j] = center - center_int
        wh[j] = (box_w * 1.0, box_h * 1.0)
        mask[j] = 1
        idx[j] = center_int[1] * hm.shape[0] + center_int[0] # Row-Major based index of element
    return hm, reg, wh, mask, idx


class DataHolder:
    debug_mode = 0
    def __init__(self,
                 train_list_files,
                 input_size,
                 num_classes,
                 batch_size,
                 max_boxes,
                 channle_means=[]):

        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_classes
        # downsample_ratio is input_size / hmap_size
        self.dsr = 4
        self.max_boxes = max_boxes
        self.hmap_size = self.input_size // self.dsr
        self.channel_means = channle_means
        self.annotations = []
        for each in train_list_files:
            with open(each) as fin:
                self.annotations.extend(fin.readlines())
        assert self.annotations
        if 1:
            self.num_batches = len(self.annotations) // self.batch_size
        else:
            self.num_batches = 20 # for test
        self.annotations = self.annotations[:self.num_batches * batch_size]
        np.random.shuffle(self.annotations)
        regex = re.compile("(?:[0-9]+,){4}([0-9]+)")
        counter = defaultdict(lambda:0)
        for line in self.annotations:
            for label in regex.findall(line):
                counter[label] += 1
        logger.info(F"Labels: {dict(counter)}")
        self._x = None

    def cache(self):
        n = len(self.annotations)
        self._hmaps = np.zeros((n, self.hmap_size, self.hmap_size, self.num_classes), np.float32)
        self._regs = np.zeros((n, self.max_boxes, 2), np.float32)
        self._sizes = np.zeros((n, self.max_boxes, 2), np.float32)
        self._masks = np.zeros((n, self.max_boxes), np.float32)
        self._indices = np.zeros((n, self.max_boxes), np.float32)
        self._x = np.zeros((n, self.input_size, self.input_size, 3), np.float32)
        for i, line in enumerate(self.annotations):
            if i and (i % 100 == 0):
                rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
                logger.info(F"{i} images loaded, {round(rss, 2)}GB memory used")
            x, hm, reg, wh, mask, idx = self._decode(line)
            self._x[i] = x
            self._hmaps[i] = hm
            self._regs[i] = reg
            self._sizes[i] = wh
            self._masks[i] = mask
            self._indices[i] = idx
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        logger.info(F"{i} images loaded, {round(rss, 2)}GB memory used")
        return self

    def generateOne(self):
        if self._x is not None:
            rand_orders = np.random.permutation(self._x.shape[0])
            for i in rand_orders:
                x = self._x[i]
                y = self._hmaps[i], self._regs[i], self._sizes[i], self._masks[i], self._indices[i]
                yield x, y
        else:
            np.random.shuffle(self.annotations)
            for line in self.annotations:
                x, hm, reg, wh, mask, idx = self._decode(line)
                yield x, (hm, reg, wh, mask, idx)

    def generateBatch(self):
        if self._x is not None:
            rand_orders = np.random.permutation(self._x.shape[0])
            for i in range(self.num_batches):
                ids = rand_orders[i * self.batch_size:(i + 1) * self.batch_size]
                x = self._x[ids]
                y = self._hmaps[ids], self._regs[ids], self._sizes[ids], self._masks[ids], self._indices[ids]
                yield x, y
        else:
            raise NotImplementedError("Only for cached data!")

    def _parse(self, line):
        items = line.strip().split(" ")
        path = items[0]
        image = read_image(path)
        if len(image.shape) != 3 or image.shape[-1] != 3:
            image = gray_to_color(image)
        boxes = []
        num_of_boxes = len(items) - 1
        assert num_of_boxes > 0
        for i in range(1, len(items)):
            if i <= self.max_boxes:
                tmp = items[i].split(',')
                xmin, ymin, xmax, ymax, category = [int(_) for _ in tmp]
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                assert xmax > 0 and ymax > 0
                boxes.append([xmin, ymin, xmax, ymax, category])
            else:
                # logger.warn(F"Found more than {self.max_boxes} boxes({len(items)-1}): {items[0]}")
                break
        num_padding = self.max_boxes - num_of_boxes
        if num_padding > 0:
            for i in range(num_padding):
                boxes.append([0, 0, 0, 0, -1]) 
        return image, np.array(boxes, dtype=np.float32)

    def _decode(self, line):
        img, boxes = self._parse(line)
        resized = letterbox_embed(img, self.input_size, self.input_size)
        corrected = []
        for xmin, ymin, xmax, ymax, c in boxes:
            if xmax > xmin and ymax > ymin: # not padding data
                xmin, ymin, xmax, ymax = points_enter_letterbox(xmin, ymin, xmax, ymax, resized.shape[0], img.shape)
                if DataHolder.debug_mode:
                    Box(xmin, ymin, xmax, ymax).draw(resized, thickness=2)
            corrected.append([xmin, ymin, xmax, ymax, c])
        if DataHolder.debug_mode:
            showImage(resized)
        label = np.stack(corrected)
        label = label[label[:, 4] != -1]
        hmap_shape = (self.hmap_size, self.hmap_size, self.num_classes)
        hm, reg, wh, mask, idx = encodeLabel(label, self.max_boxes, hmap_shape, self.dsr)
        if self.channel_means:
            resized = sub_means(resized, self.channel_means)
        x = normalize_l1(resized, 255.0, 0, True)
        return x, hm, reg, wh, mask, idx
