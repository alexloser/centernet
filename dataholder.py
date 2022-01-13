# coding: utf-8
import os, psutil
import re
import numpy as np
from collections import defaultdict
from cnnkit.preprocess import sub_means, normalize_l1
from centernet.gauss import gaussian_radius, create_gauss_heatmap
from bitcv import Box, show_image, read_image, flip2
from bitcv import letterbox_embed, correct_box_o2l
from pymagic import logW, logI


def decode_label(label, max_boxes, heatmap_shape, downsampling_ratio):
    hm = np.zeros(shape=heatmap_shape, dtype=np.float32)
    off = np.zeros(shape=(max_boxes, 2), dtype=np.float32)
    wh = np.zeros(shape=(max_boxes, 2), dtype=np.float32)
    mask = np.zeros(shape=max_boxes, dtype=np.float32)
    idx = np.zeros(shape=max_boxes, dtype=np.float32)
    for j, item in enumerate(label):
        item[:4] = item[:4] / downsampling_ratio
        xmin, ymin, xmax, ymax, category = item
        category = int(category)
        h, w = int(ymax - ymin), int(xmax - xmin)
        radius = gaussian_radius((h, w))
        radius = max(0, int(radius))
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        center = np.array([cx, cy], dtype=np.float32)
        center_int = center.astype(np.int32)
        create_gauss_heatmap(hm[:, :, category], center_int, radius)
        off[j] = center - center_int
        wh[j] = (w * 1.0, h * 1.0)
        mask[j] = 1
        idx[j] = center_int[1] * hm.shape[0] + center_int[0]
    return hm, off, wh, mask, idx


def read_flip_augment(path):
    image = read_image(path[:path.index(".flip")])
    if ".flip-x." in path:
        image = flip2(image, 'x')
    elif ".flip-y." in path:
        image = flip2(image, 'y')
    elif ".flip-xy." in path:
        image = flip2(image, 'xy')
    else:
        raise ValueError(path)
    return image


class DataHolder:
    debug_mode = 0
    def __init__(self,
                 train_list_files,
                 input_size,
                 num_classes,
                 batch_size,
                 max_boxes,
                 downsampling_ratio, 
                 channle_means=[]):

        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_classes
        # downsampling_ratio is input_size / feature_size, 
        # see last deconv layer's ouput to ensure feature_size!
        self.dsr = downsampling_ratio
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
        logI("Labels:", dict(counter))

    def init(self):
        n = len(self.annotations)
        self._hmaps = np.zeros((n, self.hmap_size, self.hmap_size, self.num_classes), np.float32)
        self._offsets = np.zeros((n, self.max_boxes, 2), np.float32)
        self._sizes = np.zeros((n, self.max_boxes, 2), np.float32)
        self._masks = np.zeros((n, self.max_boxes), np.float32)
        self._indices = np.zeros((n, self.max_boxes), np.float32)
        self._x_train = np.zeros((n, self.input_size, self.input_size, 3), np.float32)
        for i, line in enumerate(self.annotations):
            if i and (i % 100 == 0):
                rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
                logI(F"{i} images loaded, {round(rss, 2)}GB memory used")
            x, hm, off, wh, mask, idx = self._decode(line)
            self._x_train[i] = x
            self._hmaps[i] = hm
            self._offsets[i] = off
            self._sizes[i] = wh
            self._masks[i] = mask
            self._indices[i] = idx

    def generate(self):
        rand_orders = np.random.permutation(self._x_train.shape[0])
        for i in range(self.num_batches):
            ids = rand_orders[i * self.batch_size:(i + 1) * self.batch_size]
            x = self._x_train[ids]
            y = self._hmaps[ids], self._offsets[ids], self._sizes[ids], self._masks[ids], self._indices[ids]
            yield x, y

    def _parse(self, line):
        items = line.strip().split(" ")
        path = items[0]
        if path.endswith(".augment") and ".flip" in path:
            image = read_flip_augment(path)
        else:
            image = read_image(items[0])
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
                logW(F"Found more than {self.max_boxes} boxes({len(items)-1}): {items[0]}")
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
                xmin, ymin, xmax, ymax = correct_box_o2l(xmin, ymin, xmax, ymax, resized.shape[0], img.shape)
                if DataHolder.debug_mode:
                    Box(xmin, ymin, xmax, ymax).draw(resized, thickness=2)
            corrected.append([xmin, ymin, xmax, ymax, c])
        if DataHolder.debug_mode:
            show_image(resized)
        label = np.stack(corrected)
        label = label[label[:, 4] != -1]
        hmap_shape = (self.hmap_size, self.hmap_size, self.num_classes)
        hm, off, wh, mask, idx = decode_label(label, self.max_boxes, hmap_shape, self.dsr)
        if self.channel_means:
            resized = sub_means(resized, self.channel_means)
        x = normalize_l1(resized, 255.0, 0, True)
        return x, hm, off, wh, mask, idx


