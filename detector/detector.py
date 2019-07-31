# coding: utf-8
# @Author: oliver
# @Date:   2019-07-14 18:50:35

import os
import sys
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm

curr = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr)

from models.pose_dla import get_model
from models.resnet import get_model
from utils.image import transform_preds
from utils.image import get_affine_transform
from utils.decode import ctdet_decode

DEBUG = True

def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict_ = checkpoint['state_dict']

    state_dict = {}
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    model_state_dict = model.state_dict()
    # check loaded parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '\
                    'loaded shape{}.'.format(
                k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    return model

class Detector(object):
    def __init__(self, model_path, image_size=(384, 384), device='cuda'):
        self._image_size = image_size
        self._device = device

        self._num_classes = 4
        self._max_per_image = 50
        self._threshold = 0.5
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        heads = {'wh': 2, 'reg': 2, 'hm': self._num_classes}
        head_conv = 64
        self._model = get_model(18, heads, head_conv)
        for param in self._model.parameters():
            param.requires_grad = False
        self._model = load_model(self._model, model_path, self._device)
        self._model = self._model.to(self._device)
        self._model.eval()

    def pre_process(self, image):
        height, width = image.shape[0:2]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = get_affine_transform(c, s, 0, [self._image_size[1], self._image_size[0]])
        img = cv2.warpAffine(image, trans_input, (self._image_size[1], self._image_size[0]), flags=cv2.INTER_LINEAR)
        img = ((img / 255. - self._mean) / self._std).astype(np.float32)
        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
        '''
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        meta = {'c': c, 's': s}
        return img, meta

    def post_process(self, dets, meta, output_size):
        dets = dets.data.cpu().numpy()
        dets = dets.reshape(-1, dets.shape[2])
        c = meta['c']
        s = meta['s']
        h, w = output_size

        top_preds = {}
        dets[:, :2] = transform_preds(dets[:, 0:2], c, s, (w, h))
        dets[:, 2:4] = transform_preds(dets[:, 2:4], c, s, (w, h))
        classes = dets[:, -1]
        scores = dets[:, 4]
        '''
        if len(scores) > self._max_per_image:
            kth = len(scores) - self._max_per_image
            thresh = np.partition(scores, kth)[kth]
        '''

        for j in range(self._num_classes):
            inds = np.logical_and(classes == j, scores >= self._threshold)
            top_preds[j + 1] = np.concatenate([dets[inds, :4].astype(np.float32), 
                scores[inds].reshape(-1, 1).astype(np.float32)], axis=1)
        return top_preds

    def detect(self, image):
        t1 = time.time()
        img, meta = self.pre_process(image)
        t2 = time.time()
        img = img.to(self._device)
        if self._device == 'cuda':
            torch.cuda.synchronize()
        t3 = time.time()
        with torch.set_grad_enabled(False):
            output = self._model(img)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
        if self._device == 'cuda':
            torch.cuda.synchronize()
        t4 = time.time()
        dets = ctdet_decode(hm, wh, reg=reg, K=self._max_per_image)
        dets = self.post_process(dets, meta, hm.size()[2:4])
        if self._device == 'cuda':
            torch.cuda.synchronize()
        t5 = time.time()
        
        if DEBUG:
            '''
            for j in range(1, self._num_classes+1):
                for bbox in dets[j]:
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
            '''
            bbox = dets[0]
            area_ratio = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) / float(image.shape[0] * image.shape[1])
            if 0.1 < area_ratio < 1.0:
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
            print('\npre: {:.3f}s | net: {:.3f}s | post: {:.3f}s'.format(t2-t1, t4-t3, t5-t4))
        return image    

if __name__ == '__main__':
    model_path = './checkpoints/model_250.pth'
    detector = Detector(model_path, image_size=(384, 384), device='cpu')

    images_dir = './experiments/test_images'
    output_dir = './experiments/outputs'
    file_list = os.listdir(images_dir)
    total_time = 0.0
    for file in tqdm(file_list):
        file_name = os.path.join(images_dir, file)
        image = cv2.imread(file_name)
        t1 = time.time()
        show = detector.detect(image)
        t2 = time.time()
        total_time += (t2 - t1)
        cv2.imwrite(os.path.join(output_dir, file), show)
    print('avg time: {}s, {}s/{}'.format(total_time/len(file_list), total_time, len(file_list)))


