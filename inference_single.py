import glob
import os
import cv2 as cv

import torch
import numpy as np

import albumentations as A
from torchvision.transforms import functional as F

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

def get_model_config(model_type, config_path):
    assert model_type in ['vitmatte-s', 'vitmatte-b']
    config = LazyConfig.load(config_path)

    if model_type == 'vitmatte-b':
        config.model.backbone.embed_dim = 768
        config.model.backbone.num_heads = 12
        config.model.decoder.in_chans = 768
    
    return config

def load_model(model_type, checkpoint_path, config_path):
    config = get_model_config(model_type, config_path)

    model = instantiate(config.model)
    model.cuda()
    model.eval()
    DetectionCheckpointer(model).load(checkpoint_path)

    return model

def load_sample(sample_path, image_max_size):
    image = cv.imread(glob.glob(os.path.join(sample_path, 'image.*'))[0],
                      cv.IMREAD_COLOR_BGR)
    trimap = cv.imread(os.path.join(sample_path, 'trimap.png'),
                        cv.IMREAD_GRAYSCALE)
    sample_original = dict(image=image, trimap=trimap)

    transform_a = A.Compose([A.LongestMaxSize(max_size=image_max_size)],
                       additional_targets={'trimap': 'mask'})
    sample_transformed = transform_a(**sample_original)

    sample_transformed['image'] = F.to_tensor(sample_transformed['image']).unsqueeze(0).cuda()
    sample_transformed['trimap'] = F.to_tensor(sample_transformed['trimap']).unsqueeze(0).cuda()

    return dict(original=sample_original, transformed=sample_transformed)

def visualize_alpha(alpha, sample):
    image = sample['original']['image']

    image_rgba = np.zeros(alpha.shape+(4,), dtype=np.uint8)
    image_rgba[:,:,:3] = image
    image_rgba[:,:,3] = alpha

    return image_rgba

def main(sample_path, image_max_size=1024):
    model = load_model('vitmatte-b', 'checkpoint/ViTMatte_B_DIS.pth', 'configs/common/model.py')
    sample = load_sample(sample_path, image_max_size)

    with torch.no_grad():
        alpha = model(sample['transformed'])['phas'].cpu().squeeze(0, 1).numpy()
    # resize output to original image size
    h, w = sample['original']['trimap'].shape
    if alpha.shape != (h, w):
        alpha = cv.resize(alpha, (w, h))
    alpha = (255. * alpha).astype(np.uint8)

    vis = visualize_alpha(alpha, sample)

    cv.imwrite(os.path.join(sample_path, 'alpha.png'), alpha)
    cv.imwrite(os.path.join(sample_path, 'vis_alpha.png'), vis)

if __name__ == '__main__':
    main('test_data/dogs', 1024)