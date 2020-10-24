from PIL import Image
import numpy as np
from deeplabv3 import DeepLabModel
from trimap import trimap
import sys,os
sys.path.insert(1, 'FBA_Matting')
from networks.models import build_model
from demo import pred


def deeplabv3_model():
    return DeepLabModel("deeplabv3_pascal_trainval_2018_01_04.tar.gz")


def FBA_model():
    class Args:
        encoder = 'resnet50_GN_WS'
        decoder = 'fba_decoder'
        weights = 'FBA_Matting/FBA.pth'
    args=Args()
    matting_model = build_model(args)
    return matting_model

def read_image(name):
    return (np.array(name) / 255.0)
def read_trimap(name):
    trimap_im = np.array(name) / 255.0
    h, w = trimap_im.shape
    trimap = np.zeros((h, w, 2))
    trimap[trimap_im == 1, 1] = 1
    trimap[trimap_im == 0, 0] = 1
    return trimap

def auto_hair(image_url, out_, in_, MODEL1, model2):
    original_im = Image.open(image_url)
    resized_im, seg_map = MODEL1.run(original_im)
    mask = np.equal(seg_map, 15)
    mask = mask.astype(int)
    mask = mask*255
    mask = Image.fromarray(mask.astype(np.uint8))
    deeplablv3_final = resized_im.copy()
    deeplablv3_final.putalpha(mask)
    trimap_im = Image.fromarray(trimap(np.array(mask), out_, in_))
    input_resized_im = read_image(resized_im)
    input_trimap_im = read_trimap(trimap_im)
    fg, bg, alpha = pred(input_resized_im, input_trimap_im, model2)
    alpha = Image.fromarray((alpha*255).astype(np.uint8))
    final = Image.fromarray((input_resized_im*255).astype(np.uint8))
    final.putalpha(alpha)
    return resized_im, mask, trimap_im, final, deeplablv3_final
