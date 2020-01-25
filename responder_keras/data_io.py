import os
import shutil
import numpy as np
import cv2

from logger import Logger
logger = Logger.generate_logger(__name__)

class DataLoadError(Exception):
    pass

def swap_bgr_rgb(img):
    if img.shape[-1] == 3:
        output_img = img[..., [2, 1, 0]]
    elif img.shape[-1] == 4:
        output_img = img[..., [2, 1, 0, -1]]
    else:
        output_img = img
    return output_img

def imencode(input_img):
    img_encode = cv2.imencode(
        '.png', 
        swap_bgr_rgb(input_img)
    )[1]
    data_encode = np.array(img_encode)
    return data_encode.tostring()


def create_img_fromstring(bytes):
    nparr = np.fromstring(bytes, np.uint8)
    return swap_bgr_rgb(cv2.imdecode(nparr, -1))


def load_img(img_path):
    loaded_img = cv2.imread(img_path, -1)
    if loaded_img is None:
        raise DataLoadError(img_path+" cannot be loaded as img")
    return cv2.imread(img_path)

def save_as_json(save_dict, output_path):
    save_json = json.dumps(
        save_dict
    )
    with open(output_path, "w") as f:
        f.write(save_json)
