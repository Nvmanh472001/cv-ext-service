import os
import imghdr
import cv2
from PIL import Image

def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    return imgs_lists


def check_and_read_gif(img_path):
    if os.path.basename(img_path)[-3:] in ['gif', 'GIF']:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            print("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True
    return None, False


def read_cfg_from_yaml(yaml_path, cfg_op='all'):
    support_dict = ['all', 'net', 'pre_process_list', "postprocess_params"]
    assert cfg_op in support_dict, \
        Exception('when model typs is {}, backbone only support {}'
        .format(cfg_op, support_dict))
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))
    
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
        
    if res.get('net') is None:
        raise ValueError('{} has no Neraul Network Architecture'.format(yaml_path))
    
    if cfg_op == 'all':
        return res
    return res[cfg_op]

def pil2cv(image):
    if isinstance(image, Image.Image):
        import numpy as np
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
    return image

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    
    return image, (w, h)

