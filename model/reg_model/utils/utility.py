import os
from urllib.request import urlretrieve
import torch
import yaml
from modules.converter import AttrDict
import pandas as pd
import cv2
from PIL import Image, JpegImagePlugin
import numpy as np
from utils.imageproc import loadImage

def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)
    return model

def get_config(file_path, for_model=True):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

def load_config_from_yaml(file_path):
    with open(file_path, 'r', encoding='utf8') as buf:
        config = yaml.safe_load(buf)
    return config

def reformat_input(image):
    if type(image) == str:
        if image.startswith('http://') or image.startswith('https://'):
            tmp, _ = urlretrieve(image , reporthook=printProgressBar(prefix = 'Progress:', suffix = 'Complete', length = 50))
            img_cv_grey = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
            os.remove(tmp)
        else:
            img_cv_grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = os.path.expanduser(image)
        img = loadImage(image)  # can accept URL
    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2: # grayscale
            img_cv_grey = image
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            img_cv_grey = np.squeeze(image)
            img = cv2.cvtColor(img_cv_grey, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3: # BGRscale
            img = image
            img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4: # RGBAscale
            img = image[:,:,:3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif type(image) == JpegImagePlugin.JpegImageFile:
        image_array = np.array(image)
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError('Invalid input type. Supporting format = string(file path or url), bytes, numpy array')

    return img, img_cv_grey

def reformat_input_batched(image, n_width=None, n_height=None):
    """
    reformats an image or list of images or a 4D numpy image array &
    returns a list of corresponding img, img_cv_grey nd.arrays
    image:
        [file path, numpy-array, byte stream object,
        list of file paths, list of numpy-array, 4D numpy array,
        list of byte stream objects]
    """
    if ((isinstance(image, np.ndarray) and len(image.shape) == 4) or isinstance(image, list)):
        # process image batches if image is list of image np arr, paths, bytes
        img, img_cv_grey = [], []
        for single_img in image:
            clr, gry = reformat_input(single_img)
            if n_width is not None and n_height is not None:
                clr = cv2.resize(clr, (n_width, n_height))
                gry = cv2.resize(gry, (n_width, n_height))
            img.append(clr)
            img_cv_grey.append(gry)
        img, img_cv_grey = np.array(img), np.array(img_cv_grey)
        # ragged tensors created when all input imgs are not of the same size
        if len(img.shape) == 1 and len(img_cv_grey.shape) == 1:
            raise ValueError("The input image array contains images of different sizes. " +
                             "Please resize all images to same shape or pass n_width, n_height to auto-resize")
    else:
        img, img_cv_grey = reformat_input(image)
    return img, img_cv_grey

def printProgressBar(prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    def progress_hook(count, blockSize, totalSize):
        progress = count * blockSize / totalSize
        percent = ("{0:." + str(decimals) + "f}").format(progress * 100)
        filledLength = int(length * progress)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')

    return progress_hook