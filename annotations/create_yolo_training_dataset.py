from typing import List, Tuple
import cv2
import pandas as pd
import os
from tqdm import tqdm, trange
import yaml
import logging
from glob import glob

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def create_symlinks(files:List[str], subset:str, data_folder:str, image:bool=True):
    for x in files:
        bname = os.path.basename(x)
        new_path = os.path.join(data_folder, 'images' if image else 'labels', subset, bname)
        os.symlink(x, new_path)

if __name__ == '__main__':
    CONFIG_PATH = r'../config/config_sumo3.yaml'
    yolo_section = 'yolo-dataset-cam-02'
    with open(CONFIG_PATH, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    folders = ['images', 'labels']
    model_folders = ['train', 'val']
    data_folder = config[yolo_section]['yolo-path']
    raw_folder = config[yolo_section]['output-path-raw']

    for folder in model_folders:
        for subfolder in folders:
            new_folder = os.path.join(data_folder, subfolder, folder)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            
    images, labels = glob(os.path.join(raw_folder, '*', 'images', '*.jpg')), glob(os.path.join(raw_folder, '*', 'labels', '*.txt'))
    train_selection, val_selection = '202306', '202307'
    train_images, train_labels = [x for x in images if train_selection in x], [x for x in labels if train_selection in x]
    val_images, val_labels = [x for x in images if val_selection in x], [x for x in labels if val_selection in x]

    if 'cam-01' in yolo_section or 'cam-02' in yolo_section:
        cam = 'cam_' + yolo_section.split('-')[-1]
        train_images, train_labels = [x for x in train_images if cam in x], [x for x in train_labels if cam in x]
        val_images, val_labels = [x for x in val_images if cam in x], [x for x in val_labels if cam in x]

    logger.info(f'Found {len(train_images)} training images and {len(train_labels)} training labels')
    logger.info(f'Found {len(val_images)} validation images and {len(val_labels)} validation labels')

    if len(train_images) != len(train_labels) or len(val_images) != len(val_labels):
        raise Exception('Number of images and labels do not match')

    create_symlinks(train_images, 'train', data_folder, image=True)
    create_symlinks(train_labels, 'train', data_folder, image=False)

    create_symlinks(val_images, 'val', data_folder, image=True)
    create_symlinks(val_labels, 'val', data_folder, image=False)