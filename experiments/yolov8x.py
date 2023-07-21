#!/usr/bin/env python
# coding: utf-8

import yaml
import os
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm.notebook import trange, tqdm
from glob import glob
from ultralytics import YOLO

parser = argparse.ArgumentParser(
        usage="%(prog)s [GPU] [CAM], e.g. 1 01",
        description="Run YOLOv8 on harbour videos."
    )

parser.add_argument('gpu', type=int, default=0)
parser.add_argument('cam', type=str, default='01')

args = parser.parse_args()

CONFIG_PATH = r'../config/config_sumo3.yaml'
DEVICE = args.gpu
CAM = args.cam
HALF = 0 if DEVICE == 0 or DEVICE == 2 else 1

with open(CONFIG_PATH, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

camera_cfg = config[f'camera-{CAM}']

files = glob(f"{camera_cfg['data-folder']}/*.mkv")
half_idx = len(files)//2

sel_files = files[:half_idx] if HALF == 0 else files[half_idx:]

print(len(files), half_idx, sel_files[0])


model = YOLO("../saved_models/yolov8x.pt")  # load a pretrained model (recommended for training)

for f in sel_files:
    video_file_name = os.path.basename(f)
    video_file_name_no_ext = video_file_name[:-4]
    video_file = f

    results = model(video_file, stream=True, device=DEVICE, imgsz=1920, save_txt=True, save_conf=True, save=False, classes=[8])  # predict on an image

    res = []
    for result in results:
        curr = result.boxes.data.cpu().numpy()
        res.append(curr)

    with open(f'./output/{video_file_name_no_ext}.pickle', 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)