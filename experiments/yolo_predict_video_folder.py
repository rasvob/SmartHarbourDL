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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage="%(prog)s [GPU] [CFG], e.g. 0 yolo-test-model-video-1",
        description="Run YOLOv8 on harbour videos."
    )

    parser.add_argument('gpu', type=int, default=0)
    parser.add_argument('cfg', type=str, default='yolo-test-model-video-1')

    args = parser.parse_args()

    device = args.gpu
    cfg = args.cfg
    half = 0 if device == 0 or device == 2 else 1

    CONFIG_PATH = r'../config/config_sumo3.yaml'
    
    with open(CONFIG_PATH, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    params = config[cfg]
    files = glob(os.path.join(params['input-folder'], f'*{params["filter-string"]}*.mkv'))

    half_idx = len(files)//2
    sel_files = files[:half_idx] if half == 0 else files[half_idx:]
    
    print(len(files), len(sel_files))

    model = YOLO(params['input-model'])  # load a pretrained model (recommended for training)

    for f in tqdm(sel_files):
        print(f)
        results = model(f, stream=False, device=device, imgsz=params['video-width'], save_txt=True, save_conf=True, save=False, classes=[8])  # predict on an image