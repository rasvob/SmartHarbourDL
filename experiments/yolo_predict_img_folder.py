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
    CONFIG_PATH = r'../config/config_sumo3.yaml'
    DEVICE = 0
    with open(CONFIG_PATH, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    params = config['yolo-test-model']

    files = glob(os.path.join(params['input-folder'], '*.jpg'))

    model = YOLO(params['input-model'])  # load a pretrained model (recommended for training)

    res = []
    for f in tqdm(files):
        results = model(f, stream=False, device=DEVICE, imgsz=params['video-width'], save_txt=True, save_conf=True, save=False, classes=[8])  # predict on an image

        for result in results:
            curr = result.boxes.data.cpu().numpy()
            res.append(curr)

    with open(params['raw-output-path'], 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)