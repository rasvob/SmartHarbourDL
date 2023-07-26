import comet_ml
import dotenv
import yaml
import os
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm.notebook import trange, tqdm
from glob import glob
from ultralytics import YOLO
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

if __name__ == '__main__':
    dotenv.load_dotenv()

    CONFIG_PATH = r'../config/config_sumo3.yaml'
    DEVICES = [0,1,2,3]

    with open(CONFIG_PATH, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    experiment = Experiment(
        api_key = os.environ.get("COMET_API_KEY"),
        project_name = config['yolo-dataset']['comet-project-name']
        workspace=os.environ.get("COMET_WORKSPACE")
    )

    model = YOLO(config['yolo-dataset']['input-model'])  # load a pretrained model (recommended for training)
    model.train(data='coco128.yaml', epochs=100, imgsz=1920, device=DEVICES, batch=8, pretrained=True, cache=True, workers=16, seed=13)