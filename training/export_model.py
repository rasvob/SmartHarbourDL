import comet_ml
import dotenv
import yaml
import os
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
    CONFIG_NAME = 'yolo-export'
    with open(CONFIG_PATH, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    params = config[CONFIG_NAME]

    model = YOLO(params['input-model']) 
    model.export(format=params['format'], device=0) 