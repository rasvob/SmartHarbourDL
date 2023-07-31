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
    EXPERIMENT_NAME = 'yolo-dataset-cam-01'
    with open(CONFIG_PATH, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    params = config[EXPERIMENT_NAME]
    DEVICES = params['devices']
    
    experiment = Experiment(
        api_key = os.environ.get("COMET_API_KEY"),
        project_name = params['comet-project-name'],
        workspace=os.environ.get("COMET_WORKSPACE")
    )

    model = YOLO(params['input-model'])  # load a pretrained model (recommended for training)
    model.train(data=params['coco-input-file'], epochs=params['epochs'], imgsz=params['video-width'], device=DEVICES, batch=params['batch'], pretrained=True, cache=True, workers=16, seed=13)