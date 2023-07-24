import pandas as pd
import cv2
import os
import numpy as np
import yaml
from tqdm import tqdm, trange

CONFIG_PATH = r'./config/config.local.yaml'

with open(CONFIG_PATH, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

if __name__ == '__main__':
    print(cv2.getVersionString())
    print(config)