from typing import List, Tuple
import cv2
import pandas as pd
import os
from tqdm import tqdm, trange
import yaml
import logging

CAM = '01'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def df_to_yolo_arr(df:pd.DataFrame, video_width, video_height) -> List[Tuple[str, str, int, float, float, float, float]]:
    df_sub = df[['filename', 'frame_id', 'label_class', 'x', 'y', 'w', 'h']]
    df_sub.loc[:, 'x'] = df_sub['x'] / video_width
    df_sub.loc[:, 'y'] = df_sub['y'] / video_height
    df_sub.loc[:, 'w'] = df_sub['w'] / video_width
    df_sub.loc[:, 'h'] = df_sub['h'] / video_height
    return [tuple(x) for x in df_sub.to_numpy()]

def save_yolo_labels(yolo_arr:List[Tuple[str, str, int, float, float, float, float]], output_path_raw:str) -> None:
    for row in yolo_arr:
        fname, frame = row[:2]
        data = row[2:]
        data_str = ' '.join([str(x) for x in data]) + '\n'
        label = int(data[0])

        with open(os.path.join(output_path_raw, f'Camera_{CAM}', 'labels', f'{fname[:-4]}_{frame}.txt'), 'w') as f:
            if label >= 0:
                f.write(data_str)

def save_yolo_jpg(yolo_arr:List[Tuple[str, str, int, float, float, float, float]], output_path_raw, data_folder) -> None:
    yolo_set = set([x[0] for x in yolo_arr])
    yolo_dict = {x:sorted([y for y in yolo_arr if y[0] == x], key=lambda v: v[1], reverse=False) for x in yolo_set}

    for key, val in tqdm(yolo_dict.items(), total=len(yolo_dict)):
        video_path = os.path.join(data_folder, key)
        cap = cv2.VideoCapture(video_path)

        frames = [x[1] for x in val]
        max_frame = max(frames)
        for i in trange(max_frame + 1):
            ret, img = cap.read()
            if not ret:
                logger.error(f'Error reading frame {i} from {key}')
                break

            if i in frames:
                out_path = os.path.join(output_path_raw, f'Camera_{CAM}', 'images', f'{key[:-4]}_{i}.jpg')
                cv2.imwrite(out_path, img)
        cap.release()

if __name__ == '__main__':
    CONFIG_PATH = r'../config/config_sumo3.yaml'
    with open(CONFIG_PATH, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_folder = config[f'camera-{CAM}']['data-folder']
    video_width = int(config['yolo-dataset']['video-width'])
    video_height = int(config['yolo-dataset']['video-height'])

    folders = ['images', 'labels']
    for folder in folders:
        new_folder = os.path.join(config['yolo-dataset']['output-path-raw'], f'Camera_{CAM}', folder)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

    df = pd.read_csv(config['yolo-dataset']['input-labels'], sep=';', index_col=0)
    df['camera_id'] = df['camera_id'].astype(str).apply(lambda x: x.zfill(2))

    df_filtered = df[df['camera_id'] == CAM]
    
    yolo_arr = df_to_yolo_arr(df_filtered, video_width, video_height)
    save_yolo_labels(yolo_arr, config['yolo-dataset']['output-path-raw'])

    save_yolo_jpg(yolo_arr, config['yolo-dataset']['output-path-raw'], config[f'camera-{CAM}']['data-folder'])