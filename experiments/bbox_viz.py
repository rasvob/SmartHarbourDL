import pandas as pd
import cv2
import os
import numpy as np
import yaml
from tqdm import tqdm, trange

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), wv=1920, hv=1080):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  center_x, center_y = int(box[0]), int(box[1])
  left_top_x = center_x -  int((box[2]/2))
  left_top_y = center_y -  int((box[3]/2))

  right_bot_x = center_x +  int((box[2]/2))
  right_bot_y = center_y +  int((box[3]/2))
  p1, p2 = (left_top_x, left_top_y), (right_bot_x, right_bot_y)
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

def plot_bboxes(image, boxes, label_id=8, scores=None):
    labels = {8: u'boat'}
    colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),(155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),(128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),(164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),(181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),(100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),(205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),(119, 24, 48),(73, 8, 110)]
    if scores is None:
        scores = [1.0] * len(boxes)
  
  #plot each boxes
    for e, box in enumerate(boxes):
        #add score in label if score=True
        label = labels[label_id] + " " + str(round(100 * float(scores[e]),1)) + "%"
        color = colors[label_id]
        box_label(image, box, label, color,wv=1920, hv=1080)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Video',image)

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       # draw circle here (etc...)
       print('x = %d (%f), y = %d (%f)'%(x, x/1920, y, y/1080))

if __name__ == '__main__':
    CONFIG_PATH = r'./config/config.local.yaml'

    with open(CONFIG_PATH, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    annotations_path = r'.\annotations\yolov8x_no_train_labels.csv'
    video = 'cfg_raw_cam_01_fhd_h265_20230609T050002.mkv'
    CAM = '01' if '_cam_01' in video else '02'
    df = pd.read_csv(annotations_path, sep=';')
    df_file = df[df['filename'] == video]
    folder = config[f'camera-{CAM}']['data-folder']

    video_path = os.path.join(folder, video)
    # video_path = os.path.join(r'C:\Users\svo0175\Downloads', r'cfg_raw_cam_02_fhd_h265_20230609T090003 (5).avi')

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'frame count: {frame_count}, fps: {fps}')

    curr_frame = 0
    seq = trange(frame_count)
    for i in seq:
        key = cv2.waitKey(0)

        if key == ord('n') or i == 0:
            ret, frame = cap.read()
            if ret:
                boxes = df_file[df_file['frame_id'] == curr_frame][['x', 'y', 'w', 'h']].values
                confidence_scores = df_file[df_file['frame_id'] == curr_frame]['confidence'].values
                plot_bboxes(frame, boxes, label_id=8, scores=confidence_scores)
                cv2.setMouseCallback('Video', onMouse)
                curr_frame += 1
                seq.update()
            else:
                break
        elif key == ord('f'):
            fast_forward = 4*10 - 1
            for _ in range(fast_forward):
                ret, frame = cap.read()
                seq.update()
            # cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame + fast_forward)
            curr_frame += fast_forward
            seq.refresh()

            ret, frame = cap.read()
            seq.update()
            curr_frame += 1
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Video',frame)
        elif key == ord('q'):
            break

        else:
            print(f"unknown key code 0x{key:02x}")

    cv2.destroyAllWindows()