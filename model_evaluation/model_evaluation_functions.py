import pandas as pd
import numpy as np
import tqdm
from mean_average_precision import MetricBuilder

def calculate_iou(ground_truth:tuple, prediction:tuple):
    """
        Calculate intersection over union for two bounding boxes.
        Args:
            ground_truth: tuple of (x, y, w, h)
            prediction: tuple of (x, y, w, h)
    """
    gt_xtl = ground_truth[0]-ground_truth[2]/2
    gt_ytl = ground_truth[1]-ground_truth[3]/2
    gt_xbr = ground_truth[0]+ground_truth[2]/2
    gt_ybr = ground_truth[1]+ground_truth[3]/2
    pr_xtl = prediction[0]-prediction[2]/2
    pr_ytl = prediction[1]-prediction[3]/2
    pr_xbr = prediction[0]+prediction[2]/2
    pr_ybr = prediction[1]+prediction[3]/2
    intersection_xtl = max(gt_xtl, pr_xtl)
    intersection_ytl = max(gt_ytl, pr_ytl)
    intersection_xbr = min(gt_xbr, pr_xbr)
    intersection_ybr = min(gt_ybr, pr_ybr)
    intersection_area = max(0, intersection_xbr - intersection_xtl) * max(0, intersection_ybr - intersection_ytl)
    union_area = ground_truth[2] * ground_truth[3] + prediction[2] * prediction[3] - intersection_area
    return intersection_area / union_area

def evaluate_model_frame_ids_score(df_ground_truth:pd.DataFrame, df_predictions:pd.DataFrame):
    """
        Evaluate model based on ground truth and predictions based on frame_id match.
        Index of both dataframes should be 'filename' and 'frame_id' should be a column.
        TODO: add iou_threshold support to filter out predictions with low iou
    """
    df_predictions_frame_indexed = df_predictions.reset_index().set_index(['filename', 'frame_id'])
    # group quality results by name, aggregate over frame_id and calculate true positive, false positive, false negative when comparing corresponding names and frame from both dataset
    evaluation_dict = dict()
    for id in tqdm.tqdm(set(df_ground_truth.index) | set(df_predictions.index)):
        evaluation_dict[id] = dict()

        if id in df_ground_truth.index:
            if df_ground_truth.loc[id,'frame_id'].size == 1:
                ground_truth_frame_ids = set([df_ground_truth.loc[id,'frame_id']])
            else:
                ground_truth_frame_ids = set(df_ground_truth.loc[id,'frame_id'])
        else:
            ground_truth_frame_ids = set()

        if id in df_predictions.index:
            if df_predictions.loc[id,'frame_id'].size == 1:
                yolo_frame_ids = set([df_predictions.loc[id,'frame_id']])
            else:
                yolo_frame_ids = set(df_predictions.loc[id,'frame_id'])
        else:
            yolo_frame_ids = set()
        
        corresponding_frames = ground_truth_frame_ids & yolo_frame_ids
        evaluation_dict[id]['true_positive'] = len(corresponding_frames)
        evaluation_dict[id]['false_positive'] = len(yolo_frame_ids - ground_truth_frame_ids)
        evaluation_dict[id]['false_negative'] = len(ground_truth_frame_ids - yolo_frame_ids)
        if len(corresponding_frames) > 0:
            frames_iou = {}
            for frame_id in corresponding_frames:
                # calulate iou for each frame
                ground_truth_frame = df_ground_truth.loc[id].loc[df_ground_truth.loc[id].frame_id == frame_id].iloc[0]
                # prediction_frame = df_predictions.loc[id].loc[df_predictions.loc[id].frame_id == frame_id].sort_values('w', ascending=False).iloc[0] ## this line was computation heavy, therefore indexed version is used
                prediciton_frames = df_predictions_frame_indexed.loc[id].loc[frame_id]
                if len(prediciton_frames.shape) == 1:
                    prediction_frame = prediciton_frames
                else:
                    prediction_frame = prediciton_frames.sort_values(['confidence', 'w'], ascending=False).iloc[0]
                frames_iou[frame_id] = calculate_iou(ground_truth_frame[['x', 'y', 'w', 'h']].values, prediction_frame[['x', 'y', 'w', 'h']].values)
            evaluation_dict[id]['iou'] = sum(frames_iou.values()) / len(frames_iou.values())
            evaluation_dict[id]['frames_iou'] = frames_iou
        else:
            evaluation_dict[id]['iou'] = 0
            evaluation_dict[id]['frames_iou'] = []

    df_evaluation = pd.DataFrame().from_dict(evaluation_dict, orient='index')
    df_evaluation['f1'] = 2 * df_evaluation['true_positive'] / (2 * df_evaluation['true_positive'] + df_evaluation['false_positive'] + df_evaluation['false_negative'])
    df_evaluation['recall'] = df_evaluation['true_positive'] / (df_evaluation['true_positive'] + df_evaluation['false_negative'])
    df_evaluation['precision'] = df_evaluation['true_positive'] / (df_evaluation['true_positive'] + df_evaluation['false_positive'])
    
    total_eval = df_evaluation[['true_positive','false_positive','false_negative']].sum(axis=0)
    total_evaluation_dict = {
        'f1': 2*total_eval['true_positive'] / (2*total_eval['true_positive'] + total_eval['false_positive'] + total_eval['false_negative']),
        'recall': total_eval['true_positive'] / (total_eval['true_positive'] + total_eval['false_negative']),
        'precision': total_eval['true_positive'] / (total_eval['true_positive'] + total_eval['false_positive']),
        'iou': (df_evaluation['true_positive']*df_evaluation['iou']).sum() / total_eval['true_positive']
    }
    return df_evaluation, total_evaluation_dict

def evaluate_mAP(df_ground_truth:pd.DataFrame, df_predictions:pd.DataFrame):
    """
        Evaluate mAP to see the object detection quality.
        Use https://github.com/bes-dev/mean_average_precision.git for evaluation.
        Index of both dataframes should be 'filename' and 'frame_id' should be a column.
    """
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    # TODO decide which is the right evaluation procedure
    # for id in tqdm.tqdm(set(df_ground_truth.index) | set(df_yolo_predictions.index)):
    for id in tqdm.tqdm(set(df_ground_truth.index) & set(df_predictions.index)):
        # TODO there can be an error bcs id not in index    
        df_ground_truth_filtered = df_ground_truth.loc[id].set_index('frame_id')
        df_yolo_predictions_filtered = df_predictions.loc[id].set_index('frame_id')
        # if id in df_ground_truth.index:
        #     if df_ground_truth.loc[id,'frame_id'].size == 1:
        #         ground_truth_frame_ids = set([df_ground_truth.loc[id,'frame_id']])
        #     else:
        #         ground_truth_frame_ids = set(df_ground_truth.loc[id,'frame_id'])
        # else:
        #     ground_truth_frame_ids = set()

        # if id in df_yolo_predictions.index:
        #     if df_yolo_predictions.loc[id,'frame_id'].size == 1:
        #         yolo_frame_ids = set([df_yolo_predictions.loc[id,'frame_id']])
        #     else:
        #         yolo_frame_ids = set(df_yolo_predictions.loc[id,'frame_id'])
        # else:
        #     yolo_frame_ids = set()

        # TODO decide which is the right evaluation procedure - only the frames with ground truth?
        # for frame_id in list(df_ground_truth_filtered.index):
        for frame_id in list(df_ground_truth_filtered.index.union(df_yolo_predictions_filtered.index)):
            if frame_id in df_ground_truth_filtered.index:            
                gt = df_ground_truth_filtered.loc[frame_id][['xmin', 'ymin', 'xmax', 'ymax']].values
            else:
                gt = np.array([0, 0, 0, 0])
            if frame_id in df_yolo_predictions_filtered.index:
                preds = df_yolo_predictions_filtered.loc[frame_id][['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].values
            else:
                preds = np.array([0, 0, 0, 0, 0])

            if len(gt.shape) == 1:
                gt = np.array([gt])
            if len(preds.shape) == 1:
                preds = np.array([preds])
            """
            format for mAP evaluation library:
                gt [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
                preds [xmin, ymin, xmax, ymax, class_id, confidence]
            """
            gt = np.concatenate((gt, np.zeros((gt.shape[0], 3))), axis=1)
            preds = np.concatenate((preds, np.zeros((preds.shape[0], 1))), axis=1)
            preds[:, -1] = preds[:, -2]
            preds[:, -2] = 0
            metric_fn.add(preds, gt)

    return {
        'mAP_0.5_11points': metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP'], # PASCAL VOC metric
        'mAP_0.5_all_points': metric_fn.value(iou_thresholds=0.5)['mAP'], # PASCAL VOC metric at the all points
        'mAP_0.5_0.95_101points': metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP'] # COCO metric
    }