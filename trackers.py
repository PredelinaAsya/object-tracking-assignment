import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List


def compute_pairwise_iou(bboxes1, bboxes2):
    """
    Returns ious np.array with IoU values computed pairwise
    for all corresponding bboxes from input arrays.
    Input bboxes in format (x1, y1, x2, y2).
    """
    x_min_1, y_min_1 = bboxes1[:, 0], bboxes1[:, 1]
    x_max_1, y_max_1 = bboxes1[:, 2], bboxes2[:, 3]

    x_min_2, y_min_2 = bboxes2[:, 0], bboxes2[:, 1]
    x_max_2, y_max_2 = bboxes2[:, 2], bboxes2[:, 3]

    w1, h1 = bboxes1[:, 2] - bboxes1[:, 0], bboxes1[:, 3] - bboxes1[:, 1]
    w2, h2 = bboxes2[:, 2] - bboxes2[:, 0], bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    zero = np.zeros_like(x_min_1)

    inter_width = np.maximum(zero, np.minimum(x_max_1, x_max_2) - np.maximum(x_min_1,x_min_2))
    inter_height = np.maximum(zero, np.minimum(y_max_1, y_max_2) - np.maximum(y_min_1,y_min_2))
    inter_area = inter_width * inter_height
    union_area = (area1 + area2) - inter_area

    return inter_area / union_area


class SoftTracker(object):
    def __init__(self):
        self.frame_num = 0
        self.max_track_id = 0
        self.trackers = {}

    def init_new_track(self, bbox: List[int]):
        self.trackers[self.max_track_id] = bbox
        self.max_track_id += 1

    def match_by_iou(self, dets):
        iou_matrix = np.zeros((len(self.trackers), dets.shape[0]))

        for j, trk_bbox in self.trackers.items():
            trk_bbox_arr = np.array([trk_bbox]).astype(float)
            iou_matrix[j, :] = compute_pairwise_iou(trk_bbox_arr, dets)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_ids, matched_det_ids = [], []

        for trk_id, det_id in zip(row_ind, col_ind):
            if iou_matrix[trk_id, det_id] > 0:
                matched_ids.append((int(trk_id), int(det_id)))
                matched_det_ids.append(det_id)
        
        return matched_ids, matched_det_ids
    
    def compute_euclidean_dist(self, bbox1, bbox2):
        xc1, yc1 = (bbox1[0] + bbox1[2]) * 0.5, (bbox1[1] + bbox1[3]) * 0.5
        xc2, yc2 = (bbox2[0] + bbox2[2]) * 0.5, (bbox2[1] + bbox2[3]) * 0.5

        dist = np.sqrt((xc1 - xc2) ** 2 + (yc1 - yc2) ** 2)

        return dist
    
    def find_min_dist_track_for_detection(self, matched_trk_ids, det_bbox):
        min_dist, matched_trk_id = None, None

        for trk_id in range(self.max_track_id):
            if trk_id not in matched_trk_ids:
                dist = self.compute_euclidean_dist(self.trackers[trk_id], det_bbox)
                if min_dist is None or dist <= min_dist:
                    min_dist = dist
                    matched_trk_id = trk_id

        return matched_trk_id

    def update(self, el):
        if not len(self.trackers):
            for x in el['data']:
                if len(x['bounding_box']):
                    x['track_id'] = self.max_track_id
                    self.init_new_track(x['bounding_box'])

        else:
            dets = np.array([x['bounding_box'] for x in el['data'] if len(x['bounding_box'])]).astype(float)
            det_ids = [i for i, x in enumerate(el['data']) if len(x['bounding_box'])]

            if not dets.shape[0]:
                return el

            matched_ids, matched_det_ids = self.match_by_iou(dets)
            matched_trk_ids = []

            for trk_id, det_id in matched_ids:
                self.trackers[trk_id] = dets[det_id, :]
                el['data'][det_ids[det_id]]['track_id'] = trk_id
                matched_trk_ids.append(trk_id)
            
            unmatched_dets = [i for i in range(len(det_ids)) if i not in matched_det_ids]

            for det_id in unmatched_dets:
                matched_trk_id = self.find_min_dist_track_for_detection(
                    matched_trk_ids, dets[det_id, :],
                )

                trk_id = matched_trk_id if matched_trk_id is not None else self.max_track_id

                self.trackers[trk_id] = dets[det_id, :]
                el['data'][det_ids[det_id]]['track_id'] = trk_id
                matched_trk_ids.append(trk_id)

                if matched_trk_id is None:
                     self.init_new_track(dets[det_id, :])

        self.frame_num += 1

        return el