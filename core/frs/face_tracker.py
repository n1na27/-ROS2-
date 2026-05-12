import time
from collections import deque
import numpy as np


class FaceTracker:
    """人脸跟踪器类"""
    
    def __init__(self, config):
        self.config = config
        self.trackers = {}
        self.next_face_id = 0
        
    def update(self, current_boxes, frame_shape):
        """更新跟踪器状态"""
        if current_boxes is None or len(current_boxes) == 0:
            self._handle_no_detections()
            return []
        
        return self._match_and_update_trackers(current_boxes, frame_shape)
    
    def _handle_no_detections(self):
        """处理无人脸检测的情况"""
        lost_faces = []
        for fid in list(self.trackers.keys()):
            self.trackers[fid]['missed_frames'] += 1
            if self.trackers[fid]['missed_frames'] > self.config.MAX_MISSED_FRAMES:
                lost_faces.append(fid)
        
        for fid in lost_faces:
            del self.trackers[fid]
    
    def _match_and_update_trackers(self, current_boxes, frame_shape):
        """匹配和更新跟踪器"""
        matched_faces = []
        used_ids = set()

        for box in current_boxes:
            best_match_id = self._find_best_match(box, frame_shape, used_ids)
            
            if best_match_id is not None:
                self._update_existing_tracker(best_match_id, box)
                matched_faces.append((best_match_id, self.trackers[best_match_id]['last_box']))
                used_ids.add(best_match_id)
            else:
                self._create_new_tracker(box, matched_faces)
        
        self._remove_lost_trackers(used_ids)
        return matched_faces
    
    def _find_best_match(self, box, frame_shape, used_ids):
        """为检测框寻找最佳匹配的跟踪器"""
        best_match_id = None
        best_score = 0.0
        
        for fid, fdata in self.trackers.items():
            if fid in used_ids:
                continue
                
            iou = self._calculate_iou(box, fdata['last_box'])
            distance = self._calculate_center_distance(box, fdata['last_box'])
            max_distance = np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
            normalized_distance = distance / (max_distance + 1e-9)
            score = iou * 0.7 + (1 - normalized_distance) * 0.3
            
            if score > best_score and score > 0.4:
                best_score = score
                best_match_id = fid
                
        return best_match_id
    
    def _update_existing_tracker(self, fid, box):
        """更新现有跟踪器"""
        old_box = self.trackers[fid]['last_box']
        alpha = 0.3
        smoothed_box = [
            old_box[0] * (1-alpha) + box[0] * alpha,
            old_box[1] * (1-alpha) + box[1] * alpha,
            old_box[2] * (1-alpha) + box[2] * alpha,
            old_box[3] * (1-alpha) + box[3] * alpha
        ]
        self.trackers[fid]['last_box'] = smoothed_box
        self.trackers[fid]['missed_frames'] = 0
        self.trackers[fid]['detection_count'] += 1
    
    def _create_new_tracker(self, box, matched_faces):
        """创建新跟踪器"""
        if len(self.trackers) >= self.config.MAX_FACES:
            return
            
        fid = self.next_face_id
        self.trackers[fid] = self._create_tracker_data(fid, box)
        matched_faces.append((fid, box))
        self.next_face_id += 1
    
    def _create_tracker_data(self, fid, box):
        """创建跟踪器数据结构"""
        return {
            'last_box': box,
            'missed_frames': 0,
            'detection_count': 1,
            'gender_history': deque(maxlen=7),
            'last_gender_text': "",
            'stable_gender': "Unknown",
            'identity_history': deque(maxlen=self.config.IDENTITY_HISTORY_LEN),
            'ema_embedding': None,
            'last_identity': "Unknown",
            'last_identity_score': 0.0,
            'last_display_text': "Detecting...",
            'color': self.config.COLORS[fid % len(self.config.COLORS)],
            'last_update_time': time.time()
        }
    
    def _remove_lost_trackers(self, used_ids):
        """移除丢失的跟踪器"""
        lost_faces = []
        for fid in list(self.trackers.keys()):
            if fid not in used_ids:
                self.trackers[fid]['missed_frames'] += 1
                if (self.trackers[fid]['missed_frames'] > self.config.MAX_MISSED_FRAMES or
                    (self.trackers[fid]['missed_frames'] > self.config.MAX_MISSED_FRAMES_NEW and
                     self.trackers[fid]['detection_count'] < self.config.MIN_DETECTION_COUNT)):
                    lost_faces.append(fid)
        
        for fid in lost_faces:
            del self.trackers[fid]
    
    @staticmethod
    def _calculate_iou(box1, box2):
        """计算交并比"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / (area1 + area2 - intersection)
    
    @staticmethod
    def _calculate_center_distance(box1, box2):
        """计算中心点距离"""
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)