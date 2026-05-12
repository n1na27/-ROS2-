import os
import cv2
import torch
import logging
from typing import Optional

from .face_tracker import FaceTracker
from .face_recognizer import FaceRecognizer
from .visualization import Visualization
from .frs_event_publisher import FRSEventPublisher


class FaceRecognitionSystem:
    """人脸识别系统主类"""
    
    def __init__(self, config, event_publisher: Optional[FRSEventPublisher] = None):
        """
        初始化人脸识别系统
        
        Args:
            config: 配置对象
            event_publisher: FRS事件发布器
        """
        self.config = config
        self.event_publisher = event_publisher
        self.logger = logging.getLogger(__name__)
        
        self.setup_environment()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # 初始化组件
        self.face_recognizer = FaceRecognizer(self.config, self.device)
        self.face_tracker = FaceTracker(self.config)
        self.visualization = Visualization(self.config)
        
        # 状态变量
        self.frame_counter = 0
        self.last_boxes = None
        self.cap = None
        
        # 跟踪已处理的人脸，避免重复发布事件
        self.processed_faces = set()
        
    def setup_environment(self):
        """设置环境变量"""
        os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
        os.environ.setdefault('DISPLAY', ':0')
    
    def initialize_camera(self) -> bool:
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
            
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
                
            self.logger.info(f"摄像头初始化成功，索引: {self.config.CAMERA_INDEX}")
            return True
            
        except Exception as e:
            self.logger.error(f"摄像头初始化失败: {e}")
            return False
    
    def process_frame(self, frame, module_b_launcher=None):
        """处理单帧图像"""
        display_frame = frame.copy()
        small_frame = cv2.resize(frame, (self.config.DETECTION_WIDTH, self.config.DETECTION_HEIGHT))
        
        # 人脸检测
        boxes = self.detect_faces(small_frame)
        
        # 目标跟踪
        matched_faces = self.face_tracker.update(boxes, small_frame.shape)
        
        # 获取当前跟踪的人脸ID
        current_face_ids = {fid for fid, _ in matched_faces}
        
        # 处理每个人脸
        for fid, box in matched_faces:
            self.process_single_face(fid, box, frame, display_frame)
        
        # 处理离开的人脸
        self._handle_lost_faces(current_face_ids)
        
        # 添加系统信息
        self.visualization.add_system_info(display_frame, self.face_tracker, 
                                       self.event_publisher.event_publisher if self.event_publisher else None)
        
        # 添加状态信息
        if module_b_launcher:
            self.visualization.add_status_info(display_frame, module_b_launcher)
        
        # 添加帮助文本
        self.visualization.add_help_text(display_frame)
        
        return display_frame
    
    def detect_faces(self, small_frame):
        """检测人脸"""
        if self.frame_counter % self.config.PROCESSING_INTERVAL == 0:
            try:
                small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                boxes, _, _ = self.face_recognizer.mtcnn.detect(small_rgb, landmarks=True)
                self.last_boxes = boxes
            except Exception as e:
                self.logger.error(f"人脸检测错误: {e}")
                boxes = self.last_boxes
        else:
            boxes = self.last_boxes
        
        return boxes
    
    def process_single_face(self, fid, box, frame, display_frame):
        """处理单个人脸"""
        fdata = self.face_tracker.trackers[fid]
        
        # 坐标映射和调整
        x1, y1, x2, y2 = self.map_and_adjust_coordinates(box, display_frame.shape)
        
        # 检查是否已触发
        is_triggered = self.event_publisher and fid in self.event_publisher.event_publisher.triggered_faces
        
        # 绘制边界框
        self.visualization.draw_face_box(display_frame, x1, y1, x2, y2, fdata['color'], is_triggered)
        
        # 处理人脸区域
        if self.should_process_face(fdata):
            face_region = frame[y1:y2, x1:x2]
            try:
                # 性别预测
                self.face_recognizer.predict_gender(fid, face_region, self.face_tracker)
                
                # 人脸识别
                self.face_recognizer.recognize_face(fid, face_region, self.face_tracker)
                
                # 发布事件
                if self.event_publisher:
                    self.event_publisher.publish_face_recognition(fid, fdata, face_region)
                    
            except Exception as e:
                self.logger.error(f"处理人脸 {fid} 时发生错误: {e}")
        
        # 添加文本信息
        self.visualization.add_face_info(display_frame, fdata, x1, y1)
    
    def _handle_lost_faces(self, current_face_ids):
        """处理离开的人脸"""
        # 找出离开的人脸ID
        lost_face_ids = self.processed_faces - current_face_ids
        
        # 通知事件发布器
        if self.event_publisher:
            for fid in lost_face_ids:
                self.event_publisher.remove_face(fid)
        
        # 更新已处理人脸集合
        self.processed_faces = current_face_ids
    
    def map_and_adjust_coordinates(self, box, display_shape):
        """映射和调整坐标"""
        scale_x = display_shape[1] / self.config.DETECTION_WIDTH
        scale_y = display_shape[0] / self.config.DETECTION_HEIGHT
        
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)
        
        padding = int(min(x2 - x1, y2 - y1) * 0.15)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(display_shape[1], x2 + padding)
        y2 = min(display_shape[0], y2 + padding)
        
        return x1, y1, x2, y2
    
    def should_process_face(self, fdata):
        """判断是否应该处理该人脸"""
        return (self.frame_counter % self.config.PROCESSING_INTERVAL == 0 and
                fdata['detection_count'] % 2 == 0 and
                fdata['last_box'][2] - fdata['last_box'][0] > 30 and
                fdata['last_box'][3] - fdata['last_box'][1] > 30)
    
    def load_reference_features(self) -> bool:
        """加载参考特征"""
        try:
            ref_features = self.face_recognizer.load_reference_features()
            if not ref_features:
                self.logger.warning("未找到有效的参考图像")
                return False
            
            self.logger.info(f"成功加载 {len(ref_features)} 个参考图像")
            return True
            
        except Exception as e:
            self.logger.error(f"加载参考特征时发生错误: {e}")
            return False
    
    def reset_triggers(self):
        """重置所有触发状态"""
        if self.event_publisher:
            self.event_publisher.clear_all_faces()
        self.logger.info("已重置所有触发状态")
    
    def get_system_status(self) -> dict:
        """获取系统状态"""
        status = {
            'frame_counter': self.frame_counter,
            'tracked_faces': len(self.face_tracker.trackers),
            'device': str(self.device),
            'camera_active': self.cap is not None and self.cap.isOpened()
        }
        
        if self.event_publisher:
            status.update(self.event_publisher.event_publisher.get_trigger_statistics())
        
        return status
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("系统资源已清理")