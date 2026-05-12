"""
FRS事件发布器
将FRS系统的人脸识别结果转换为事件并发布
"""

import time
import logging
from typing import Optional
from ..event.event_types import FaceRecognitionEvent
from ..event.event_publisher import EventPublisher


class FRSEventPublisher:
    """
    FRS事件发布器
    
    负责将FRS系统的人脸识别结果转换为标准事件并发布到事件系统
    """
    
    def __init__(self, event_publisher: EventPublisher):
        """
        初始化FRS事件发布器
        
        Args:
            event_publisher: 事件发布器实例
        """
        self.event_publisher = event_publisher
        self.logger = logging.getLogger(__name__)
        
        # 跟踪已发布的人脸事件，避免重复发布
        self.published_faces = {}
        
        self.logger.info("FRS事件发布器初始化完成")
    
    def publish_face_recognition(self, face_id: int, face_data: dict, face_image=None):
        """
        发布人脸识别事件
        
        Args:
            face_id: 人脸跟踪ID
            face_data: 人脸数据字典，包含识别结果
            face_image: 人脸图像（可选）
        """
        try:
            # 提取关键信息
            identity = face_data.get('last_identity', 'Unknown')
            confidence = face_data.get('last_identity_score', 0.0)
            gender = face_data.get('stable_gender', None)
            
            # 创建人脸识别事件
            event = FaceRecognitionEvent(
                face_id=face_id,
                identity=identity,
                confidence=confidence,
                timestamp=time.time(),
                face_image=face_image,
                gender=gender,
                metadata={
                    'display_text': face_data.get('last_display_text', ''),
                    'gender_text': face_data.get('last_gender_text', ''),
                    'detection_count': face_data.get('detection_count', 0),
                    'color': face_data.get('color', (255, 255, 255))
                }
            )
            
            # 检查是否需要发布事件
            if self._should_publish_event(face_id, event):
                self.event_publisher.publish_recognition_event(event)
                self._update_published_face(face_id, event)
                
                self.logger.debug(f"已发布人脸识别事件: {event}")
            else:
                self.logger.debug(f"跳过发布事件，人脸 {face_id} 状态未变化")
                
        except Exception as e:
            self.logger.error(f"发布人脸识别事件时发生异常: {e}")
    
    def _should_publish_event(self, face_id: int, event: FaceRecognitionEvent) -> bool:
        """
        判断是否应该发布事件
        
        Args:
            face_id: 人脸ID
            event: 人脸识别事件
            
        Returns:
            bool: 是否应该发布
        """
        # 如果是第一次识别到这个人脸
        if face_id not in self.published_faces:
            return True
        
        # 获取上次发布的事件
        last_event = self.published_faces[face_id]
        
        # 如果身份发生变化
        if last_event.identity != event.identity:
            return True
        
        # 如果置信度显著提升（提升超过0.1）
        if event.confidence - last_event.confidence > 0.1:
            return True
        
        # 如果从未知变为已知
        if last_event.identity == "Unknown" and event.identity != "Unknown":
            return True
        
        return False
    
    def _update_published_face(self, face_id: int, event: FaceRecognitionEvent):
        """
        更新已发布的人脸事件记录
        
        Args:
            face_id: 人脸ID
            event: 人脸识别事件
        """
        self.published_faces[face_id] = event
    
    def remove_face(self, face_id: int):
        """
        移除人脸记录（当人脸离开时调用）
        
        Args:
            face_id: 人脸ID
        """
        if face_id in self.published_faces:
            del self.published_faces[face_id]
            self.logger.debug(f"已移除人脸记录: {face_id}")
        
        # 通知事件发布器重置触发状态
        self.event_publisher.reset_trigger_state(face_id)
    
    def clear_all_faces(self):
        """清除所有人脸记录"""
        self.published_faces.clear()
        self.event_publisher.reset_all_trigger_states()
        self.logger.info("已清除所有人脸记录")
    
    def get_published_faces_count(self) -> int:
        """
        获取已发布人脸的数量
        
        Returns:
            int: 人脸数量
        """
        return len(self.published_faces)
    
    def get_published_faces_info(self) -> dict:
        """
        获取已发布人脸的信息
        
        Returns:
            dict: 人脸信息字典
        """
        return {
            face_id: {
                'identity': event.identity,
                'confidence': event.confidence,
                'timestamp': event.timestamp,
                'is_known_face': event.is_known_face
            }
            for face_id, event in self.published_faces.items()
        }