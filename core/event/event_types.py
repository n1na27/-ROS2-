"""
事件类型定义
"""

import time
from typing import Optional, Any, Dict
from dataclasses import dataclass


@dataclass
class FaceRecognitionEvent:
    """
    人脸识别事件
    
    当FRS系统识别到人脸时触发此事件
    """
    face_id: int  # 人脸跟踪ID
    identity: str  # 识别身份（"Unknown"表示陌生人）
    confidence: float  # 识别置信度（0-1）
    timestamp: float  # 识别时间戳
    face_image: Optional[Any] = None  # 人脸图像（可选）
    gender: Optional[str] = None  # 性别（可选）
    age: Optional[int] = None  # 年龄（可选）
    metadata: Optional[Dict[str, Any]] = None  # 其他元数据
    
    def __post_init__(self):
        """初始化后处理"""
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_known_face(self) -> bool:
        """是否为已知人脸"""
        return self.identity != "Unknown"
    
    @property
    def is_high_confidence(self) -> bool:
        """是否为高置信度识别"""
        return self.confidence >= 0.75
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'face_id': self.face_id,
            'identity': self.identity,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'gender': self.gender,
            'age': self.age,
            'is_known_face': self.is_known_face,
            'is_high_confidence': self.is_high_confidence,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"FaceRecognitionEvent(id={self.face_id}, identity={self.identity}, confidence={self.confidence:.3f})"


@dataclass
class SystemEvent:
    """
    系统事件
    
    用于系统内部状态变化的通知
    """
    event_type: str  # 事件类型
    timestamp: float  # 事件时间戳
    message: str  # 事件消息
    data: Optional[Dict[str, Any]] = None  # 事件数据
    
    def __post_init__(self):
        """初始化后处理"""
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.data is None:
            self.data = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'message': self.message,
            'data': self.data
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"SystemEvent(type={self.event_type}, message={self.message})"


class EventTypes:
    """事件类型常量"""
    
    # 人脸识别事件
    FACE_RECOGNIZED = "face_recognized"
    FACE_UNKNOWN = "face_unknown"
    FACE_LOST = "face_lost"
    
    # 系统事件
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    SYSTEM_ERROR = "system_error"
    MODULE_B_TRIGGERED = "module_b_triggered"
    MODULE_B_STARTED = "module_b_started"
    MODULE_B_FAILED = "module_b_failed"
    
    # 配置事件
    CONFIG_UPDATED = "config_updated"
    MODULE_B_CHANGED = "module_b_changed"