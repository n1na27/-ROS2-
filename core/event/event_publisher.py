"""
事件发布器
负责处理和发布人脸识别事件
"""

import time
import logging
from typing import Set, Dict, List, Callable, Optional
from .event_types import FaceRecognitionEvent, SystemEvent, EventTypes


class EventPublisher:
    """
    事件发布器类
    
    负责接收人脸识别事件，并根据配置决定是否触发模块B
    """
    
    def __init__(self, module_b_launcher, config_manager):
        """
        初始化事件发布器
        
        Args:
            module_b_launcher: 模块B启动器实例
            config_manager: 配置管理器实例
        """
        self.module_b_launcher = module_b_launcher
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # 触发状态管理
        self.triggered_faces: Set[int] = set()  # 已触发的人脸ID集合
        self.last_trigger_time: Dict[int, float] = {}  # 每个人脸的最后触发时间
        self.trigger_count: int = 0  # 总触发次数
        
        # 事件监听器
        self.event_listeners: Dict[str, List[Callable]] = {}
        
        # 配置参数
        self.trigger_cooldown = config_manager.get('system.trigger_cooldown', 30)
        self.max_concurrent_triggers = config_manager.get('system.max_concurrent_triggers', 1)
        
        self.logger.info(f"事件发布器初始化完成，冷却时间: {self.trigger_cooldown}秒")
    
    def publish_recognition_event(self, event: FaceRecognitionEvent):
        """
        发布人脸识别事件
        
        Args:
            event: 人脸识别事件
        """
        self.logger.debug(f"收到人脸识别事件: {event}")
        
        # 通知事件监听器
        self._notify_listeners(EventTypes.FACE_RECOGNIZED, event)
        
        # 只对已知人脸触发模块B
        if event.is_known_face:
            self._handle_known_face(event)
        else:
            self.logger.debug(f"识别到陌生人，不触发模块B: {event.identity}")
    
    def _handle_known_face(self, event: FaceRecognitionEvent):
        """
        处理已知人脸事件
        
        Args:
            event: 人脸识别事件
        """
        current_time = time.time()
        face_id = event.face_id
        
        # 检查冷却时间
        if face_id in self.last_trigger_time:
            elapsed = current_time - self.last_trigger_time[face_id]
            if elapsed < self.trigger_cooldown:
                self.logger.debug(f"人脸 {face_id} 仍在冷却期内，剩余时间: {self.trigger_cooldown - elapsed:.1f}秒")
                return
        
        # 检查是否已经触发过
        if face_id in self.triggered_faces:
            self.logger.debug(f"人脸 {face_id} 已经触发过模块B")
            return
        
        # 触发模块B
        self._trigger_module_b(event)
    
    def _trigger_module_b(self, event: FaceRecognitionEvent):
        """
        触发模块B
        
        Args:
            event: 人脸识别事件
        """
        try:
            self.logger.info(f"触发模块B，识别到已知人脸: {event.identity} (置信度: {event.confidence:.3f})")
            
            # 启动模块B
            success = self.module_b_launcher.launch_module_b(event)
            
            if success:
                # 更新触发状态
                self.triggered_faces.add(event.face_id)
                self.last_trigger_time[event.face_id] = time.time()
                self.trigger_count += 1
                
                # 发布系统事件
                system_event = SystemEvent(
                    event_type=EventTypes.MODULE_B_TRIGGERED,
                    timestamp=time.time(),
                    message=f"模块B已触发，识别到: {event.identity}",
                    data={
                        'face_id': event.face_id,
                        'identity': event.identity,
                        'confidence': event.confidence,
                        'trigger_count': self.trigger_count
                    }
                )
                self._notify_listeners(EventTypes.MODULE_B_TRIGGERED, system_event)
                
                self.logger.info(f"模块B触发成功，总触发次数: {self.trigger_count}")
            else:
                self.logger.error("模块B触发失败")
                
                # 发布失败事件
                system_event = SystemEvent(
                    event_type=EventTypes.MODULE_B_FAILED,
                    timestamp=time.time(),
                    message="模块B启动失败",
                    data={'event': event.to_dict()}
                )
                self._notify_listeners(EventTypes.MODULE_B_FAILED, system_event)
                
        except Exception as e:
            self.logger.error(f"触发模块B时发生异常: {e}")
    
    def reset_trigger_state(self, face_id: int):
        """
        重置指定人脸的触发状态
        
        Args:
            face_id: 人脸ID
        """
        if face_id in self.triggered_faces:
            self.triggered_faces.remove(face_id)
            self.logger.debug(f"已重置人脸 {face_id} 的触发状态")
        
        if face_id in self.last_trigger_time:
            del self.last_trigger_time[face_id]
    
    def reset_all_trigger_states(self):
        """重置所有触发状态"""
        self.triggered_faces.clear()
        self.last_trigger_time.clear()
        self.logger.info("已重置所有触发状态")
    
    def add_event_listener(self, event_type: str, callback: Callable):
        """
        添加事件监听器
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        
        self.event_listeners[event_type].append(callback)
        self.logger.debug(f"已添加事件监听器: {event_type}")
    
    def remove_event_listener(self, event_type: str, callback: Callable):
        """
        移除事件监听器
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self.event_listeners:
            try:
                self.event_listeners[event_type].remove(callback)
                self.logger.debug(f"已移除事件监听器: {event_type}")
            except ValueError:
                self.logger.warning(f"尝试移除不存在的监听器: {event_type}")
    
    def _notify_listeners(self, event_type: str, event):
        """
        通知事件监听器
        
        Args:
            event_type: 事件类型
            event: 事件对象
        """
        if event_type in self.event_listeners:
            for callback in self.event_listeners[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"事件监听器回调执行失败: {e}")
    
    def get_trigger_statistics(self) -> Dict[str, any]:
        """
        获取触发统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_triggers': self.trigger_count,
            'currently_triggered_faces': len(self.triggered_faces),
            'triggered_face_ids': list(self.triggered_faces),
            'cooldown_period': self.trigger_cooldown,
            'max_concurrent_triggers': self.max_concurrent_triggers
        }
    
    def update_config(self, config_manager):
        """
        更新配置
        
        Args:
            config_manager: 新的配置管理器
        """
        self.config_manager = config_manager
        self.trigger_cooldown = config_manager.get('system.trigger_cooldown', 30)
        self.max_concurrent_triggers = config_manager.get('system.max_concurrent_triggers', 1)
        self.logger.info(f"配置已更新，冷却时间: {self.trigger_cooldown}秒")