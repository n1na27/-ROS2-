"""
模块B启动器
负责启动和管理模块B程序
"""

import os
import sys
import subprocess
import logging
import time
import threading
from typing import Optional, Dict, Any
from ..event.event_types import FaceRecognitionEvent, SystemEvent, EventTypes


class ModuleBLauncher:
    """
    模块B启动器类
    
    负责根据人脸识别事件启动指定的模块B程序
    """
    
    def __init__(self, config_manager):
        """
        初始化模块B启动器
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # 模块B路径
        self.module_b_path = config_manager.get_module_b_path()
        
        # 运行状态
        self.is_running = False
        self.current_process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.last_trigger_event: Optional[FaceRecognitionEvent] = None
        
        # 统计信息
        self.launch_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        self.logger.info(f"模块B启动器初始化完成,当前模块B路径: {self.module_b_path}")
    
    def launch_module_b(self, event: FaceRecognitionEvent) -> bool:
        """
        启动模块B程序
        
        Args:
            event: 人脸识别事件
            
        Returns:
            bool: 启动是否成功
        """
        if self.is_running:
            self.logger.warning("模块B已在运行中,跳过启动")
            return False
        
        # 检查是否是同一人脸的重复触发
        if self.last_trigger_event and self.last_trigger_event.face_id == event.face_id:
            # 如果是同一人脸且距离上次触发时间不足冷却时间，跳过
            elapsed = time.time() - self.last_trigger_event.timestamp
            cooldown = self.config_manager.get('system.trigger_cooldown', 30)
            if elapsed < cooldown:
                self.logger.debug(f"人脸 {event.face_id} 仍在冷却期内，剩余时间: {cooldown - elapsed:.1f}秒")
                return False
        
        try:
            # 检查模块B文件是否存在
            if not os.path.exists(self.module_b_path):
                self.logger.error(f"模块B文件不存在: {self.module_b_path}")
                return False
            
            # 准备启动参数
            cmd = [sys.executable, self.module_b_path]
            
            # 添加事件信息作为环境变量
            env = os.environ.copy()
            env.update({
                'FRS_TRIGGER_IDENTITY': event.identity,
                'FRS_TRIGGER_CONFIDENCE': str(event.confidence),
                'FRS_TRIGGER_FACE_ID': str(event.face_id),
                'FRS_TRIGGER_TIMESTAMP': str(event.timestamp)
            })
            
            self.logger.info(f"启动模块B: {' '.join(cmd)}")
            self.logger.info(f"触发信息: 身份={event.identity}, 置信度={event.confidence:.3f}")
            
            # 启动模块B
            self.current_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 更新状态
            self.is_running = True
            self.start_time = time.time()
            self.last_trigger_event = event
            self.launch_count += 1
            
            self.logger.info(f"模块B启动成功,进程ID: {self.current_process.pid}")
            
            # 异步监控进程
            self._monitor_process()
            
            return True
            
        except Exception as e:
            self.logger.error(f"启动模块B时发生异常: {e}")
            self.failure_count += 1
            return False
    
    def _monitor_process(self):
        """异步监控模块B进程状态"""
        def monitor_thread():
            if self.current_process is None:
                return
            
            try:
                # 等待进程结束
                stdout, stderr = self.current_process.communicate()
                return_code = self.current_process.returncode
                
                # 进程已结束
                self.is_running = False
                
                if return_code == 0:
                    self.success_count += 1
                    self.logger.info(f"模块B正常退出,进程ID: {self.current_process.pid}")
                    if stdout:
                        self.logger.debug(f"模块B输出: {stdout.strip()}")
                else:
                    self.logger.warning(f"模块B异常退出,返回码: {return_code}")
                    self.failure_count += 1
                    if stderr:
                        self.logger.warning(f"模块B错误输出: {stderr.strip()}")
                
                self.current_process = None
                self.start_time = None
                
                # 发布模块B完成事件（可选，用于通知其他组件）
                # 注意：这需要主控制器注册相应的事件监听器
                self.logger.debug("模块B进程已结束，状态已重置，可再次触发")
                
            except Exception as e:
                self.logger.error(f"监控模块B进程时发生异常: {e}")
                self.is_running = False
                self.current_process = None
        
        # 启动异步监控线程
        thread = threading.Thread(target=monitor_thread, daemon=True)
        thread.start()
    
    def stop_module_b(self) -> bool:
        """
        停止模块B程序
        
        Returns:
            bool: 停止是否成功
        """
        if not self.is_running or self.current_process is None:
            self.logger.warning("模块B未在运行")
            return False
        
        try:
            self.logger.info(f"停止模块B,进程ID: {self.current_process.pid}")
            
            # 尝试优雅终止
            self.current_process.terminate()
            
            # 等待进程结束
            try:
                self.current_process.wait(timeout=5)
                self.logger.info("模块B已优雅终止")
            except subprocess.TimeoutExpired:
                # 强制终止
                self.logger.warning("模块B未在指定时间内终止,强制终止")
                self.current_process.kill()
                self.current_process.wait()
            
            self.is_running = False
            self.current_process = None
            self.start_time = None
            
            return True
            
        except Exception as e:
            self.logger.error(f"停止模块B时发生异常: {e}")
            return False
    
    def set_module_b_path(self, new_path: str) -> bool:
        """
        设置新的模块B路径
        
        Args:
            new_path: 新的模块B路径
            
        Returns:
            bool: 设置是否成功
        """
        if self.is_running:
            self.logger.warning("模块B正在运行中,无法更改路径")
            return False
        
        # 检查新路径是否存在
        if not os.path.exists(new_path):
            self.logger.error(f"新的模块B文件不存在: {new_path}")
            return False
        
        old_path = self.module_b_path
        self.module_b_path = new_path
        
        # 更新配置
        self.config_manager.update_module_b_path(new_path)
        
        self.logger.info(f"模块B路径已更新: {old_path} -> {new_path}")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取启动器状态
        
        Returns:
            状态信息字典
        """
        status = {
            'is_running': self.is_running,
            'module_b_path': self.module_b_path,
            'launch_count': self.launch_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_count / max(self.launch_count, 1)
        }
        
        if self.is_running and self.current_process:
            status.update({
                'process_id': self.current_process.pid,
                'running_time': time.time() - self.start_time if self.start_time else 0
            })
        
        if self.last_trigger_event:
            status['last_trigger'] = {
                'identity': self.last_trigger_event.identity,
                'confidence': self.last_trigger_event.confidence,
                'face_id': self.last_trigger_event.face_id,
                'timestamp': self.last_trigger_event.timestamp
            }
        
        return status
    
    def reset_statistics(self):
        """重置统计信息"""
        self.launch_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.logger.info("统计信息已重置")
    
    def update_config(self, config_manager):
        """
        更新配置
        
        Args:
            config_manager: 新的配置管理器
        """
        self.config_manager = config_manager
        new_path = config_manager.get_module_b_path()
        
        if new_path != self.module_b_path:
            if self.set_module_b_path(new_path):
                self.logger.info(f"配置已更新,模块B路径: {new_path}")
            else:
                self.logger.warning(f"配置更新失败，保持原路径: {self.module_b_path}")