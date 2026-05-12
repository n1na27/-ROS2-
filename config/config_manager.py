"""
配置管理器
提供配置文件的读取、更新和管理功能
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config_data = {}
        self._load_config()
        self._setup_logging()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config_data = yaml.safe_load(f) or {}
            else:
                print(f"配置文件 {self.config_file} 不存在，将创建默认配置")
                self.config_data = {}
                self._create_default_config()
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            self.config_data = {}
    
    def _create_default_config(self):
        """创建默认配置文件"""
        default_config = {
            'system': {
                'module_b_path': 'modules/try.py',
                'trigger_cooldown': 30,
                'max_concurrent_triggers': 1,
                'enable_logging': True,
                'log_level': 'INFO'
            },
            'frs': {
                'threshold_cosine': 0.65,
                'threshold_high_confidence': 0.75,
                'gender_confidence_threshold': 0.8,
                'min_consecutive_frames': 5,
                'identity_history_len': 7,
                'embedding_ema_alpha': 0.5,
                'processing_interval': 3,
                'max_faces': 5,
                'max_missed_frames': 15,
                'max_missed_frames_new': 8,
                'min_detection_count': 3
            },
            'camera': {
                'camera_width': 640,
                'camera_height': 480,
                'detection_width': 320,
                'detection_height': 240,
                'camera_index': 0
            },
            'display': {
                'window_name': 'FRS Trigger System',
                'window_width': 800,
                'window_height': 600,
                'show_fps': True,
                'show_face_count': True
            },
            'paths': {
                'ref_image_dir': 'Img',
                'model_dir': 'models',
                'log_dir': 'logs'
            }
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
            self.config_data = default_config
            print(f"已创建默认配置文件: {self.config_file}")
        except Exception as e:
            print(f"创建默认配置文件失败: {e}")
    
    def _setup_logging(self):
        """设置日志"""
        log_level = self.get('system.log_level', 'INFO')
        enable_logging = self.get('system.enable_logging', True)
        
        if enable_logging:
            # 创建日志目录
            log_dir = self.get('paths.log_dir', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # 配置日志
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(os.path.join(log_dir, 'frs_trigger.log')),
                    logging.StreamHandler()
                ]
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套键
        
        Args:
            key: 配置键，支持 'section.key' 格式
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键，支持 'section.key' 格式
            value: 配置值
        """
        keys = key.split('.')
        config = self.config_data
        
        # 导航到目标位置
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def save(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return False
    
    def update_module_b_path(self, module_b_path: str):
        """
        更新模块B路径
        
        Args:
            module_b_path: 新的模块B路径
        """
        self.set('system.module_b_path', module_b_path)
        self.save()
    
    def get_module_b_path(self) -> str:
        """获取模块B路径"""
        return self.get('system.module_b_path', 'modules/try.py')
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        required_paths = [
            'system.module_b_path',
            'paths.ref_image_dir',
            'paths.model_dir'
        ]
        
        for path in required_paths:
            if not self.get(path):
                print(f"配置项 {path} 缺失或为空")
                return False
        
        # 检查模块B文件是否存在
        module_b_path = self.get_module_b_path()
        if not os.path.exists(module_b_path):
            print(f"模块B文件不存在: {module_b_path}")
            return False
        
        return True
    
    def get_absolute_path(self, config_key: str) -> str:
        """
        获取配置项的绝对路径
        
        Args:
            config_key: 配置键
            
        Returns:
            绝对路径
        """
        relative_path = self.get(config_key)
        if not relative_path:
            return ""
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.config_file)))
        return os.path.join(base_dir, relative_path)