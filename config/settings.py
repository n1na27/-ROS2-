import os
import yaml


class Config:
    """配置参数类"""
    
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        self._load_config()
        
    def _load_config(self):
        """从YAML文件加载配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            # 系统配置
            system_config = config_data.get('system', {})
            self.module_b_path = system_config.get('module_b_path', "modules/try.py")
            self.trigger_cooldown = system_config.get('trigger_cooldown', 30)
            self.max_concurrent_triggers = system_config.get('max_concurrent_triggers', 1)
            self.enable_logging = system_config.get('enable_logging', True)
            self.log_level = system_config.get('log_level', "INFO")
            
            # FRS配置
            frs_config = config_data.get('frs', {})
            self.THRESHOLD_COSINE = frs_config.get('threshold_cosine', 0.65)
            self.THRESHOLD_HIGH_CONFIDENCE = frs_config.get('threshold_high_confidence', 0.75)
            self.GENDER_CONFIDENCE_THRESHOLD = frs_config.get('gender_confidence_threshold', 0.8)
            self.MIN_CONSECUTIVE_FRAMES = frs_config.get('min_consecutive_frames', 5)
            self.IDENTITY_HISTORY_LEN = frs_config.get('identity_history_len', 7)
            self.EMBEDDING_EMA_ALPHA = frs_config.get('embedding_ema_alpha', 0.5)
            self.PROCESSING_INTERVAL = frs_config.get('processing_interval', 3)
            self.MAX_FACES = frs_config.get('max_faces', 5)
            self.MAX_MISSED_FRAMES = frs_config.get('max_missed_frames', 15)
            self.MAX_MISSED_FRAMES_NEW = frs_config.get('max_missed_frames_new', 8)
            self.MIN_DETECTION_COUNT = frs_config.get('min_detection_count', 3)
            
            # 摄像头配置
            camera_config = config_data.get('camera', {})
            self.CAMERA_WIDTH = camera_config.get('camera_width', 640)
            self.CAMERA_HEIGHT = camera_config.get('camera_height', 480)
            self.DETECTION_WIDTH = camera_config.get('detection_width', 320)
            self.DETECTION_HEIGHT = camera_config.get('detection_height', 240)
            self.CAMERA_INDEX = camera_config.get('camera_index', 0)
            
            # 显示配置
            display_config = config_data.get('display', {})
            self.WINDOW_NAME = display_config.get('window_name', "FRS Trigger System")
            self.WINDOW_WIDTH = display_config.get('window_width', 800)
            self.WINDOW_HEIGHT = display_config.get('window_height', 600)
            self.SHOW_FPS = display_config.get('show_fps', True)
            self.SHOW_FACE_COUNT = display_config.get('show_face_count', True)
            
            # 路径配置
            paths_config = config_data.get('paths', {})
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.REF_IMAGE_DIR = os.path.join(base_dir, paths_config.get('ref_image_dir', "Img"))
            self.MODEL_DIR = os.path.join(base_dir, paths_config.get('model_dir', "models"))
            self.LOG_DIR = os.path.join(base_dir, paths_config.get('log_dir', "logs"))
            
            # 颜色配置
            self.COLORS = [
                (255, 0, 0),    # 蓝色
                (0, 255, 0),    # 绿色
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 紫色
            ]
            
            # 模型均值
            self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
            
            # 性别列表
            self.GENDER_LIST = ['Male', 'Female']
            
        except FileNotFoundError:
            print(f"配置文件 {self.config_file} 未找到，使用默认配置")
            self._load_default_config()
        except Exception as e:
            print(f"加载配置文件时出错: {e}，使用默认配置")
            self._load_default_config()
    
    def _load_default_config(self):
        """加载默认配置"""
        # 系统配置
        self.module_b_path = "modules/try.py"
        self.trigger_cooldown = 30
        self.max_concurrent_triggers = 1
        self.enable_logging = True
        self.log_level = "INFO"
        
        # FRS配置
        self.THRESHOLD_COSINE = 0.65
        self.THRESHOLD_HIGH_CONFIDENCE = 0.75
        self.GENDER_CONFIDENCE_THRESHOLD = 0.8
        self.MIN_CONSECUTIVE_FRAMES = 5
        self.IDENTITY_HISTORY_LEN = 7
        self.EMBEDDING_EMA_ALPHA = 0.5
        self.PROCESSING_INTERVAL = 3
        self.MAX_FACES = 5
        self.MAX_MISSED_FRAMES = 15
        self.MAX_MISSED_FRAMES_NEW = 8
        self.MIN_DETECTION_COUNT = 3
        
        # 摄像头配置
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480
        self.DETECTION_WIDTH = 320
        self.DETECTION_HEIGHT = 240
        self.CAMERA_INDEX = 0
        
        # 显示配置
        self.WINDOW_NAME = "FRS Trigger System"
        self.WINDOW_WIDTH = 800
        self.WINDOW_HEIGHT = 600
        self.SHOW_FPS = True
        self.SHOW_FACE_COUNT = True
        
        # 路径配置
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.REF_IMAGE_DIR = os.path.join(base_dir, "Img")
        self.MODEL_DIR = os.path.join(base_dir, "models")
        self.LOG_DIR = os.path.join(base_dir, "logs")
        
        # 颜色配置
        self.COLORS = [
            (255, 0, 0),    # 蓝色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
        ]
        
        # 模型均值
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        
        # 性别列表
        self.GENDER_LIST = ['Male', 'Female']
    
    def update_module_b_path(self, new_path):
        """更新模块B路径"""
        self.module_b_path = new_path
        self._save_config()
    
    def _save_config(self):
        """保存配置到文件"""
        try:
            # 读取现有配置
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 更新模块B路径
            if 'system' not in config_data:
                config_data['system'] = {}
            config_data['system']['module_b_path'] = self.module_b_path
            
            # 保存配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                
        except Exception as e:
            print(f"保存配置文件时出错: {e}")