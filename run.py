#!/usr/bin/env python3
"""
FRS触发系统一键启动脚本
"""

import os
import sys
import subprocess
import logging
from pathlib import Path


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 7):
        print("错误: 需要Python 3.7或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    return True


def check_dependencies():
    """检查依赖"""
    logger = logging.getLogger(__name__)
    
    # 检查必需的包
    required_packages = [
        'torch',
        'torchvision',
        'opencv-python',
        'facenet-pytorch',
        'numpy',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("缺少以下依赖包:")
        for package in missing_packages:
            logger.error(f"  - {package}")
        
        logger.info("请运行以下命令安装依赖:")
        logger.info(f"pip install -r requirements.txt")
        return False
    
    # 检查NumPy版本兼容性
    try:
        import numpy as np
        import torch
        # 检查NumPy版本是否与PyTorch兼容
        if hasattr(torch, '__version__'):
            torch_version = torch.__version__.split('.')
            major_version = int(torch_version[0])
            if major_version >= 2 and np.__version__.startswith('2.'):
                logger.warning("检测到NumPy 2.x与PyTorch 2.x兼容性问题")
                logger.warning("建议降级NumPy到1.x版本或升级PyTorch到兼容版本")
                # 不返回False，允许用户继续，但给出警告
    except Exception as e:
        logger.warning(f"检查NumPy兼容性时出错: {e}")
    
    logger.info("所有依赖包检查通过")
    return True


def check_camera():
    """检查摄像头"""
    logger = logging.getLogger(__name__)
    
    try:
        import cv2
        #cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture("rtsp://192.168.1.16:8554/live")
        if cap.isOpened():
            cap.release()
            logger.info("摄像头检查通过")
            return True
        else:
            logger.error("无法打开摄像头")
            return False
    except Exception as e:
        logger.error(f"摄像头检查失败: {e}")
        return False


def check_model_files():
    """检查模型文件"""
    logger = logging.getLogger(__name__)
    
    model_dir = Path("models")
    required_files = [
        "gender_deploy.prototxt",
        "gender_net.caffemodel"
    ]
    
    if not model_dir.exists():
        logger.error(f"模型目录不存在: {model_dir}")
        return False
    
    missing_files = []
    for file_name in required_files:
        file_path = model_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.error("缺少以下模型文件:")
        for file_name in missing_files:
            logger.error(f"  - {file_name}")
        return False
    
    logger.info("模型文件检查通过")
    return True


def check_reference_images():
    """检查参考图像"""
    logger = logging.getLogger(__name__)
    
    img_dir = Path("Img")
    if not img_dir.exists():
        logger.warning("参考图像目录不存在，将创建空目录")
        img_dir.mkdir(exist_ok=True)
        return True
    
    # 检查图像文件
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in img_dir.iterdir() 
                  if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning("参考图像目录为空，系统将无法识别人脸")
        logger.info("请将参考人脸图像放入Img目录")
        return True
    
    logger.info(f"找到 {len(image_files)} 个参考图像")
    return True


def check_config():
    """检查配置文件"""
    logger = logging.getLogger(__name__)
    
    config_file = Path("config.yaml")
    if not config_file.exists():
        logger.error("配置文件不存在: config.yaml")
        return False
    
    try:
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置项
        required_keys = [
            'system.module_b_path',
            'frs.threshold_cosine',
            'camera.camera_index'
        ]
        
        for key in required_keys:
            keys = key.split('.')
            current = config
            try:
                for k in keys:
                    current = current[k]
            except (KeyError, TypeError):
                logger.error(f"配置文件缺少关键项: {key}")
                return False
        
        logger.info("配置文件检查通过")
        return True
        
    except Exception as e:
        logger.error(f"配置文件检查失败: {e}")
        return False


def check_module_b():
    """检查模块B文件"""
    logger = logging.getLogger(__name__)
    
    try:
        import yaml
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        module_b_path = config.get('system', {}).get('module_b_path', 'modules/try.py')
        
        if not Path(module_b_path).exists():
            logger.error(f"模块B文件不存在: {module_b_path}")
            return False
        
        logger.info(f"模块B文件检查通过: {module_b_path}")
        return True
        
    except Exception as e:
        logger.error(f"模块B检查失败: {e}")
        return False


def install_dependencies():
    """安装依赖"""
    logger = logging.getLogger(__name__)
    
    logger.info("正在安装依赖包...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"依赖包安装失败: {e}")
        return False


def setup_environment():
    """设置环境"""
    logger = logging.getLogger(__name__)
    
    # 设置环境变量
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    os.environ.setdefault('DISPLAY', ':0')
    
    # 创建必要的目录
    directories = ['logs', 'Img', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info("环境设置完成")
    return True


def main():
    """主函数"""
    logger = setup_logging()
    
    logger.info("=" * 50)
    logger.info("FRS触发系统启动检查")
    logger.info("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        return 1
    
    # 检查依赖
    if not check_dependencies():
        logger.info("尝试自动安装依赖...")
        if not install_dependencies():
            logger.error("依赖安装失败，请手动安装")
            return 1
    
    # 检查环境
    if not setup_environment():
        return 1
    
    # 检查配置文件
    if not check_config():
        return 1
    
    # 检查模型文件
    if not check_model_files():
        return 1
    
    # 检查参考图像
    if not check_reference_images():
        return 1
    
    # 检查模块B
    if not check_module_b():
        return 1
    
    # 检查摄像头
    if not check_camera():
        logger.error("摄像头检查失败，请检查摄像头连接和权限")
        logger.info("Linux用户可能需要添加到video组:")
        logger.info("  sudo usermod -a -G video $USER")
        logger.info("  然后重新登录或运行: newgrp video")
        return 1
    
    logger.info("=" * 50)
    logger.info("所有检查通过，启动FRS触发系统...")
    logger.info("=" * 50)
    
    try:
        # 导入并启动主控制器
        from main_controller import main as controller_main
        return controller_main()
        
    except ImportError as e:
        logger.error(f"无法导入主控制器: {e}")
        return 1
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
