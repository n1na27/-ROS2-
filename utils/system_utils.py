"""
系统工具函数
"""

import os
import sys
import platform
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional


def get_system_info() -> Dict[str, str]:
    """
    获取系统信息
    
    Returns:
        包含系统信息的字典
    """
    return {
        'platform': platform.platform(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'python_version': platform.python_version(),
        'architecture': platform.architecture()[0]
    }


def check_camera_permissions() -> bool:
    """
    检查摄像头权限
    
    Returns:
        是否有摄像头权限
    """
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            return True
        return False
    except Exception:
        return False


def get_camera_devices() -> List[str]:
    """
    获取可用的摄像头设备列表
    
    Returns:
        摄像头设备路径列表
    """
    devices = []
    
    if platform.system() == "Linux":
        # Linux系统
        try:
            result = subprocess.run(['ls', '/dev/video*'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                devices = result.stdout.strip().split('\n')
                devices = [d for d in devices if d.strip()]
        except Exception:
            pass
    
    elif platform.system() == "Windows":
        # Windows系统
        try:
            import cv2
            for i in range(10):  # 检查前10个摄像头索引
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    devices.append(f"Camera {i}")
                    cap.release()
                else:
                    break
        except Exception:
            pass
    
    elif platform.system() == "Darwin":
        # macOS系统
        try:
            result = subprocess.run(['system_profiler', 'SPCameraDataType'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # 解析系统信息获取摄像头
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Camera' in line and ':' in line:
                        device_name = line.split(':')[1].strip()
                        if device_name:
                            devices.append(device_name)
        except Exception:
            pass
    
    return devices


def setup_camera_permissions_linux() -> bool:
    """
    在Linux系统上设置摄像头权限
    
    Returns:
        是否设置成功
    """
    if platform.system() != "Linux":
        return True
    
    try:
        # 检查用户是否在video组中
        result = subprocess.run(['groups'], capture_output=True, text=True)
        if 'video' in result.stdout:
            return True
        
        # 尝试添加用户到video组
        username = os.environ.get('USER', '')
        if username:
            subprocess.run(['sudo', 'usermod', '-a', '-G', 'video', username])
            print(f"已将用户 {username} 添加到video组")
            print("请重新登录或运行 'newgrp video' 以使权限生效")
            return True
        
    except Exception as e:
        print(f"设置摄像头权限失败: {e}")
        return False


def check_display_environment() -> bool:
    """
    检查显示环境
    
    Returns:
        是否有可用的显示环境
    """
    if platform.system() == "Linux":
        # 检查DISPLAY环境变量
        if not os.environ.get('DISPLAY'):
            return False
        
        # 检查X11是否可用
        try:
            result = subprocess.run(['xset', 'q'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    return True  # 其他系统通常有显示环境


def create_directories(directories: List[str]) -> bool:
    """
    创建必要的目录
    
    Args:
        directories: 目录路径列表
        
    Returns:
        是否创建成功
    """
    success = True
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"创建目录失败 {directory}: {e}")
            success = False
    return success


def check_disk_space(path: str = '.', min_space_mb: int = 100) -> bool:
    """
    检查磁盘空间
    
    Args:
        path: 检查路径
        min_space_mb: 最小所需空间（MB）
        
    Returns:
        是否有足够空间
    """
    try:
        stat = os.statvfs(path)
        free_space = stat.f_bavail * stat.f_frsize
        free_space_mb = free_space / (1024 * 1024)
        return free_space_mb >= min_space_mb
    except Exception:
        return True  # 如果无法检查，假设有足够空间


def get_network_interfaces() -> List[str]:
    """
    获取网络接口列表
    
    Returns:
        网络接口名称列表
    """
    interfaces = []
    
    try:
        if platform.system() == "Linux":
            result = subprocess.run(['ip', 'link', 'show'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if ': <' in line:
                        interface = line.split(':')[1].strip()
                        if interface != 'lo':  # 排除回环接口
                            interfaces.append(interface)
        
        elif platform.system() == "Windows":
            result = subprocess.run(['ipconfig', '/all'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'adapter' in line.lower():
                        interface = line.split(':')[0].strip()
                        interfaces.append(interface)
        
        elif platform.system() == "Darwin":
            result = subprocess.run(['ifconfig', '-l'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                interfaces = result.stdout.strip().split(' ')
                interfaces = [i for i in interfaces if i and i != 'lo']
    
    except Exception:
        pass
    
    return interfaces


def check_python_packages(packages: List[str]) -> Dict[str, bool]:
    """
    检查Python包是否已安装
    
    Args:
        packages: 包名列表
        
    Returns:
        包名到安装状态的映射
    """
    result = {}
    
    for package in packages:
        try:
            # 处理包名中的连字符
            import_name = package.replace('-', '_')
            __import__(import_name)
            result[package] = True
        except ImportError:
            result[package] = False
    
    return result


def install_python_package(package: str) -> bool:
    """
    安装Python包
    
    Args:
        package: 包名
        
    Returns:
        是否安装成功
    """
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', package
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def get_system_resources() -> Dict[str, any]:
    """
    获取系统资源信息
    
    Returns:
        系统资源信息字典
    """
    resources = {}
    
    try:
        import psutil
        
        # CPU信息
        resources['cpu_count'] = psutil.cpu_count()
        resources['cpu_percent'] = psutil.cpu_percent(interval=1)
        
        # 内存信息
        memory = psutil.virtual_memory()
        resources['memory_total'] = memory.total
        resources['memory_available'] = memory.available
        resources['memory_percent'] = memory.percent
        
        # 磁盘信息
        disk = psutil.disk_usage('.')
        resources['disk_total'] = disk.total
        resources['disk_free'] = disk.free
        resources['disk_percent'] = disk.percent
        
    except ImportError:
        # 如果没有psutil，使用基本检查
        resources['cpu_count'] = os.cpu_count() or 1
        resources['memory_total'] = 'unknown'
        resources['memory_available'] = 'unknown'
        resources['disk_total'] = 'unknown'
        resources['disk_free'] = 'unknown'
    
    return resources


def validate_system_requirements() -> Dict[str, bool]:
    """
    验证系统要求
    
    Returns:
        要求检查结果
    """
    requirements = {
        'python_version': sys.version_info >= (3, 7),
        'camera_permission': check_camera_permissions(),
        'display_environment': check_display_environment(),
        'disk_space': check_disk_space(),
        'network_available': len(get_network_interfaces()) > 0
    }
    
    return requirements