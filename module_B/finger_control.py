#!/usr/bin/env python3
"""
模块B程序 - 手指控制
根据识别到的人脸身份伸出对应序号的手指
1号：伸食指
2号：伸食指+中指
3号：伸食指+中指+无名指
"""

import os
import sys
import time
import signal
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading

# 导入自定义消息和服务
from interface_pkg.srv import (
    SetFingerPositions, 
    ClearError, 
    RobotEnableControl,
    GlobalSpeedSet,
    MoveToJointPositions,
    MotControl,
    NavControl
)
from interface_pkg.msg import Robotstatus, MotFeedback, NavStatus

class FingerController(Node):
    def __init__(self):
        super().__init__('finger_controller')
        self.logger = self.get_logger()
        
        # 创建服务客户端
        self.callback_group = ReentrantCallbackGroup()
        
        self.finger_client = self.create_client(
            SetFingerPositions, '/set_finger_positions', callback_group=self.callback_group
        )
        self.clear_error_client = self.create_client(
            ClearError, '/robot_clear_error', callback_group=self.callback_group
        )
        self.enable_control_client = self.create_client(
            RobotEnableControl, '/robot_enable_control', callback_group=self.callback_group
        )
        self.global_speed_client = self.create_client(
            GlobalSpeedSet, '/global_speed_set', callback_group=self.callback_group
        )
        self.move_joints_client = self.create_client(
            MoveToJointPositions, '/move_joint_positions', callback_group=self.callback_group
        )
        self.motor_control_client = self.create_client(
            MotControl, '/motor_control', callback_group=self.callback_group
        )
        
        # 创建话题订阅
        self.robot_status_sub = self.create_subscription(
            Robotstatus, 'robot_status', self.robot_status_callback, 10
        )
        self.motor_feedback_sub = self.create_subscription(
            MotFeedback, 'motor_feedback', self.motor_feedback_callback, 10
        )
        
        # 状态变量
        self.robot_status = None
        self.motor_feedback = None
        self.is_initialized = False
        
    def wait_for_services(self, timeout_sec=30):
        """等待所有服务可用，带超时处理"""
        services = [
            (self.finger_client, '/set_finger_positions'),
            (self.clear_error_client, '/robot_clear_error'),
            (self.enable_control_client, '/robot_enable_control'),
            (self.global_speed_client, '/global_speed_set'),
            (self.move_joints_client, '/move_joint_positions'),
            (self.motor_control_client, '/motor_control')
        ]
        
        start_time = time.time()
        all_services_available = False
        
        while not all_services_available and (time.time() - start_time) < timeout_sec:
            all_services_available = True
            
            for service, name in services:
                if not service.service_is_ready():
                    self.logger.info(f'等待服务 {name}...')
                    all_services_available = False
                    time.sleep(0.5)
                    break  # 跳出内层循环，重新检查所有服务
            
            if all_services_available:
                self.logger.info("所有服务都已就绪!")
                return True
        
        # 检查哪些服务没有就绪
        for service, name in services:
            if not service.service_is_ready():
                self.logger.error(f"服务 {name} 超时未就绪!")
        
        return False
    
    # 回调函数
    def robot_status_callback(self, msg):
        self.robot_status = msg
        
    def motor_feedback_callback(self, msg):
        self.motor_feedback = msg
    
    # 机器人状态检查
    def get_robot_status(self, timeout=30):
        """等待机器人状态正常"""
        start_time = time.time()
        self.logger.info("等待手臂状态正常...")
        
        while time.time() - start_time < timeout:
            if self.robot_status is not None:
                is_alarming = self.robot_status.is_alarming
                joint_move_complete = self.robot_status.joint_move_complete
                
                self.logger.info(f"手臂状态: is_alarming={is_alarming}, joint_move_complete={joint_move_complete}")
                
                if not is_alarming and joint_move_complete:
                    elapsed = time.time() - start_time
                    self.logger.info(f"手臂状态正常! 耗时: {elapsed:.2f}秒")
                    return True
            time.sleep(0.5)
        
        self.logger.error("等待手臂状态超时!")
        return False
    
    def get_motor_status(self, timeout=30):
        """等待电机状态正常"""
        start_time = time.time()
        self.logger.info("等待电机状态正常...")
        
        while time.time() - start_time < timeout:
            if self.motor_feedback is not None:
                waist_ready = self.motor_feedback.waist_ready
                ascend_ready = self.motor_feedback.ascend_ready
                
                if waist_ready == 1 and ascend_ready == 1:
                    elapsed = time.time() - start_time
                    self.logger.info(f"电机状态正常! 耗时: {elapsed:.2f}秒")
                    return True
            time.sleep(0.5)
        
        self.logger.error("等待电机状态超时!")
        return False
    
    def initialize_robot(self):
        """初始化机器人"""
        if self.is_initialized:
            return True
        
        self.logger.info("初始化机器人...")
        
        # 清除错误
        if not self.clear_error_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("清除错误服务未就绪!")
            return False
        
        clear_request = ClearError.Request(clear_error=True)
        clear_future = self.clear_error_client.call_async(clear_request)
        clear_future.add_done_callback(self._clear_response_callback)
        
        # 使能控制
        if not self.enable_control_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("使能控制服务未就绪!")
            return False
        
        enable_request = RobotEnableControl.Request(enable=True)
        enable_future = self.enable_control_client.call_async(enable_request)
        enable_future.add_done_callback(self._enable_response_callback)
        
        # 设置速度
        if not self.global_speed_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("设置速度服务未就绪!")
            return False
        
        speed_request = GlobalSpeedSet.Request(speed=40.0)
        speed_future = self.global_speed_client.call_async(speed_request)
        speed_future.add_done_callback(self._speed_response_callback)
        
        time.sleep(1)  # 等待初始化完成
        
        self.is_initialized = True
        self.logger.info("机器人初始化完成!")
        return True
    
    # 响应回调函数
    def _clear_response_callback(self, future):
        try:
            response = future.result()
            print(f'清错结果：{response.success}')
        except Exception as e:
            print(f'清错结果：false')
    
    def _enable_response_callback(self, future):
        try:
            response = future.result()
            print(f'使能结果：{response.success}')
        except Exception as e:
            print(f'使能结果：false')
    
    def _speed_response_callback(self, future):
        try:
            response = future.result()
            print(f'速度结果：{response.success}')
        except Exception as e:
            print(f'速度结果：false')
    
    def _robot_response_callback(self, future):
        try:
            response = future.result()
            print(f'运动结果：{response.success}')
        except Exception as e:
            print(f'运动结果：false')
    
    def _line_response_callback(self, future):
        try:
            response = future.result()
            print(f'升降结果：{response.success}')
        except Exception as e:
            print(f'升降结果：false')
    
    def _rhand_response_callback(self, future):
        try:
            response = future.result()
            print(f'手爪结果：{response.success}')
        except Exception as e:
            print(f'手爪结果：false')
    
    # 手臂回原点
    def robot_move_origin(self):
        """手臂回原点"""
        self.logger.info("手臂回原点...")
        
        return self.arm_move([0.0]*7, [0.0]*7)
    
    # 手臂运动
    def arm_move(self, left_joints, right_joints):
        """控制手臂运动"""
        if not self.move_joints_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("手臂运动服务未就绪!")
            return False
        
        self.logger.info(f"控制手臂运动: left={left_joints}, right={right_joints}")
        request = MoveToJointPositions.Request()
        request.left_joints = left_joints
        request.right_joints = right_joints
        
        future = self.move_joints_client.call_async(request)
        future.add_done_callback(self._robot_response_callback)
        time.sleep(0.5)
        
        # 等待状态正常
        self.get_robot_status()
        
        return True
    
    # 线运动
    def line_move(self, angle=0.0, pos=0.0, head_angle=0.0, 
                 speed_waist=1, speed_ascend=1, speed_head=1):
        """控制腰部、升降和头部运动"""
        if not self.motor_control_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("线运动服务未就绪!")
            return False
        
        self.logger.info(f"控制线运动: angle={angle}, pos={pos}, head_angle={head_angle}")
        request = MotControl.Request()
        request.angle = angle
        request.speed_waist = int(speed_waist)
        request.pos = pos
        request.speed_ascend = int(speed_ascend)
        request.head_angle = head_angle
        request.speed_head = int(speed_head)
        
        future = self.motor_control_client.call_async(request)
        future.add_done_callback(self._line_response_callback)
        time.sleep(0.5)
        
        # 等待状态正常
        self.get_motor_status()
        
        return True
    
    # 手爪控制
    def rhand_use(self, positions_left, positions_right):
        """发送手爪命令"""
        if not self.finger_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("抓服务未就绪!")
            return False
        
        self.logger.info("发送手爪命令...")
        request = SetFingerPositions.Request()
        request.positions_left = positions_left
        request.positions_right = positions_right
        
        future = self.finger_client.call_async(request)
        future.add_done_callback(self._rhand_response_callback)
        return True
    
    def set_finger_positions(self, left_positions, right_positions):
        """设置手指位置"""
        if not self.finger_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("抓服务未就绪!")
            return False
        
        request = SetFingerPositions.Request()
        request.positions_left = left_positions
        request.positions_right = right_positions
        
        future = self.finger_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        try:
            response = future.result()
            self.logger.info(f'手爪结果：{response.success}')
            return response.success
        except Exception as e:
            self.logger.error(f'手爪控制失败：{e}')
            return False
    
    def show_finger_number(self, number):
        """根据数字伸出对应数量的手指
        所有手指动作只用右手完成，左手始终保持初始状态
        """
        self.logger.info(f"显示手指数量：{number}")
        
        # 初始化机器人
        if not self.initialize_robot():
            self.logger.error("机器人初始化失败!")
            return False
        
        try:
            # 初始化动作
            if not self.line_move(angle=0.0, pos=0.0, head_angle=0.0):
                self.logger.error("初始化线运动失败!")
                return False
            
            if not self.robot_move_origin():
                self.logger.error("手臂回原点失败!")
                return False
            
            # 左手保持初始状态不变
            left_joints_f = [3676, 17837, 17606, 17654, 17486, 200]
            # 右手初始状态（所有手指收拢）
            right_joints_f = [3676, 17837, 17606, 17654, 17486, 200]
            self.rhand_use(left_joints_f, right_joints_f)
            
            # 手臂运动到展示位置
            # 左手保持原位
            left_joints = [0.0] * 7
            # 右手抬起到展示位置
            right_joints = [-174.74, 127.50, 96.14, -84.34, -82.98, -24.95, 78.53]
            if not self.arm_move(left_joints, right_joints):
                self.logger.error("手臂动作失败!")
                return False
            
            # 根据数字显示手指（只用右手）
            if number == 1:
                # 1号：右手伸出食指
                self.logger.info("1号人识别 - 右手伸出食指")
                right_joints_f = [276, 17837, 9781, 10138, 9886, 200]
                self.rhand_use(left_joints_f, right_joints_f)
                time.sleep(1)
            elif number == 2:
                # 2号：右手伸出食指+中指
                self.logger.info("2号人识别 - 右手伸出食指+中指")
                right_joints_f = [276, 17837, 17606, 10138, 9886, 200]
                self.rhand_use(left_joints_f, right_joints_f)
                time.sleep(1)
            elif number == 3:
                # 3号：右手伸出食指+中指+无名指
                self.logger.info("3号人识别 - 右手伸出食指+中指+无名指")
                right_joints_f = [276, 17837, 17606, 17654, 9886, 200]
                self.rhand_use(left_joints_f, right_joints_f)
                time.sleep(1)
            
            # 保持动作片刻
            time.sleep(2)
            
            # 恢复初始手爪位置
            right_joints_f = [3676, 17837, 17606, 17654, 17486, 200]
            self.rhand_use(left_joints_f, right_joints_f)
            time.sleep(1)
            
            # 手臂回原点
            if not self.arm_move([0.0]*7, [0.0]*7):
                self.logger.error("手臂回原点失败!")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"显示手指数量失败：{e}")
            return False
    
    def reset_position(self):
        """重置机器人位置，双手自然下垂"""
        self.logger.info("重置机器人位置，双手自然下垂")
        
        try:
            # 初始化机器人
            if not self.initialize_robot():
                self.logger.error("机器人初始化失败!")
                return False
            
            # 手臂回原点（双手自然下垂）
            if not self.arm_move([0.0]*7, [0.0]*7):
                self.logger.error("手臂回原点失败!")
                return False
            
            # 双手手指自然收拢
            natural_left = [3676, 17837, 17606, 17654, 17486, 200]
            natural_right = [3676, 17837, 17606, 17654, 17486, 200]
            
            return self.set_finger_positions(natural_left, natural_right)
        except Exception as e:
            self.logger.error(f"重置位置失败: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    
    # 创建控制器实例
    controller = FingerController()
    
    # 使用多线程执行器
    executor = MultiThreadedExecutor()
    executor.add_node(controller)
    
    # 在后台运行执行器
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    # 注册信号处理
    def signal_handler(sig, frame):
        controller.logger.info("接收到终止信号，关闭中...")
        controller.reset_position()
        executor.shutdown()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 等待服务就绪
        if not controller.wait_for_services():
            controller.logger.error("服务初始化失败，退出程序")
            return 1
        
        # 获取识别到的身份
        identity = os.environ.get('FRS_TRIGGER_IDENTITY', 'Unknown')
        confidence = os.environ.get('FRS_TRIGGER_CONFIDENCE', '0.0')
        
        controller.logger.info(f"识别到身份: {identity} (置信度: {confidence})")
        
        # 根据身份显示对应数量的手指
        # 身份格式为文件名（不含扩展名），例如"1"、"2"、"3"
        if identity == '1号' or identity == '1':
            controller.logger.info("识别到1号人")
            controller.show_finger_number(1)
        elif identity == '2号' or identity == '2':
            controller.logger.info("识别到2号人")
            controller.show_finger_number(2)
        elif identity == '3号' or identity == '3':
            controller.logger.info("识别到3号人")
            controller.show_finger_number(3)
        else:
            controller.logger.info(f"未识别的身份: {identity}，双手自然下垂")
            controller.reset_position()
        
        # 保持运行一段时间，让动作完成
        time.sleep(2)
        
        # 重置位置
        controller.reset_position()
        
    except KeyboardInterrupt:
        controller.logger.info("用户中断操作")
    except Exception as e:
        controller.logger.error(f"发生异常: {str(e)}")
    finally:
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
