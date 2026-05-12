import cv2
import time
import logging


class Visualization:
    """可视化类"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # FPS计算
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def setup_window(self):
        """设置显示窗口"""
        cv2.namedWindow(self.config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.config.WINDOW_NAME, self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT)
        self.logger.info(f"显示窗口已设置: {self.config.WINDOW_NAME}")
    
    def add_system_info(self, display_frame, face_tracker, event_publisher=None):
        """添加系统信息"""
        y_offset = 30
        
        # 显示人脸数量
        if self.config.SHOW_FACE_COUNT:
            cv2.putText(display_frame, f"Faces: {len(face_tracker.trackers)}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30
        
        # 显示FPS
        if self.config.SHOW_FPS:
            self._update_fps()
            cv2.putText(display_frame, f"FPS: {self.current_fps:.1f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30
        
        # 显示触发统计
        if event_publisher:
            stats = event_publisher.get_trigger_statistics()
            cv2.putText(display_frame, f"Triggers: {stats['total_triggers']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 30
            
            cv2.putText(display_frame, f"Active: {stats['currently_triggered_faces']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def add_face_info(self, display_frame, fdata, x1, y1):
        """添加人脸信息文本"""
        text_y = max(y1 - 10, 10)
        
        if fdata['last_display_text']:
            cv2.putText(display_frame, f"{fdata['last_display_text']}",
                        (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fdata['color'], 2)
            text_y -= 30
            
        if fdata['last_gender_text']:
            cv2.putText(display_frame, f"Gender: {fdata['last_gender_text']}",
                        (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fdata['color'], 2)
            
        # 显示置信度
        if fdata['last_identity_score'] > 0:
            confidence_text = f"Conf: {fdata['last_identity_score']:.3f}"
            cv2.putText(display_frame, confidence_text,
                        (x1, text_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fdata['color'], 1)
    
    def draw_face_box(self, display_frame, x1, y1, x2, y2, color, is_triggered=False):
        """绘制人脸边界框"""
        # 根据是否已触发选择不同的线条样式
        thickness = 3 if is_triggered else 2
        line_type = cv2.LINE_AA if is_triggered else cv2.LINE_8
        
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness, line_type)
        
        # 如果已触发，添加特殊标记
        if is_triggered:
            # 在左上角添加触发标记
            trigger_size = 15
            cv2.rectangle(display_frame, (x1, y1), (x1 + trigger_size, y1 + trigger_size), 
                        (0, 255, 0), -1)
            cv2.putText(display_frame, "T", (x1 + 3, y1 + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def show_frame(self, display_frame):
        """显示帧"""
        cv2.imshow(self.config.WINDOW_NAME, display_frame)
    
    def _update_fps(self):
        """更新FPS计算"""
        self.fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed >= 1.0:  # 每秒更新一次
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def add_status_info(self, display_frame, module_b_launcher=None):
        """添加状态信息"""
        y_offset = display_frame.shape[0] - 60
        
        if module_b_launcher:
            status = module_b_launcher.get_status()
            
            # 模块B状态
            if status['is_running']:
                status_text = f"ModuleB: RUNNING (PID: {status.get('process_id', 'N/A')})"
                color = (0, 255, 0)  # 绿色
            else:
                status_text = "ModuleB: STOPPED"
                color = (0, 0, 255)  # 红色
            
            cv2.putText(display_frame, status_text, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
            
            # 统计信息
            stats_text = f"Launch: {status['launch_count']} | Success: {status['success_count']} | Fail: {status['failure_count']}"
            cv2.putText(display_frame, stats_text, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def add_help_text(self, display_frame):
        """添加帮助文本"""
        help_texts = [
            "Press 'q' to quit",
            "Press 'r' to reset triggers",
            "Press 's' to show statistics"
        ]
        
        y_offset = display_frame.shape[0] - 80
        x_offset = display_frame.shape[1] - 250
        
        for i, text in enumerate(help_texts):
            cv2.putText(display_frame, text, 
                       (x_offset, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)