import os
from collections import Counter, deque
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceRecognizer:
    """人脸识别器类"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.ref_features = {}
        self.identity_attribute_map = {}
        
        # 初始化模型
        self.mtcnn = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=40,
            thresholds=[0.7, 0.8, 0.9],
            factor=0.85,
            post_process=True  
        )
        
        # 此处加载权重文件（*.pt）
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device) 

        # 加载性别模型
        self._load_gender_model()
        
    def _load_gender_model(self):
        """加载性别预测模型"""
        genderProto = os.path.join(self.config.MODEL_DIR, "gender_deploy.prototxt")
        genderModel = os.path.join(self.config.MODEL_DIR, "gender_net.caffemodel")
        
        if not os.path.exists(genderProto) or not os.path.exists(genderModel):
            raise FileNotFoundError("Gender model files not found. Please check the models directory.")
        
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)
    
    def load_reference_features(self):
        """从Img文件夹加载参考人脸特征"""
        img_dir = self.config.REF_IMAGE_DIR
        if not os.path.exists(img_dir):
            print(f"Warning: Reference image directory {img_dir} does not exist")
            os.makedirs(img_dir, exist_ok=True)
            print(f"Created directory: {img_dir}")
            return {}
        
        ref_feats = {}
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"No reference images found in {img_dir}")
            return {}
        
        for img_file in image_files:
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            emb = self.extract_aligned_embedding_from_image(img)
            if emb is not None:
                # 确保emb是tensor，然后转换为tensor
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb)
                ref_feats[img_file] = emb.to(self.device)
                identity_name = os.path.splitext(img_file)[0]
                self.identity_attribute_map[identity_name] = {
                    'gender': 'Unknown',
                    'gender_confidence': 0.0,
                    'update_count': 0
                }
                print(f"Loaded reference: {img_file}")
        
        self.ref_features = ref_feats
        print(f"Successfully loaded {len(ref_feats)} reference images")
        return ref_feats
    
    def extract_aligned_embedding_from_image(self, img_bgr):
        """从图像中提取对齐的人脸特征"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(img_rgb)
        if boxes is None or len(boxes) == 0:
            return None
            
        idx = self._largest_face_index(boxes)
        faces_tensor = self.mtcnn(img_rgb)
        
        if faces_tensor is None or faces_tensor.dim() == 0:
            return None
            
        if faces_tensor.dim() == 3:
            faces_tensor = faces_tensor.unsqueeze(0)
            
        face_tensor = faces_tensor[idx:idx+1].to(self.device)
        with torch.no_grad():
            emb = self.resnet(face_tensor)
        return self._l2_normalize(emb).detach()
    
    def extract_aligned_embedding_from_region(self, face_region_bgr):
        """从人脸区域提取特征"""
        if face_region_bgr.size == 0:
            return None
            
        region_rgb = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2RGB)
        face_tensor = self.mtcnn(region_rgb)
        
        if face_tensor is None:
            return None
            
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        else:
            face_tensor = face_tensor[0:1]
            
        face_tensor = face_tensor.to(self.device)
        with torch.no_grad():
            emb = self.resnet(face_tensor)
        return self._l2_normalize(emb).detach()
    
    def recognize_face(self, face_id, face_region_bgr, face_tracker):
        """识别人脸并进行稳定化处理"""
        fdata = face_tracker.trackers[face_id]
        emb = self.extract_aligned_embedding_from_region(face_region_bgr)
        if emb is None:
            return

        # EMA平滑
        if fdata['ema_embedding'] is None:
            fdata['ema_embedding'] = emb
        else:
            fdata['ema_embedding'] = self._l2_normalize(
                self.config.EMBEDDING_EMA_ALPHA * emb + 
                (1.0 - self.config.EMBEDDING_EMA_ALPHA) * fdata['ema_embedding']
            )

        # 寻找最相似的参考人脸
        best_name, best_score = self._find_best_match(fdata['ema_embedding'])
        fdata['last_identity_score'] = best_score

        # 阈值判断
        if best_score >= self.config.THRESHOLD_COSINE:
            name_without_ext = os.path.splitext(best_name)[0]
            fdata['identity_history'].append(name_without_ext)
        else:
            fdata['identity_history'].append("Unknown")

        # 多数表决
        fdata['last_identity'] = self._majority_vote(fdata['identity_history'])
        fdata['last_display_text'] = self._generate_display_text(fdata)
    
    def predict_gender(self, face_id, face_region_bgr, face_tracker):
        """预测性别"""
        fdata = face_tracker.trackers[face_id]
        if face_region_bgr.size <= 0:
            return
        
        # 检查是否可以使用历史属性
        if self._should_use_historical_attributes(fdata):
            self._apply_historical_attributes(fdata)
            return
        
        # 正常预测
        self._predict_with_dnn(face_id, face_region_bgr, face_tracker)
    
    def _should_use_historical_attributes(self, fdata):
        """判断是否应该使用历史属性"""
        identity_confidence = fdata['last_identity_score']
        current_identity = fdata['last_identity']
        
        return (identity_confidence > self.config.THRESHOLD_HIGH_CONFIDENCE and 
                current_identity != "Unknown" and 
                current_identity in self.identity_attribute_map and
                self.identity_attribute_map[current_identity]['gender_confidence'] > self.config.GENDER_CONFIDENCE_THRESHOLD)
    
    def _apply_historical_attributes(self, fdata):
        """应用历史属性"""
        identity_data = self.identity_attribute_map[fdata['last_identity']]
        adjusted_gender_conf = identity_data['gender_confidence'] * 0.9
        
        fdata['gender_history'].append(identity_data['gender'])
        fdata['stable_gender'] = identity_data['gender']
        fdata['last_gender_text'] = f"{identity_data['gender']} ({adjusted_gender_conf*100:.1f}%)*"
    
    def _predict_with_dnn(self, face_id, face_region_bgr, face_tracker):
        """使用DNN进行性别预测"""
        fdata = face_tracker.trackers[face_id]
        
        blob = cv2.dnn.blobFromImage(face_region_bgr, 1.0, (227, 227), 
                                   self.config.MODEL_MEAN_VALUES, swapRB=False)
        self.genderNet.setInput(blob)
        genderPreds = self.genderNet.forward()

        gender_conf = float(np.max(genderPreds[0]))
        current_gender = "Unknown"
        
        if gender_conf >= self.config.GENDER_CONFIDENCE_THRESHOLD:
            current_gender = self.config.GENDER_LIST[int(np.argmax(genderPreds[0]))]

        # 更新历史
        fdata['gender_history'].append(current_gender)

        # 稳定化处理
        self._stabilize_gender_prediction(fdata, gender_conf)
        
        # 更新身份属性映射
        self._update_identity_attributes(fdata)
    
    def _stabilize_gender_prediction(self, fdata, gender_conf):
        """稳定化性别预测"""
        if len(fdata['gender_history']) >= self.config.MIN_CONSECUTIVE_FRAMES:
            g_cnt = Counter(fdata['gender_history'])
            g, g_count = g_cnt.most_common(1)[0]
            if g != "Unknown" and g_count >= len(fdata['gender_history']) * 0.6:
                fdata['stable_gender'] = g
                fdata['last_gender_text'] = f"{g} ({gender_conf*100:.1f}%)"
            else:
                fdata['last_gender_text'] = ""
        else:
            fdata['last_gender_text'] = ""
    
    def _update_identity_attributes(self, fdata):
        """更新身份属性映射"""
        identity_confidence = fdata['last_identity_score']
        current_identity = fdata['last_identity']
        
        if (identity_confidence > self.config.THRESHOLD_HIGH_CONFIDENCE and 
            current_identity != "Unknown"):
            
            if current_identity not in self.identity_attribute_map:
                self.identity_attribute_map[current_identity] = {
                    'gender': 'Unknown', 
                    'gender_confidence': 0.0, 
                    'update_count': 0
                }
            
            self._update_gender_attribute(fdata, current_identity)
    
    def _update_gender_attribute(self, fdata, current_identity):
        """更新性别属性"""
        old_data = self.identity_attribute_map[current_identity]
        g_cnt = Counter(fdata['gender_history'])
        g, g_count = g_cnt.most_common(1)[0]
        
        if g_count >= len(fdata['gender_history']) * 0.8:
            if old_data['gender'] == g:
                old_data['gender_confidence'] = min(1.0, old_data['gender_confidence'] + 0.1)
            else:
                if g_count >= len(fdata['gender_history']) * 0.9:
                    old_data['gender'] = g
                    old_data['gender_confidence'] = 0.7  # 初始置信度
            old_data['update_count'] += 1
    
    def _find_best_match(self, embedding):
        """寻找最佳匹配"""
        best_name = "Unknown"
        best_score = -1.0
        for ref_name, ref_emb in self.ref_features.items():
            score = self._cosine_similarity(embedding, ref_emb)
            if score > best_score:
                best_score = score
                best_name = ref_name
        return best_name, best_score
    
    def _majority_vote(self, identity_history):
        """多数表决"""
        if len(identity_history) >= 3:
            cnt = Counter(identity_history)
            top_name, _ = cnt.most_common(1)[0]
            return top_name
        else:
            return identity_history[-1] if identity_history else "Unknown"
    
    def _generate_display_text(self, fdata):
        """生成显示文本"""
        if fdata['last_identity'] != "Unknown":
            if fdata['stable_gender'] == "Male":
                return f"Welcome, Mr. {fdata['last_identity']}!"
            elif fdata['stable_gender'] == "Female":
                return f"Welcome, Miss. {fdata['last_identity']}!"
            else:
                return f"Welcome, {fdata['last_identity']}!"
        else:
            if fdata['stable_gender'] == "Male":
                return "Welcome, New Guest (Male)!"
            elif fdata['stable_gender'] == "Female":
                return "Welcome, New Guest (Female)!"
            else:
                return "Welcome, New Guest!"
    
    @staticmethod
    def _l2_normalize(embedding):
        """L2归一化"""
        return embedding / (embedding.norm(dim=1, keepdim=True) + 1e-10)
    
    @staticmethod
    def _cosine_similarity(a, b):
        """计算余弦相似度"""
        return float(torch.sum(a * b).item())
    
    @staticmethod
    def _largest_face_index(boxes):
        """找到最大人脸的索引"""
        areas = [(box[2]-box[0])*(box[3]-box[1]) for box in boxes]
        return int(np.argmax(areas))