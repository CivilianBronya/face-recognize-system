import cv2
import os
import dlib
import numpy as np

# 根据欧式距离计算相似度
def face_recognize(face_vec,face_models,face_ids):
    
    scores = []
    for model in face_models:
        N = model.shape[0]
        diffMat = np.tile(face_vec,(N,1))-model
    
        # 计算欧式距离
        sqDiffMat=diffMat**2
        sqDistances=sqDiffMat.sum(axis=1)
        distances=sqDistances**0.5
        
        # 找到最小距离
        score = np.min(distances)
        scores.append(score)
        
    index = np.argmin(scores)    
    
    # 返回id编号与距离
    return face_ids[index],scores[index] 

# 根据cos距离计算相似度
def face_recognize_cos(face_vec,face_models,face_ids):
    
    scores = []
    for k,model in enumerate(face_models):
        N = model.shape[0]
        face_vec = face_vec-np.mean(face_vec)
        model = model-np.mean(model,axis=1,keepdims=True)
        
        tt1 = np.dot(model,face_vec) # N
        tt2 = np.dot(face_vec,face_vec) #1
        tt3 = np.sum(model*model,axis=1)#N
        cos_distance = tt1/(np.sqrt(tt2)*np.sqrt(tt3))
        # 找到最大距离
        score = np.max(cos_distance)
        scores.append(score)
        
    index = np.argmax(scores)    
    
    # 返回id编号与距离
    return face_ids[index],scores[index]




def load_model(file_scp):
    with open(file_scp,'r',encoding='utf-8') as f:
        lines = f.read().splitlines()
    face_ids = [line.split()[0] for line in lines]
    face_models =[np.load(line.split()[-1]) for line in lines]
    return face_ids, face_models

if __name__ == "__main__":
    # 加载训练好的人脸模型
    face_ids,face_models = load_model("model.scp")
    
    # Dlib 人脸检测器
    detector = dlib.get_frontal_face_detector()

    # Dlib 标志点检测器 
    sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

    # Dlib 人脸特征提取器
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:  
        # 读取一帧图像
        success, img = cap.read()
        
        # BGR 转 RGB
        img_rgb = cv2.cvtColor(img, cv2. COLOR_BGR2RGB)
        
        # 转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸区域
        face_rects = detector(gray, 0) 
       
        # 遍历检测的人脸
        for k, rect in enumerate(face_rects):
            
            # 画框
            cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0), 3)
            
            # 标志点检测
            shape = sp(img_rgb, rect)
            
            # 获取人脸特征
            face_vector = facerec.compute_face_descriptor(img_rgb, shape)
            
            
            # 进行识别返回ID与距离
            
            ''' 方法1 计算欧式距离'''
            # face_id, score = face_recognize(np.array(face_vector),face_models,face_ids)
            
            # if (score < 0.45):
                # str_face = face_id
                # str_confidence = "  %.2f"%(score)
            # else:
                # str_face = "unknown"
                # str_confidence = "  %.2f"%(score)
                
            
            ''' 方法2 计算余弦相似度'''    
            face_id, score = face_recognize_cos(np.array(face_vector),face_models,face_ids)
            str_face = face_id
            str_confidence = "  %.2f"%(score)
            
            if (score > 0.95):
                str_face = face_id
                str_confidence = "  %.2f"%(score)
            else:
                str_face = "unknown"
                str_confidence = "  %.2f"%(score)
            
            
            
            
            # 检测结果文字输出
            cv2.putText(img, str_face+str_confidence, (rect.left()+5,rect.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
           
        # 显示检测结果
        cv2.imshow("FACE",img)
        
        # 按键 "q" 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
          
    cap.release() 