import cv2
import uuid
import numpy as np
import os
import dlib
# 获取最大的人脸
def getMax_face(face_rects):
    
    if len(face_rects)==0:
        return 0,0    
    face_areas =[ ]
    for rect in face_rects:
        area = (rect.bottom()-rect.top())*(rect.right()-rect.left())
        face_areas.append(area)
    index = np.argmax(face_areas)
    return face_areas[index],face_rects[index]
    
def gen_face_name(str_face_id):
    return str_face_id+'_'+str(uuid.uuid4())+'.jpg'


def det_draw_one_face(img,model_det_face):
    # 转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸区域
    face_rects = model_det_face(gray, 0) 
    
    # 得到最大的人脸区域
    max_area,max_rect = getMax_face(face_rects)
    
    # 画框
    if max_area>10:
        cv2.rectangle(img, (max_rect.left(),max_rect.top()), 
                            (max_rect.right(),max_rect.bottom()), 
                            (255, 0, ), 3) 
    return img,max_rect,max_area


def get_id_list(path_face):
    list_faces = []
    for roots,dirs,files in os.walk(path_face):
        for dir in dirs:
            list_faces.append(dir)
    return list_faces



def file2vector(file_img,sp,facerec,b_file=True):
   
    if b_file:
        img = cv2.imread(file_img)
    else:
        img = file_img
    img = cv2.cvtColor(img, cv2. COLOR_BGR2RGB)

    # 存放的是已经截取好的人脸图片，所以在整图内检测标志点
    img = np.array(img)
    h,w,_ = np.shape(img)
    rect = dlib.rectangle(0,0,w,h)
    # 辅助人脸定位
    shape = sp(img, rect)
    # 获取128维人脸特征
    face_vector = facerec.compute_face_descriptor(img,shape)
    return face_vector


def face_recognize(face_vec,path_vectors,method):
   
    vector_files = []
    for roots,dirs,files in os.walk(path_vectors):
        for file in files:
            # print(file)
            if file.endswith(".npy"):
                file_vector = os.path.join(roots,file)
                # print(file_vector)
                vector_files.append(file_vector)
    
    scores = np.zeros(len(vector_files))
    for i,vector_file in enumerate(vector_files):
        # print(vector_file)
        vector = np.load(vector_file)
        if method == "dis_euc":
            score = np.sqrt(np.sum((face_vec-vector)**2))
        else:
            tt1 = np.dot(face_vec,vector)
            tt2 = np.dot(face_vec,face_vec)
            tt3 = np.dot(vector,vector)
            score = tt1/(np.sqrt(tt2)*np.sqrt(tt3))
        scores[i] = score

    if method == "dis_euc":
        index = np.argmin(scores)
    else:
        index = np.agrmin(scores)

    out_sore = scores[index]
    dirname = os.path.split(vector_files[index])[0]
    # print(dirname)
    str_id = os.path.split(dirname)[-1]
    return str_id,out_sore 




    









   
















         


if __name__ =="__main__":
    zz = get_id_list('D:\\工作相关\\我设计的课程\\python与人工智能课程设计\应用篇\\facerecognize\\faces')
    print(zz)

# def save_face_rect(img,rect):
#     # 截取人脸图像
#     roi = img[rect.top():rect.bottom(),rect.left():rect.right()]
#     # 生成文件名
#     save_face_name = os.path.join('faces',str_face_id,gen_face_name(str_face_id))
#     # 文件保存
#     cv2.imwrite(save_face_name,roi)
#     print('save_face',save_face_name)

    




    
    # # 用来记录所有模型信息
    # with open('model.scp','w',encoding='utf-8')as f:
    
    #     # 遍历faces文件夹
    #     base_path = 'faces'
    #     for face_id in os.listdir(base_path):
            
    #         face_dir = os.path.join(base_path,face_id)
    #         if os.path.isdir(face_dir):
    #             # 遍历 base_path/face_id 文件夹
    #             file_face_model = os.path.join(face_dir,face_id+'.npy')
    #             face_vectors = []
                
    #             for face_img in os.listdir(face_dir):
                 
    #                 if os.path.splitext(face_img)[-1]=='.jpg':
    #                     # 读取人俩图像并转换为RGB
    #                     img = cv2.imread(os.path.join(face_dir,face_img))
                      
    #                     img = cv2.cvtColor(img, cv2. COLOR_BGR2RGB)
                    
    #                     # 存放的是已经截取好的人脸图片，所以在整图内检测标志点
    #                     img = np.array(img)
    #                     h,w,_ = np.shape(img)
    #                     rect = dlib.rectangle(0,0,w,h)
    #                     # 辅助人脸定位
    #                     shape = sp(img, rect)
    #                     print("Generate face vector of ",face_img)
    #                     # 获取128维人脸特征
    #                     face_vector = facerec.compute_face_descriptor(img,shape)
                    
    #                     # 保存图像和人脸ID
    #                     face_vectors.append(face_vector)
    #             if len(face_vectors)>0:
    #                 np.save(file_face_model,face_vectors)
    #                 f.write('%s %s\n'%(face_id,file_face_model))