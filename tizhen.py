#coding=gbk
import cv2
import os, sys, stat
import time
from tqdm import tqdm
import pandas as pd

#提取离散帧和连续帧

#离散帧
#real：5-250
#fake：5-50

#连续帧
#real：50-99
#fake：90-99

discrete_real_list=[]
for i in range(5,251,5):
    discrete_real_list.append(i)
discrete_fake_list=[]
for i in range(5,51,5):
    discrete_fake_list.append(i)
consecutive_real_list=[]
for i in range(50,100):
    consecutive_real_list.append(i)
consecutive_fake_list=[]
for i in range(90,100):
    consecutive_fake_list.append(i)
#print(discrete_real_list,"\n",discrete_fake_list,"\n",consecutive_real_list,"\n",consecutive_fake_list)


#主程序
for num in range(13,25):
    metadata = pd.read_json(
        "/data1/pbw_deepfake/train/dfdc_train_part_%d/dfdc_train_part_%d/metadata.json" % (num, num))
    # metadata
    g = os.walk(
        "/data1/pbw_deepfake/train/dfdc_train_part_%d/dfdc_train_part_%d/" % (num, num))
    for path, d, filelist in g:
        for filename in tqdm(filelist):
            if filename.endswith('mp4'):
                a = os.path.join(path, filename)
                #print(a)
                if not os.path.exists("/data2/pbw_dfdc/experiment_1/consecutive/train/"+metadata.loc['label'][filename]+"/pic_"+str(num)+"/"+filename[:-4]+"_jpg"):
                    os.mkdir("/data2/pbw_dfdc/experiment_1/consecutive/train/"+metadata.loc['label'][filename]+"/pic_"+str(num)+"/"+filename[:-4]+"_jpg", 777)
                    os.chmod("/data2/pbw_dfdc/experiment_1/consecutive/train/"+metadata.loc['label'][filename]+"/pic_"+str(num)+"/"+filename[:-4]+"_jpg", stat.S_IRWXU+stat.S_IRWXG+stat.S_IRWXO)
                if not os.path.exists("/data2/pbw_dfdc/experiment_1/discrete/train/"+metadata.loc['label'][filename]+"/pic_"+str(num)+"/"+filename[:-4]+"_jpg"):
                    os.mkdir("/data2/pbw_dfdc/experiment_1/discrete/train/"+metadata.loc['label'][filename]+"/pic_"+str(num)+"/"+filename[:-4]+"_jpg", 777)
                    os.chmod("/data2/pbw_dfdc/experiment_1/discrete/train/"+metadata.loc['label'][filename]+"/pic_"+str(num)+"/"+filename[:-4]+"_jpg", stat.S_IRWXU+stat.S_IRWXG+stat.S_IRWXO)
                cap = cv2.VideoCapture("/data1/pbw_deepfake/train/dfdc_train_part_%d/dfdc_train_part_%d/" % (num, num)+filename)
                suc = cap.isOpened()
                frame_count = 0
                #print(suc)
                while suc:
                    frame_count += 1
                    suc, frame = cap.read()
                    #print(frame_count)
                    if metadata.loc['label'][filename]=='FAKE':
                        if suc and frame_count in consecutive_fake_list:
                            #print('consecutive_fake_list')
                            cv2.imwrite(os.path.join("/data2/pbw_dfdc/experiment_1/consecutive/train/"+metadata.loc['label'][filename]+"/pic_"+str(num)+"/"+filename[:-4]+"_jpg/%d.jpg" % frame_count), frame)
                        if suc and frame_count in discrete_fake_list:
                            #print('discrete_fake_list')
                            cv2.imwrite(os.path.join("/data2/pbw_dfdc/experiment_1/discrete/train/"+metadata.loc['label'][filename]+"/pic_"+str(num)+"/"+filename[:-4]+"_jpg/%d.jpg" % frame_count), frame)
                    else:
                        if suc and frame_count in consecutive_real_list:
                            #print('consecutive_real_list')
                            cv2.imwrite(os.path.join("/data2/pbw_dfdc/experiment_1/consecutive/train/"+metadata.loc['label'][filename]+"/pic_"+str(num)+"/"+filename[:-4]+"_jpg/%d.jpg" % frame_count), frame)
                        if suc and frame_count in discrete_real_list:
                            #print('discrete_real_list')
                            cv2.imwrite(os.path.join("/data2/pbw_dfdc/experiment_1/discrete/train/"+metadata.loc['label'][filename]+"/pic_"+str(num)+"/"+filename[:-4]+"_jpg/%d.jpg" % frame_count), frame)
                cap.release()
                #print("unlock mp4: ", frame_count-1)