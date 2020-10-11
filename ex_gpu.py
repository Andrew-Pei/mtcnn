from src import detect_faces, show_bboxes
from PIL import Image
import os, sys, stat
import glob
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader,Dataset
import numpy as np 
from torchvision import transforms

from skimage import io,transform
#import tqdm

BATCH_SIZE = 32      # 批训练的数据个数

class DFDCData(Data.Dataset): #继承Dataset
    def __init__(self, root_dir, transform=None): #__init__是初始化该类的一些基础参数
        self.root_dir = root_dir   #文件目录
        self.transform = transform #变换
        self.images = os.listdir(self.root_dir)#目录里的所有文件
    
    def __len__(self):#返回整个数据集的大小
        return len(self.images)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        image_index = self.images[index]#根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)#获取索引为index的图片的路径名
        print(img_path)
        img = Image.open(img_path)# 读取该图片
        img=np.array(img)
        label = "face"#img_path.split('\\')[-1].split('.')[0]# 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。我这里是"E:\\Python Project\\Pytorch\\dogs-vs-cats\\train\\cat.0.jpg"，所以先用"\\"分割，选取最后一个为['cat.0.jpg']，然后使用"."分割，选取[cat]作为该图片的标签
        sample = {'image':img,'label':label}#根据图片和标签创建字典
        
        if self.transform:
            sample = self.transform(sample)#对样本进行变换
        return sample #返回该样本
        #return img

if __name__=='__main__':
    data = DFDCData('/data1/pbw_deepfake/test/2/aalyqplqns_jpg',transform=None)#初始化类，设置数据集所在路径以及变换
    workers = 2
    dataloader = DataLoader(data,batch_size=24,shuffle=False,num_workers=workers,pin_memory=False)#使用DataLoader加载数据
    for i_batch,batch_data in enumerate(dataloader):
        print(i_batch)#打印batch编号
        #print(batch_data['image'].size())#打印该batch里面图片的大小
        #print(batch_data['label'])#打印该batch里面图片的标签
        #print('the first:',type(batch_data['image']))
        imgs=[]
        for i in range(batch_data['image'].shape[0]):
            tensor=torch.transpose(batch_data['image'][i],1,2)
            tensor=torch.transpose(tensor,0,1)
            img=transforms.ToPILImage()(tensor).convert('RGB')
            imgs.append(img)
        #torch.transpose(batch_data['image'],1,3)
        #print(batch_data['image'].shape)
        #torch.transpose(batch_data['image'],2,3)
        #print(batch_data['image'].shape)
        #print('the second:',type(imgs))
        bounding_boxes, landmarks = detect_faces(imgs)