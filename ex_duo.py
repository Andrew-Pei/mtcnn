from src import detect_faces, show_bboxes
from PIL import Image
import os
import sys
import stat
import glob
import multiprocessing as mp
import threading as td
import config
#from queue import Queue
#import tqdm


def job(l):
    print('%s start' % mp.current_process())
    # 把传进来的list内容都提脸，写jpg
    for dname in l:
        print(dname[27:-4])
        os.mkdir(os.path.join(
            config.str1, dname[27:-4]), 777)
        os.chmod(os.path.join(config.str1,
                 dname[27:-4]), stat.S_IRWXU+stat.S_IRWXG+stat.S_IRWXO)
        for fname in os.listdir(dname):
            # print(fname)
            # pass
            img = Image.open(os.path.join(dname, fname))
            bounding_boxes, landmarks = detect_faces(img)
            #show_bboxes(img, bounding_boxes, landmarks)
            if len(bounding_boxes):
                if bounding_boxes.size != 0:
                    bb = bounding_boxes[0]
                    # print(bb)
                    w1 = bb[0]
                    h1 = bb[1]
                    w2 = bb[2]
                    h2 = bb[3]
                    aa = []
                    w3 = w1-0.1*(w2-w1)
                    h3 = h1-0.2*(h2-h1)
                    w4 = w2+0.3*(w2-w1)
                    h4 = h2+0.2*(h2-h1)
                    if (h4-h3) > (w4-w3):
                        w3 -= 0.5*((h4-h3)-(w4-w3))
                        w4 += 0.5*((h4-h3)-(w4-w3))
                    if (h4-h3) > (w4-w3):
                        w3 -= 0.5*((h4-h3)-(w4-w3))
                        w4 += 0.5*((h4-h3)-(w4-w3))
                    if (h4-h3) > (w4-w3):
                        w3 -= 0.5*((h4-h3)-(w4-w3))
                        w4 += 0.5*((h4-h3)-(w4-w3))
                    if w3 < 0:
                        w3 = 0
                    if h3 < 0:
                        h3 = 0
                    if w4 > img.size[0]:
                        w4 = img.size[0]
                    if h4 > img.size[1]:
                        h4 = img.size[1]
                    aa.append(w3)
                    aa.append(h3)
                    aa.append(w4)
                    aa.append(h4)
                    # print(w3)
                    # print(h3)
                    # print(w4-w3)
                    # print(h4-h3)

                    face = img.crop(aa[:4])
                    # face.show()
                    # print(face.size)
                    face.save(os.path.join(
                        config.str1, dname[27:-4], fname))  # TODO
                    print("saved")
                else:
                    print("no face found!")
            else:
                print("bounding_boxes is empty!")

    print('%s finish' % mp.current_process())


def mp1():
    # q =Queue()    #q中存放返回值，代替return的返回值
    mps = []
    # data = [[1,2,3],[3,4,5],[4,4,4],[5,5,5]]
    ex_list = []
    for dname in glob.glob(config.str2, recursive=True):
        ex_list.append(dname)
    print(ex_list)
    n = 6  # 4个线程
    td_len = len(ex_list)/n  # 每一个td要处理的长度
    # 将ex_list平均分成n段
    h = 0
    new_list=[]
    for i in range(0,n):
        m=int(len(ex_list)/n)
        if i == n-1:
            obj=ex_list[h:]
        else:
            obj=ex_list[h:h+m]
        new_list.append(obj)
        h+=m
    for i in range(n):
        m = mp.Process(target=job, args=(new_list[i],))
        m.start()
        mps.append(m)

    for multip in mps:
        multip.join()
    print("test 2 all done")

if __name__=='__main__':
    mp1()