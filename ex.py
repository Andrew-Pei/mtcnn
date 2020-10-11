from src import detect_faces, show_bboxes
from PIL import Image
import os, sys, stat
import glob
#import tqdm

for dname in glob.glob("/data1/pbw_deepfake/test/0/**/*_jpg",recursive=True):
    print(dname[27:-4])
    os.mkdir(os.path.join("/data1/pbw_deepfake/test/extract_face_jpg/0",dname[27:-4]),777)
    os.chmod(os.path.join("/data1/pbw_deepfake/test/extract_face_jpg/0",dname[27:-4]),stat.S_IRWXU+stat.S_IRWXG+stat.S_IRWXO)
    for fname in os.listdir(dname):
        #print(fname)
        #pass
        img = Image.open(os.path.join(dname,fname))
        bounding_boxes, landmarks = detect_faces(img)
        #show_bboxes(img, bounding_boxes, landmarks)
        if len(bounding_boxes):
            if bounding_boxes.size!=0:
                bb=bounding_boxes[0]
                #print(bb)
                w1=bb[0]
                h1=bb[1]
                w2=bb[2]
                h2=bb[3]
                aa=[]        
                w3=w1-0.1*(w2-w1)
                h3=h1-0.2*(h2-h1)
                w4=w2+0.3*(w2-w1)
                h4=h2+0.2*(h2-h1)
                if (h4-h3)>(w4-w3):
                    w3-=0.5*((h4-h3)-(w4-w3))
                    w4+=0.5*((h4-h3)-(w4-w3))
                if (h4-h3)>(w4-w3):
                    w3-=0.5*((h4-h3)-(w4-w3))
                    w4+=0.5*((h4-h3)-(w4-w3))
                if (h4-h3)>(w4-w3):
                    w3-=0.5*((h4-h3)-(w4-w3))
                    w4+=0.5*((h4-h3)-(w4-w3))
                if w3<0:
                    w3=0        
                if h3<0:
                    h3=0        
                if w4>img.size[0]:
                    w4=img.size[0]
                if h4>img.size[1]:
                    h4=img.size[1]
                aa.append(w3)
                aa.append(h3)
                aa.append(w4)
                aa.append(h4)
                #print(w3)
                #print(h3)
                #print(w4-w3)
                #print(h4-h3)
                
                face=img.crop(aa[:4])
                #face.show()
                #print(face.size)
                face.save(os.path.join("/data1/pbw_deepfake/test/extract_face_jpg/0",dname[27:-4],fname))#TODO
                print("saved")
            else:
                print("no face found!")
        else:
            print("bounding_boxes is empty!")