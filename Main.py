import cv2
import numpy as np
import os
from PIL import Image 
import PIL 
knn=cv2.ml.KNearest_create()
def traindata():
    path='TrainingData'
    myList=os.listdir(path)
    images=[]
    classNo=[]
    #entering into directory and opening every image in sub directory
    #Then appending it to images and mapping its identity in classNo array
    for x in range(0,len(myList)):
        myPicList=os.listdir(path+"/"+str(x))
        for y in myPicList:
            curImg=cv2.imread(path+"/"+str(x)+"/"+y,cv2.IMREAD_GRAYSCALE)
            curImg=cv2.resize(curImg,(40,40))
            curImg=curImg.flatten()
            images.append(curImg)
            classNo.append(x)
    images=np.array(images,dtype=np.float32)
    classNo=np.array(classNo)

    #kNN train
    knn.train(images,cv2.ml.ROW_SAMPLE,classNo)

def test(boxes):
    boxes=np.array(boxes,dtype=np.float32)
    ret, result, neighbours, dist=knn.findNearest(boxes,k=3)
    return result

def prepocess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.blur(img,(3,3))
    img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,5)
    return img 
def biggestcontour(contours):
    biggest=np.array([])
    maxarea=0
    for i in contours:
        area=cv2.contourArea(i)
        if area>50:
            peri=cv2.arcLength(i,True)
            #gives number of corners
            approx=cv2.approxPolyDP(i,0.02*peri, True)
            if area>maxarea and len(approx)==4:
                biggest=approx
                maxarea=area
    return biggest,maxarea            

def findpoints(biggest):
    biggest=biggest.reshape((4,2))
    newpoint=np.zeros((4,1,2),dtype=np.int32)
    add=biggest.sum(1)
    #max of i+j will be bottom right corner, 
    #min of i+j will be top left corner or 0,0,
    #on subtracting i and j, max be get top right
    #on subtracting i and j, min be get bottom left
    newpoint[0]=biggest[np.argmin(add)]
    newpoint[3]=biggest[np.argmax(add)]
    diff=np.diff(biggest,axis=1)
    newpoint[1]=biggest[np.argmin(diff)]
    newpoint[2]=biggest[np.argmax(diff)]
    return newpoint
def splitboxes(img):
    img=cv2.bitwise_not(img)
    rows=np.vsplit(img,9)  
    boxes=[]
    for r in rows:
        cells=np.hsplit(r,9)
        for c in cells:
            c=cv2.resize(c,(40,40))
            c=cv2.blur(c,(5,5))
            c=c.flatten()
            boxes.append(c)
    return boxes        


img=cv2.imread('1.jpg')
img=cv2.resize(img,(450,450))

#preprocess the image
img=prepocess(img)
#find all outer contours
contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#find biggest contour
biggest,maxArea=biggestcontour(contours)
#rearrange the points
biggest=findpoints(biggest)
#two points found for perspective transformation
pts1=np.float32(biggest)
pts2=np.float32([[0,0],[450,0],[0,450],[450,450]])
#perspective transformed
matrix=cv2.getPerspectiveTransform(pts1,pts2)
img=cv2.warpPerspective(img,matrix,(450,450))
img2=img.copy()
#splitting image
boxes=splitboxes(img)


traindata()
characters=test(boxes)
for i in range(1,len(boxes)+1):
    if(i%9==0):
        print(characters[i-1],end="\n")
    else:
        print(characters[i-1],end=" ")

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows