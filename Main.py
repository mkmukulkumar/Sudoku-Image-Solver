import cv2
import numpy as np
import os

knn=cv2.ml.KNearest_create()

def printing(arr):
    for i in range(9):
        for j in range(9):
            print(arr[i][j], end = " ")
        print()
 
def isSafe(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False
    for x in range(9):
        if grid[x][col] == num:
            return False
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True
 
def solveSuduko(grid, row, col):
    if (row == 9 - 1 and col == 9):
        return True
    if col == 9:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return solveSuduko(grid, row, col + 1)
    for num in range(1, 9 + 1, 1):
        if isSafe(grid, row, col, num):
            grid[row][col] = num
            if solveSuduko(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False
 


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

def ocr(boxes):
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
            contours,hierarchy=cv2.findContours(c,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(c,contours,-1,(255,255,255),10)
            c=c.flatten()
            boxes.append(c)
    return boxes        

# Driver Code
if __name__=="__main__":
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
    #training model
    traindata()
    #ocr
    characters=ocr(boxes)
    temp=[]
    for i in range(0,81):
        temp.append(int(characters[i]))
    grid=np.reshape(temp, (9,9))

    if (solveSuduko(grid, 0, 0)):
        printing(grid)
    else:
        print("no solution  exists ")



    