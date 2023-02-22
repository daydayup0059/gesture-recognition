#手势识别代码，并绘制手势的每个点，用多边形连接
import cv2
import numpy as np

#选取人体手指颜色范围
highHSV=np.array([ 15 ,255,255])
lowHSV=np.array([ 0 ,50 ,50])

def img_hand(img):
    # if img.shape[0]>1000 and img.shape[1]>1000:
    #     img=cv2.resize(img,None,fx=0.2,fy=0.2)
    faimg=np.copy(img)  #复制图像，保留副本，在原图上操作，副本用于后面绘制轮廓在图像向上
    #BGR和HSV颜色空间概念  https://blog.csdn.net/liu_taiting/article/details/107154484
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #ksize:⾼斯卷积核的⼤⼩，注意 ： 卷积核的宽度和⾼度都应为奇数，且可以不同
    cv2.GaussianBlur(img,[5,5],0)  #高斯滤波 去除噪音
    #HSV色彩空间表和cv2.inRange()的用法https://blog.csdn.net/ayfen/article/details/120096179
    #cv2.inRange函数设阈值，去除背景部分
    img=cv2.inRange(img,lowHSV,highHSV)  #获得HSV掩膜
    #开运算 是先腐蚀后膨胀，其作⽤是：分离物体，消除⼩区域。特点：消除噪点，去除⼩的⼲扰块，⽽不影响原来的图像。
    #cv2.MORPH_CLOSE闭运算是先膨胀后腐蚀，作⽤是消除/“闭合”物体⾥⾯的孔洞，特点：可以填充闭合区域。
    kernel=np.ones([3,3],dtype=np.uint8)
    img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,iterations=1)
    # MORPH_ERODE:腐蚀(黑变多),与腐蚀函数cv2.erode效果相同 MORPH_DILATE:膨胀(白变多),
    #kernel内核改变
    kernel=np.ones([5,5],dtype=np.uint8)
    newimg=cv2.morphologyEx(img,cv2.MORPH_DILATE,kernel,iterations=1)
    #cv2.RETR_TREE：返回所有的轮廓，建⽴⼀个完整的组织结构的轮廓。
    #cv2.CHAIN_APPROX_SIMPLE：压缩⽔平⽅向，垂直⽅向，对⻆线⽅向的元素，只保留该⽅向的终点坐标，例如⼀个矩形轮廓只需4个点来保存轮廓信息。
    # 注意opencv4版本，cv2.findContours 只返回2个值
    #contours返回的⼆值图像，num检测出的轮廓，所有轮廓的列表结构，每个轮廓是⽬标对象边界点的坐标的数组
    contours,num=cv2.findContours(newimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #每张图片检测出很多轮廓，用for遍历每个轮廓进行操作
    for contour in contours:
        area=cv2.contourArea(contour)
        lenth=cv2.arcLength(contour,True)
        #if area > 200 and lenth > 100:
        if area>20000 and lenth>1000:
            ## cv2.arcLength(cnt, True) 计算轮廓的周长 参数2 表示轮廓是否封闭
            #在这里epsilon作为一个参数，这个需要调一下。0.02就是调出来的
            epsilon = 0.02*cv2.arcLength(contour,True)
            #approx返回的点集
            approx = cv2.approxPolyDP(contour,epsilon,True)
            #len(approx)和approx相同行，2列
            approx=approx.reshape(len(approx),2)
            #创建numpy数组，类型dtype=np.int32
            approx=np.array(approx,dtype=np.int32)
            #cv2.polylines()方法用于在图像副本faimg 绘制任何图像上的多边形。
            cv2.polylines(faimg, [approx], True, [255, 125, 100], 4, 16)
    return faimg
#video = cv2.VideoCapture('../videos/1.mp4') # 输入0就是表示读取0号摄像头 #https://blog.csdn.net/m0_51545690/article/details/123883328
video=cv2.VideoCapture(0)
while video.isOpened():
    #通过read方法可以读取到每一帧的图片，这个函数返回2个值，第一个是一个布尔值，成功就返回True，第二个值就是这一帧图像。
    res,img=video.read()
    if res== True:
        newimg=img_hand(img)
        cv2.imshow('frams',newimg)
    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()
video.release()
