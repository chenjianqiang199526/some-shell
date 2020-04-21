
import cv2
import matplotlib.pyplot as plt
import numpy as np


def BGR2RGB(img):
    b,g,r=cv2.split(img)
    im=cv2.merge([r,g,b])
    return im

def motion_blur(img,degree=12,angle=45):
    #img：输入的图像，默认是BGR形式的
    #获得运动模糊的掩码,degree越大运动模糊程度越大
    mask=cv2.getRotationMatrix2D((degree/2,degree/2),angle,1)
    motion_blur_kernel=np.diag(np.ones(degree))
    motion_blur_kernel=cv2.warpAffine(motion_blur_kernel,mask,(degree,degree))

    motion_blur_kernel=motion_blur_kernel/degree
    blur_img=cv2.filter2D(img,-1,motion_blur_kernel)
    cv2.normalize(blur_img,blur_img,0,255,cv2.NORM_MINMAX)
    blur_img=np.array(blur_img,np.uint8)
    return blur_img

def defocus_blur(img,r=3,sigma=1):
    #这里使用高斯低通滤波来做对焦模糊
    #sigma越大图像的模糊程度越大
    blur_img=cv2.GaussianBlur(img,(r,r),sigma)
    return blur_img

img=cv2.imread('timg.jpg')
img=cv2.resize(img,(200,300))
img_motion=motion_blur(img)
img_defocus=defocus_blur(img)
img=BGR2RGB(img)
img_motion=BGR2RGB(img_motion)
img_defocus=BGR2RGB(img_defocus)
plt.subplot(1,3,1)
plt.imshow(img)
plt.subplot(1,3,2)
plt.imshow(img_motion)
plt.subplot(1,3,3)
plt.imshow(img_defocus)
plt.show()