import os
import numpy as np
import cv2

# def load_data(data_directory):
    
#     images = []
#     for image_reference in os.listdir(data_directory):
#         image_ = os.path.join(data_directory,image_reference)
#         img = cv2.imread(image_, 0)/255        
#         images.append(img.reshape(100,100,1))   
#     return  np.array(images, np.float32).reshape(-1,100,100,1)#[-1, 101, 101, nc]

def load_data(data_directory):
    
    w, h = 50, 50 # height and width of image
    images = []
    for image_reference in os.listdir(data_directory):
        image_ = os.path.join(data_directory,image_reference)
        img = cv2.imread(image_, 0)
        img = cv2.resize(img, (w,h))
        img = img.reshape(w,h,1)
        img = cv2.normalize(img, None, -1, 1, cv2.NORM_MINMAX)
        images.append(img)   
    return  np.array(images, np.float32).reshape(-1,w,h,1)#[-1, 101, 101, nc]
