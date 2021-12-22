import matplotlib as plt
import numpy as np
import cv2
import os
from skimage import data, filters

#get a list of files in the folder with pics
from os import listdir
from os.path import isfile, join
folder_path = os.path.dirname(__file__)+'/Smpl_Im'
processed_path = os.path.dirname(__file__)+'/Thresholded'
file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

# finction creates a circular mask based on the dimentions, radius and the desired loaction
#I modified it from here:
#https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    #Compute square of the radius to avoid computing sqrt on every step
    radius_sq = radius**2

    Y, X = np.ogrid[:h, :w]
    dist_from_center_sq = (X - center[0])**2 + (Y-center[1])**2

    mask = dist_from_center_sq <= radius_sq
    return mask

def define_circular_ROI(image):
    
    H, W = image.shape[:2]
    Y = H/2
    X = W/2

    radius= 0.4*W
    
    return Y,X,radius


#Everything below T1 is background, everything above T2 is colony
T1 = 60
T2 = 100

for file in file_names:
    #Read the image images
    img = cv2.imread(folder_path+'/'+file)

    #Define the center and the radius of the ROI
    X_cent, Y_cent, Rad = define_circular_ROI(img)

    #Set everything outside of the ROI to 0
    H,W = img.shape[:2]
    mask = create_circular_mask(H, W, (X_cent, Y_cent), Rad)
    img[~mask] = 0
    #img[img<60] = 0
    #img = cv2.GaussianBlur(img, (7,7),2)

    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)
    kp_filtered = [point for point in kp if point.size<50]
    img=cv2.drawKeypoints(gray, kp_filtered, np.array([]), (0,255,0), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    #Write the final image
    cv2.imwrite(processed_path+'/'+file, img)