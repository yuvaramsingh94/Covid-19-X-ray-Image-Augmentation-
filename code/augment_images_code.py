
#Create augmented images from original images

# python script command for execution: python augment_images.py -i ./original -o ./augmented

import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse


#parameters for augmentation for powerlax,average mask and sharpening
gamma = np.array([0.4, 0.45,0.5, 0.55, 0.6])
power_c = 10
avg_kernel = [3,5]
kernel = np.ones((3,3),np.float32)/9
lam = 3


def parse_arguments(): 
  parser = argparse.ArgumentParser()
 
  parser.add_argument("-i", "--input_path", help="input directory path",
                    action="store",dest='input_directory_path') 	
  parser.add_argument("-o", "--output_path", help="output  directory path",
                    action="store",dest='output_directory_path') 
  args = parser.parse_args()
  return args

args=parse_arguments()


#power law transformation
def power_law(img, img_name):  
    for k in range(len(gamma)):             
        pow_law_img = np.array(power_c*np.power(img,gamma[k]))#,'uint8')
        #save image
        cv2.imwrite(args.output_directory_path+"./"+"aug_" + img_name[:-4]+'_PL_'+str(k)+'.jpg',pow_law_img)  

#averaging
def avg_mask(img, img_name): 
    for i in range(len(avg_kernel)):
        kernel = np.ones((avg_kernel[i],avg_kernel[i]),np.float32)/(avg_kernel[i]*avg_kernel[i])
        avg_img = cv2.filter2D(img,-1,kernel)
        #save image
        cv2.imwrite(args.output_directory_path +"./"+"aug_" + img_name[:-4]+'_avg_'+str(i)+'.jpg',avg_img)


#sharpening
def sharpen(img, high_freq, img_name, sharp_id):        
    sharp_img = img+(lam*high_freq)
    #save image
    cv2.imwrite(args.output_directory_path+"./"+"aug_" + img_name[:-4]+'_sharp_'+str(sharp_id)+'.jpg',sharp_img) 


#function to augment and write images to output directory
def aug_and_write_images(img, img_name):
  
    #power_law
    power_law(img, img_name)

    #average mask
    avg_mask(img, img_name)

    #smooth filter for sharpening
    smooth_img = cv2.filter2D(img,-1,kernel)
    high_freq1 = img-smooth_img
    sharpen(img, high_freq1, img_name, 1)

    #canny edge filter for sharpening
    canny_edge = cv2.Canny(img,50,225)
    high_freq4 = np.zeros(img.shape,'uint8')
    #canny edge sharpening
    high_freq4[:,:,0] = canny_edge
    high_freq4[:,:,1] = canny_edge
    high_freq4[:,:,2] = canny_edge
    sharpen(img, high_freq4, img_name, 4)


def augment_images():
  #get list of path of images
  img_list = os.listdir(args.input_directory_path+"/")
  #parse over path
  for i in tqdm(range(len(img_list))): 
      img_name = img_list[i] 
      img_path = args.input_directory_path+ "./"+ img_list[i] 
      #read image from image path
      img = cv2.imread(img_path)
      if(img is not None):
        aug_and_write_images(img,img_name)


def main():
  augment_images()

if __name__ == '__main__':
    main()