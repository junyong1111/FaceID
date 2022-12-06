import random
import tensorflow as tf
import numpy as np
import os
import cv2
import glob
from PIL import Image
from PIL import ImageEnhance
import PIL.ImageOps    
import copy
from random import randint


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

 
def SaltPepper(img): ## 소금후추 노이즈
    # Getting the dimensions of the image
    if img.ndim > 2:  # color
        height, width, _ = img.shape
    else:  # gray scale
        height, width = img.shape
 
    result = copy.deepcopy(img)
 
    # Randomly pick some pixels in the image
    # Pick a random number between height*width/80 and height*width/10
    number_of_pixels = randint(int(height * width / 100), int(height * width / 10))
 
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = randint(0, height - 1)
 
        # Pick a random x coordinate
        x_coord = randint(0, width - 1)
 
        if result.ndim > 2:
            result[y_coord][x_coord] = [randint(0, 255), randint(0, 255), randint(0, 255)]
        else:
            # Color that pixel to white
            result[y_coord][x_coord] = 255
 
    # Randomly pick some pixels in image
    # Pick a random number between height*width/80 and height*width/10
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = randint(0, height - 1)
 
        # Pick a random x coordinate
        x_coord = randint(0, width - 1)
 
        if result.ndim > 2:
            result[y_coord][x_coord] = [randint(0, 255), randint(0, 255), randint(0, 255)]
        else:
            # Color that pixel to white
            result[y_coord][x_coord] = 0
 
    return result




def ImageAgu(num_augmented_images, file_path, augment_cnt, Username):

    file_names = os.listdir(file_path)
    print(file_names)
    total_origin_image_num = len(file_names)
    for i in range(1, num_augmented_images):
        change_picture_index = random.randrange(1, total_origin_image_num-1)
        print(change_picture_index)
        print(file_names[change_picture_index])
        file_name = file_names[change_picture_index]
        
        origin_image_path = file_path + file_name
        print(origin_image_path)
        image = Image.open(origin_image_path)
        random_augment = random.randrange(1,7)
        
        
        
        if(random_augment == 1):
            #이미지 좌우 반전
            print("invert")
            inverted_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            inverted_image.save(file_path  + Username + str(augment_cnt) + '.jpg')
            
        elif(random_augment == 2):
            #이미지 기울이기
            print("rotate")
            rotated_image = image.rotate(random.randrange(-20, 20))
            rotated_image.save(file_path + Username + str(augment_cnt) + '.jpg')
            
        elif(random_augment == 3):
            #노이즈 추가하기
            img = cv2.imread(origin_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("noise")
            row,col,ch= img.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy_array = img + gauss
            noisy_image = Image.fromarray(np.uint8(noisy_array)).convert()
            noisy_image.save(file_path  + Username +str(augment_cnt) + '.jpg')
       
        elif(random_augment ==4):
            enhancerImage = ImageEnhance.Brightness(image)
            brightImg = enhancerImage.enhance(random.uniform(0.8, 1.8))
            brightImg.save(file_path  + Username +str(augment_cnt) + '.jpg')
            print("Bright")
            
        elif(random_augment ==5):
            enhancerContrastImage = ImageEnhance.Contrast(image)
            contrastImg = enhancerContrastImage.enhance(random.uniform(0.8, 1.8))
            contrastImg.save(file_path  + Username +str(augment_cnt) + '.jpg')
            print("Contrast")
            
        elif(random_augment ==6):
            img = cv2.imread(origin_image_path) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = SaltPepper(img)
            result = Image.fromarray(np.uint8(result)).convert()
            result.save(file_path  + Username + str(augment_cnt) + '.jpg')
        augment_cnt += 1