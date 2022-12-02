import random
import numpy as np
import os
import cv2
import glob
from PIL import Image
from PIL import ImageEnhance
import PIL.ImageOps    

#다음 변수를 수정하여 새로 만들 이미지 갯수를 정합니다.

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
        random_augment = random.randrange(1,6)
        
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
            brightImg = enhancerImage.enhance(random.uniform(0.8, 2))
            brightImg.save(file_path  + Username +str(augment_cnt) + '.jpg')
            print("Bright")
        elif(random_augment ==5):
            enhancerContrastImage = ImageEnhance.Contrast(image)
            contrastImg = enhancerContrastImage.enhance(random.uniform(0.8, 2))
            contrastImg.save(file_path  + Username +str(augment_cnt) + '.jpg')
            print("Contrast")
            
        augment_cnt += 1