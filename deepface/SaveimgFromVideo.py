import shutil
import time
import cv2
import os

def SaveImg(source, name , maximumPic, fps):
    cap = cv2.VideoCapture(source)
    
    count = 1
    if os.path.exists(name):
        shutil.rmtree(name)
    os.makedirs(name, exist_ok= True)
    ## 이미지 저장 폴더가 있다면 삭제 후 재생성
    print(name)
    
    if source == 0:
        while True:
            ret, frame = cap.read()
            # frame = cv2.flip(frame, -1)
            if not ret:
                print("종료")
                break
            if(int(cap.get(1)) % fps ==0):
                #-- 지정한 프레임마다 저장
                time.sleep(0.2)
                #-- 0.2초 간격
                print('Saved frame number :' + str(int(cap.get(1))))
                cv2.imwrite(name + "/" + name + "%d.jpg" % count, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                print("Saved frame%d.jpg" %count)
                count = count +1
                if(count == maximumPic):
                    print("Saved")
                    break
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, -1) #-- 상하좌우 대칭 변환
            # frame = cv2.flip(frame, 1) #-- 상하좌우 대칭 변환
            
            if not ret:
                print('종료')
                break
            if(int(cap.get(1)) % fps ==0):
                #-- 지정한 프레임마다 저장
                print('Saved frame number :' + str(int(cap.get(1))))
                cv2.imwrite(name + "/" + name + "%d.jpg" % count, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                print("Saved frame%d.jpg" %count)
                time.sleep(0.2)
                #-- 0.2초 가견
                count = count +1
                if(count == maximumPic):
                    print("Saved")
                    break
                #-- 지정한 maximumPic만큼 사진 저장
    cap.release() 
    #-- 자원반납