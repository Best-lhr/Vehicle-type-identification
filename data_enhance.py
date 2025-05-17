import cv2
import os
import numpy as np

new_filepath = r'D:\Python\PythonProject\Truck_identification\enhanced_data\car'

def read_path(file_pathname):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):

        img = cv2.imread(file_pathname+'/'+filename)
        ####change to gray
      #（下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图）
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #image_np=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #####save figure
        #cv2.imwrite(r'D:\Python\PythonProject\Truck_identification\data\SUV_new'+"/"+filename,image_np)

        # 水平镜像
        h_flip = cv2.flip(img, 1)
        cv2.imwrite(new_filepath + "/" + filename + '_h_flip.jpg', h_flip)

        # 垂直镜像
        v_flip = cv2.flip(img, 0)
        cv2.imwrite(new_filepath + "/" + filename + '_v_flip.jpg', v_flip)

        # 水平垂直镜像
        hv_flip = cv2.flip(img, -1)
        cv2.imwrite(new_filepath + "/" + filename + 'hv_flip.jpg', hv_flip)

        # 30度旋转
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
        dst_30 = cv2.warpAffine(img, M, (cols, rows))
        cv2.imwrite(new_filepath + "/" + filename + 'dst_30.jpg', dst_30)

        # 45度旋转
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        dst_45 = cv2.warpAffine(img, M, (cols, rows))
        cv2.imwrite(new_filepath + "/" + filename + 'dst_45.jpg', dst_45)

        # 60度旋转
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 60, 1)
        dst_60 = cv2.warpAffine(img, M, (cols, rows))
        cv2.imwrite(new_filepath + "/" + filename + 'dst_60.jpg', dst_60)

        # 90度旋转
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        dst_90 = cv2.warpAffine(img, M, (cols, rows))
        cv2.imwrite(new_filepath + "/" + filename + 'dst_90.jpg',
                    dst_90)


        # 缩放
        #height, width = img.shape[:2]
        #res = cv2.resize(img, (2 * width, 2 * height))
        #cv2.imwrite(new_filepath + "/" + filename +'res.jpg', res)

        #剪切
        height, width = img.shape[:2]
        cropped = img[int(height / 9):height, int(width / 9):width]
        cv2.imwrite(new_filepath + "/" + filename + 'cro.jpg', cropped)

        # 仿射变换
        # 对图像进行变换（三点得到一个变换矩阵）
        # 我们知道三点确定一个平面，我们也可以通过确定三个点的关系来得到转换矩阵
        # 然后再通过warpAffine来进行变换
        rows, cols = img.shape[:2]
        point1 = np.float32([[50, 50], [300, 50], [50, 200]])
        point2 = np.float32([[10, 100], [300, 50], [100, 250]])
        M = cv2.getAffineTransform(point1, point2)
        dst1 = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))
        cv2.imwrite(new_filepath + "/" + filename + 'dst1.jpg', dst1)


#读取的目录
read_path(r"D:\Python\PythonProject\Truck_identification\enhanced_data\car")

