import os
import random


class ImageRename():
    def __init__(self):
        self.path = r'D:\Python\PythonProject\Truck_identification\Random_enhanced_data\car'  # 图片所在文件夹

    def rename(self):
        filelist = os.listdir(self.path)
        random.shuffle(filelist)
        total_num = len(filelist)

        i = 1

        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), '小车' + format(str(i), '0>1s') + '.jpg')
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                i = i + 1
        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()
