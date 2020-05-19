import cv2
import numpy as np
import re

'''
The following is to crop the image to specific size.
originalsize:424*512
goal: 424*480
'''

path = r'./data/label/'

#load the list of all names
name_list = open(path+r'xaa','r').read().splitlines()
for i in range(len(name_list)):
    img = cv2.imread(path+name_list[i])
    crop = img[0:423,15:495]
    img_num = re.findall(r'\d+',name_list[i]) #\d+ 正则表达式，匹配多个数字
    new_name = 'img'+str(img_num[0])+r'.png'
    cv2.imwrite(path+new_name, crop)

print(cv2.imread(path+name_list[0]).shape) #(423,480,3)