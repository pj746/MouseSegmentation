import cv2

'''
The following is for converting the labeled RGB image into grey image, and change the color of background to black.
Before running this piece of code, run the `save_labeled-data.py` first. Then use `find -name '*.png' | sed -e 's/\.\///' | split`
in the terminal to creat the file of names.
'''

path = r'./data/'
#load the list of all names
name = open(path+r'training/xab','r').read().splitlines()

convert the RGB image into grey image
for i in range(imagNums):
    img = cv2.imread(path+r'Seg/'+name[i])
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for j in range(gray_img.shape[0]):
        for k in range(gray_img.shape[1]):
            if (gray_img[j,k] == 30):
                gray_img[j,k] *= 0

    cv2.imwrite(r'my-lab-labeled-data/SegGrey/'+name[i], gray_img)