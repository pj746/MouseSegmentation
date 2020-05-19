import matplotlib.pyplot as plt
import pickle

MAT = pickle.load(open('data.pkl', 'rb'))

imagemasks, imagenames = MAT['imagemasks'], MAT['imagenames']
plt.figure()
i=2
filename = imagenames[i]
pathtoimage = 'rgb/'

plt.subplot(1,2,1)
plt.imshow(plt.imread(pathtoimage + filename))

plt.subplot(1,2,2)
plt.imshow(imagemasks[i])
plt.axis('scaled')