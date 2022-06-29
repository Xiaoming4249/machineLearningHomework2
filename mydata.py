from PIL import Image
from knn import IMG_PATH, plotImg
import numpy as np

def readPng():
    imgs = []
    for i in range(20):
        img = Image.open(IMG_PATH + f'myhand\\{i}.png')
        img = img.convert('L')
        cols, rows = img.size

        value = [[0]*cols for a in range(rows)]
        one = []

        for x in range(rows):
            for y in range(0, cols):
                imgArray = np.array(img)
                value[x][y] = imgArray[x, y]
                one.append(imgArray[x, y])
        imgs.append(one)
    labels = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,]
    return np.array(imgs), np.array(labels)

# imgs = readPng()
# plotImg(imgs)
