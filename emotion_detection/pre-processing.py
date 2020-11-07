import os
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

matplotlib.use('TKAgg')

def get_data():
    base_dir = '../dataset/KDEF_and_AKDEF/KDEF'
    datas = []

    for i, j, k in os.walk(base_dir):
        #print(k)
        for img in k:
            #print(img)
            name = img.split('.')[0]
            Number = name[:4]
            expression = name[4:6]
            angle = name[6:]
            #print(Number, expression, angle)
            switch_expression = {
                "AF": 0,
                "AN": 1,
                "DI": 2,
                "HA": 3,
                "NE": 4,
                "SA": 5,
                "SU": 6
            }
            switch_angle = {
                "FL": 0,
                "FR": 1,
                "HL": 2,
                "HR": 3,
                "S" : 4
            }
            #print(name)
            #print(expression, angle)
            data =  Number + '/' + img + ',' + \
                    str(switch_expression[expression]) + ',' + \
                    str(switch_angle[angle])
            datas.append(data)

if __name__ == "__main__":
    get_data()
    print("done")
    mapImg = Image.open('../dataset/KDEF_and_AKDEF/KDEFmap/AM05.JPG')
    #mapImg = Image.open('../dataset/KDEF_and_AKDEF/KDEFmap/AM05.JPG')
    # plt.figure(figsize=(12,12))
    # plt.imshow(mapImg)
    plt.plot([1,2,3],[5,7,4])

    plt.show()
    #mapImg.show()