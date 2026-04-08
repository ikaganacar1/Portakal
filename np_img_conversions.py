import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.image as mpimg

def img_to_np(img_:str):
    return mpimg.imread(img_)


def np_to_img(arr_):
    
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')

    ax.imshow(arr_, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    arr = img_to_np("img.jpg")
    np_to_img(arr)
