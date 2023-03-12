import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import utils


def getDatasetImageName(dataset):
    if dataset == "memorial":
        prefix_name = "./../data/memorial/memorial00"
        sufix_name = ".png"
        imgs_name = [prefix_name+str(idx)+sufix_name for idx in range(61,77)]

    if dataset == "debug":
        prefix_name = "./../data/debug/p"
        sufix_name = ".jpg"
        imgs_name = [prefix_name+str(idx)+sufix_name for idx in range(1,3)]
    
    return imgs_name

def main(args):
    imgs_name = getDatasetImageName(args.dataset)
    imgs = [np.asarray(Image.open(img_name).convert('RGB')) for img_name in imgs_name] # read image as numpy array

    img1 = imgs[0]
    img2 = imgs[1]
    mtb = utils.MTB(5)

    offsets = mtb.mtb(img2, img1)
    img3 = mtb.shift(img2, offsets[1], offsets[0])

    
    plt.subplot(1,3,1)
    plt.imshow(img2)
    plt.title('fix image')
    plt.subplot(1,3,2)
    plt.imshow(img1)
    plt.title('image to be aligned')
    plt.subplot(1,3,3)
    plt.imshow(img3)
    plt.title('alignment result')
    plt.show()
    return

if __name__ == '__main__':
    parser = ArgumentParser()
    '''add argument here'''
    parser.add_argument("--dataset", type=str, default="memorial", choices=["memorial", "debug"]) # currently debug mode is for testing MTB
    args = parser.parse_args()
    main(args)


