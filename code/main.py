import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from mtb import MTB

def getDatasetImageName(dataset):
    if dataset == "memorial":
        prefix_name = "./../data/memorial/memorial00"
        sufix_name = ".png"
        imgs_name = [prefix_name+str(idx)+sufix_name for idx in range(61,77)]

    if dataset == "debug":
        prefix_name = "./../data/debug/p"
        sufix_name = ".jpg"
        imgs_name = [prefix_name+str(idx)+sufix_name for idx in range(1,5)]
    
    return imgs_name

def main(args):
    imgs_name = getDatasetImageName(args.dataset)
    imgs = [np.asarray(Image.open(img_name).convert('RGB')) for img_name in imgs_name] # read image as numpy array

    mtb = MTB(8)
    shift_imgs = mtb.run(imgs, True) # list of numpy array, which are the aligned images

    # TODO: finish HDR algorithm

        
    return

if __name__ == '__main__':
    parser = ArgumentParser()
    '''add argument here'''
    parser.add_argument("--dataset", type=str, default="debug", choices=["memorial", "debug"]) # currently debug mode is for testing MTB
    args = parser.parse_args()
    main(args)


