import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from mtb import MTB
from hdr_generator import debevec, get_exposure, resize_imgs
from tone_mapping import tone_mapping_Reinhard

def getDatasetImageName(dataset):
    if dataset == "memorial":
        prefix_name = "./../data/memorial/memorial00"
        sufix_name = ".png"
        imgs_name = [prefix_name+str(idx)+sufix_name for idx in range(61,77)]

    if dataset == "debug":
        prefix_name = "./../data/debug/p"
        sufix_name = ".jpg"
        imgs_name = [prefix_name+str(idx).zfill(2)+sufix_name for idx in range(1,5)]

    if dataset == "ntu_sample1":
        prefix_name = "./../data/ntu_sample1/_C4A15"
        sufix_name = ".JPG"
        imgs_name = [prefix_name+str(idx).zfill(2)+sufix_name for idx in range(66,77)]

    if dataset == "ntu_sample2":
        prefix_name = "./../data/ntu_sample2/_C4A15"
        sufix_name = ".JPG"
        imgs_name = [prefix_name+str(idx).zfill(2)+sufix_name for idx in range(32,41)]

    if dataset == "ntu_sample3":
        prefix_name = "./../data/ntu_sample3/_C4A15"
        sufix_name = ".JPG"
        imgs_name = [prefix_name+str(idx).zfill(2)+sufix_name for idx in range(22, 29)]

    if dataset == "ntu_sample4":
        prefix_name = "./../data/ntu_sample4/_C4A15"
        sufix_name = ".JPG"
        imgs_name = [prefix_name+str(idx).zfill(2)+sufix_name for idx in range(8, 16)]
    
    return imgs_name

def main(args):
    imgs_name = getDatasetImageName(args.dataset)
    imgs = [np.asarray(Image.open(img_name).convert('RGB')) for img_name in imgs_name] # read image as numpy array
    imgs = resize_imgs(imgs, 768)

    # image alignment
    if not args.disable_mtb:
        mtb = MTB(6)
        print('running MTB...')
        shift_imgs = mtb.run(imgs, False) # list of numpy array, which are the aligned images
    else:
        shift_imgs = imgs

    # HDR algorithm
    print('reading exposure of imgs...')
    exps = get_exposure(args.dataset_info)   # list of exposure times for all images
    P    = len(exps)                         # number of images
    print('running debevec...')
    hdr_img = debevec(shift_imgs, exps, P, args.N)
    output_dir = "./../data/output/"
    cv2.imwrite(output_dir + "hdr_image.hdr", hdr_img)

    # Tone mapping from hdr image
    hdr_path = "./../data/output/hdr_image.hdr"
    img = cv2.imread(hdr_path, flags=cv2.IMREAD_ANYDEPTH)
    toneMapping = tone_mapping_Reinhard(key=args.tm_key)
    global_tone, local_tone = toneMapping.run(img)
    # save image
    tmp = Image.fromarray(np.uint8(global_tone*255))
    tmp.save('./../data/output/global_op.png')
    tmp = Image.fromarray(np.uint8(local_tone*255))
    tmp.save('./../data/output/local_op.png')

        
    return

if __name__ == '__main__':
    parser = ArgumentParser()
    '''add argument here'''
    parser.add_argument("--dataset", type=str, default="ntu_sample1", choices=["memorial", "debug", "ntu_sample1", "ntu_sample2", "ntu_sample3", "ntu_sample4"]) # currently debug mode is for testing MTB
    parser.add_argument("--dataset_info", type=str, default="./../data/memorial/image_list.txt", help="Path to text file with info about dataset exposure times.")
    parser.add_argument("--N", type=int, default=40, help="Number of sample points per image, per channel.")
    parser.add_argument("--disable_mtb", help="flags to disable MTB image alignment", action="store_true")
    parser.add_argument("--tm_key", type=float, default=0.5, help="reinhard photographic tone mapping's key parameter",)
    args = parser.parse_args()
    args.dataset_info = f"./../data/{args.dataset}/image_list.txt"
    main(args)


