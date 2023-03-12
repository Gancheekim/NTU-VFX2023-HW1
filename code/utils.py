import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

class MTB():
    '''Median Threshold Bitmap image alignment'''
    def __init__(self, stage):
        self.stage = stage # indicate how many stage of downscaling. eg: stage = 4, image pool scale: [1, 1/4, 1/16, 1/64]

    def rgb2gray(self, img):
        Y = 54/256*img[:,:,0] + 183/256*img[:,:,1] + 19/256*img[:,:,2]
        return Y
    
    def getBinaryThresholdMask(self, img):
        median = np.median(img)
        return median, (img >= median).astype(np.uint8)
        
    def getExclusionMap(self, img, median, threshold=10):
        return ~cv2.inRange(img, median-threshold, median+threshold)
    
    def getScaledDownImages(self, img):
        imgs = [img]
        h,w = img.shape
        for i in range(self.stage-1):
            h, w = h//2, w//2
            imgs.append(cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA))
        return imgs
    
    def shift(self, img, dx, dy):
        img = np.roll(img, dy, axis=0)
        img = np.roll(img, dx, axis=1)
        if dy>0:
            img[:dy, :] = 0
        elif dy<0:
            img[dy:, :] = 0
        if dx>0:
            img[:, :dx] = 0
        elif dx<0:
            img[:, dx:] = 0
        return img
    
    def getMinCostOffset(self, img1, img2):
        minCost = math.inf
        offset = (0,0)
        for dy in range(-1,2):
            for dx in range(-1,2):
                cost = np.sum(img1 != self.shift(img2,dx,dy))
                # print(dy, dx, cost)
                if cost < minCost:
                    minCost = cost
                    offset = (dy,dx)
        return offset
    
    def mtb(self, img1, img2):
        '''
        main function here
        fix img1 position, calculate the offset for img2 to be aligned with img1
        '''
        gray1 = self.rgb2gray(img1)
        gray2 = self.rgb2gray(img2)

        med1, mask1 = self.getBinaryThresholdMask(gray1)
        med2, mask2 = self.getBinaryThresholdMask(gray2)

        exmask1 = self.getExclusionMap(gray1, med1)
        exmask2 = self.getExclusionMap(gray2, med2)

        mask1_list = self.getScaledDownImages(mask1 & exmask1)
        mask2_list = self.getScaledDownImages(mask2 & exmask2)

        offsets = []
        for scale in range(self.stage):
            idx = self.stage-1-scale
            curr_offset = self.getMinCostOffset(mask1_list[idx], mask2_list[idx])
            offsets.append(curr_offset)
            
            for i in range(scale+1, self.stage):
                idx = self.stage-1-i
                mask2_list[idx] = self.shift(mask2_list[idx], curr_offset[1]*2*(i-scale), curr_offset[0]*2*(i-scale))

        offsets.reverse()
        total_offset = [0,0]
        for i in range(len(offsets)):
            total_offset[0] += offsets[i][0] * (2**i)
            total_offset[1] += offsets[i][1] * (2**i)
        print(f'(MTB) offsets are y:{total_offset[0]} x:{total_offset[1]}')
        return total_offset


# for debug
if __name__ == '__main__':
    mtb = MTB(5)