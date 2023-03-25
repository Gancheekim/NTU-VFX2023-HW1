import cv2
import numpy as np
import matplotlib.pyplot as plt

class tone_mapping_Reinhard():
    def __init__(self, key=0.18, Lwhite=0., sigma=0.000001, gamma=1/2, phi=8.0, a=1.0, base_sigma=0.5, eps=0.001):
        # global tone mapping parameter:
        self.key = key
        self.Lwhite = Lwhite
        self.sigma = sigma
        self.gamma = gamma
        # local tone mapping paramter:
        self.phi = phi
        self.a = a
        self.base_sigma = base_sigma
        self.eps = eps

    def rgb2gray(self, img):
        # Lw = 54/256*img[:,:,0] + 183/256*img[:,:,1] + 19/256*img[:,:,2]
        # Lw = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
        Lw = 0.27*img[:,:,0] + 0.67*img[:,:,1] + 0.06*img[:,:,2] # refer to original paper
        return Lw

    def normalize(self, irradiance_img):
        min_val = irradiance_img.min()
        max_val = irradiance_img.max()
        return (irradiance_img - min_val) / (max_val - min_val)

    def cal_Ld(self, Lw):
        Lw_bar = np.exp(np.mean(np.log(self.sigma + Lw)))
        Lm = self.key/Lw_bar * Lw
        if self.Lwhite == 0:
            self.Lwhite = Lm.max() 
        Ld = Lm * (1 + (Lm/(self.Lwhite**2)) ) / (1+Lm)
        return Ld, Lm
    
    def global_ops(self, irradiance_img):
        H,W,C = irradiance_img.shape
        irradiance_img = irradiance_img**self.gamma # gamma correction        
        Lw_unnormalized = self.rgb2gray(irradiance_img)
        Lw = self.normalize(Lw_unnormalized)
        
        Ld, Lm = self.cal_Ld(Lw)
        hdr_img = np.zeros((H,W,C))
        for c in range(C):
            hdr_img[:,:,c] =  Ld * irradiance_img[:,:,c] / Lw_unnormalized
        hdr_img = np.clip(hdr_img, 0.0, 1.0)
        return hdr_img, Lm
    
    def cal_variance(self, Ls_blur, Ls_plus1_blur, s):
        vs = (Ls_blur - Ls_plus1_blur) / (self.a*2**self.phi * self.key / (s**2) + Ls_blur)
        return abs(vs)
    
    def local_ops(self, Lm, irradiance_img):
        Ld = np.zeros_like(Lm)
        H,W = Lm.shape

        Ls_blur_list = []
        for i, k in enumerate(range(3,14,2)):
            Ls_blur = cv2.GaussianBlur(Lm, (k, k), self.base_sigma*(1.6**i))
            Ls_blur_list.append(Ls_blur)

        for x in range(H):
            for y in range(W):
                for i in range(len(Ls_blur_list)-1):
                    vs = self.cal_variance(Ls_blur_list[i][x,y], Ls_blur_list[i+1][x,y], i*2+3)
                    if vs > self.eps or i == len(Ls_blur_list)-2:
                        smax = i
                        break
                Ld[x,y] = Lm[x,y] / (1 + Ls_blur_list[smax][x,y])
        
        Lw = self.rgb2gray(irradiance_img)
        hdr_img = np.zeros_like(irradiance_img)
        for c in range(3):
            hdr_img[:,:,c] =  Ld * irradiance_img[:,:,c] / Lw
        hdr_img = np.clip(hdr_img, 0.0, 1.0)
        return np.nan_to_num(hdr_img)
    
    def run(self, irradiance_map):
        '''
        main function here
        - input:
        :-- irradiance_map: numpy array of the real number (float-type data) of the irradiance map
        - output:
        :-- global_op_result: numpy array of global tone mapping operation result
        :-- local_op_result: numpy array of local tone mapping operation result
        '''
        global_op_result, Lm = toneMapping.global_ops(irradiance_map)
        local_op_result = toneMapping.local_ops(Lm, global_op_result)
        return global_op_result, local_op_result
                


if __name__ == "__main__":
    #hdr_path = "./../data/debug/sample_hdr2.hdr"
    hdr_path = "./../data/output/hdr_image.hdr"
    # IMREAD_ANYDEPTH is needed because even though the data is stored in 8-bit channels
    # when it's read into memory it's represented at a higher bit depth
    img1 = cv2.imread(hdr_path, flags=cv2.IMREAD_ANYDEPTH)
    img = np.empty_like(img1)
    
    # bgr to rgb
    img[:,:,0] = img1[:,:,2] 
    img[:,:,1] = img1[:,:,1]
    img[:,:,2] = img1[:,:,0]

    # Using our own Debevec method to generate hdr, the channels are already in RGB
    img = img1

    toneMapping = tone_mapping_Reinhard(key=0.5)
    global_tone, local_tone = toneMapping.run(img)

    from PIL import Image
    tmp = Image.fromarray(np.uint8(global_tone*255))
    tmp.save('./../data/output/global_op.png')
    tmp = Image.fromarray(np.uint8(local_tone*255))
    tmp.save('./../data/output/local_op.png')

    for c in range(3):
        img[:,:,c] = toneMapping.normalize(img[:,:,c])

    plt.subplot(1,4,1)
    plt.imshow(img*255)
    plt.subplot(1,4,2)
    plt.imshow(global_tone)
    plt.subplot(1,4,3)
    plt.imshow(local_tone)

    tonemapReinhard = cv2.createTonemapReinhard(intensity=2.0, light_adapt=0.0, color_adapt=0.0)
    ldrReinhard = tonemapReinhard.process(img)
    plt.subplot(1,4,4)
    plt.imshow(ldrReinhard)

    plt.show()



