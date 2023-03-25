import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser
from mtb import MTB

# Paths
output_dir = "./../data/output/"

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



#============================
# For HDR Generation
#============================
def get_exposure(txt_path):
    exposure_time_list = []
    with open(txt_path) as fin:

        for line in fin:     
            # 1. Skip comments or numbers (this is number of images)
            if (line.startswith("#") or (line.split()[0].isnumeric())):
                continue

            # 2. Extract info
            (fname, inv_shutter_speed, f_per_stop, db_gain, nd_filters) = line.split()

            # Compute exposure time, and list them
            exposure_time_list.append(1/float(inv_shutter_speed))
        fin.close()

        # print(f"Exposure time list: {exposure_time_list}")
        return exposure_time_list

def uniform_img_sampler(imgs_list, P, H, W, N, num_channels=0):
    ''' Samples image at specific channel
    '''
    sample_row_num = np.linspace(0, H-1, num=N, dtype=int)   # which rows to sample from the original image (i)
    sample_col_num = np.linspace(0, W-1, num=N, dtype=int)   # which columns to sample from the original image (j) 
    Z_samples = []

    # Sampling
    for channel_n in range(num_channels):
        Z_sampled = np.zeros((P, N, N))
        for img_idx, img in enumerate(imgs_list):
            for Z_i, i in enumerate(sample_row_num):
                for Z_j, j in enumerate(sample_col_num):
                    Z_sampled[img_idx][Z_i][Z_j] = img[i][j][channel_n] 
        Z_samples.append(Z_sampled)

    #print(f"Sample x-coords: {sample_col_num}\nSample y-coords: {sample_row_num}")
    return Z_samples

def get_response_func(Z_samples, B, Lambda, w):
    '''
    Solve Ax=b linear problem. Reference: MatLab code from lec03.pdf of lecture.
    Z: the pixel values of pixel location number i in image p; shape = P x N*N
    B: the log base 2 of exposure time
    Lambda: Lambda
    w: weighting
    '''
    n = 256
    g_funcs = []
    lsq_errors = []
    for Z in Z_samples:
        # Make matrix A
        Z = Z.reshape(Z.shape[0], Z.shape[1] * Z.shape[2]) # shape: P x N*N
        Z_num_rows = np.size(Z, 0)
        Z_num_cols = np.size(Z, 1)
        A = np.zeros((Z_num_rows * Z_num_cols + n + 1, n + Z_num_cols), dtype=np.float32)
        
        # Make matric b
        A_num_rows = np.size(A, 0)
        b = np.zeros((A_num_rows, 1), dtype=np.float32)

        
        # Include the data-fitting equations
        k = 0
        for i in range(Z_num_rows):
            for j in range(Z_num_cols):
                z = int(Z[i][j])
                wij = w[z]
                A[k][z] = wij
                A[k][n+j] = -wij
                b[k] = wij * B[i]
                k += 1
        # Fix the curve by setting the middle value to 0
        A[k][128] = 1
        k += 1

        # Include the smoothness equations
        for i in range(n-1):
            A[k][i] = Lambda * w[i+1]
            A[k][i+1] = -2 * Lambda * w[i+1]
            A[k][i+2] = Lambda * w[i+1]
            k += 1
        # Solve the system using SVD
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        g = x[:n]
        lE = x[n:]

        # Record the results
        g_funcs.append(g)
        lsq_errors.append(lE)

    return g_funcs, lsq_errors


def get_hdr(imgs_list, g_funcs, ln_exp_times, weight, W, H, C, P):
    # Separate the images by channel
    imgs_R = [img[:,:,0] for img in imgs_list]  # shape: P x H x W  red channel
    imgs_G = [img[:,:,1] for img in imgs_list]  # shape: P x H x W  green channel
    imgs_B = [img[:,:,2] for img in imgs_list]  # shape: P x H x W  blue channel
    imgs_RGB = [imgs_R, imgs_G, imgs_B]

    # Generate HDR image
    hdr_img = np.zeros(shape=(H, W, C), dtype=np.float32)   # initialize array
    for c, img_list in enumerate(imgs_RGB):
        Z = [img.flatten().tolist() for img in img_list]   # flatten length = H*W
        g = g_funcs[c]
        
        # Get ln(E)
        num_pixels = H*W
        ln_E      = np.zeros(num_pixels)
        
        for i in range(num_pixels):
            sum_numerator = 0
            sum_denominator = 0          # sum from image 1 to P, of weight(Zij)
            for j in range(P):
                Zij    = Z[j][i]         # value at pixel i, of image j 
                w_Zij  = weight[Zij]     # weight for pixel i, of image j        
                g_Zij  = g[Zij]          # response for pixel i, of image j
                ln_t_j = ln_exp_times[j] # log_2 of exposure time, of image j
                
                # Iterate through all images, and do summation
                sum_numerator   += (w_Zij * (g_Zij - ln_t_j))
                sum_denominator += w_Zij
                # print(f"sum_numerator: {sum_numerator}")
                # print(f"sum_denominator: {sum_denominator}")
            
            # Get ln(Ei)
            ln_Ei = sum_numerator / sum_denominator  # irradiance at pixel i
            ln_E[i] = ln_Ei                          # set in ln_E
        
        # Get E
        E = np.exp(ln_E).reshape((H,W))  # shape: H x W, irradiance computed from all images in dataset, for this channel
        hdr_img[..., c] = E              # generate HDR image channel-by-channel

    # HDR Generation Complete
    return hdr_img

def plot_response_curve(g_funcs):
    plt.figure(figsize=(5,5))
    g_func_r, g_func_g, g_func_b = g_funcs
    plt.plot(g_func_r, range(256), 'red')
    plt.plot(g_func_g, range(256), 'green')
    plt.plot(g_func_b, range(256), 'blue')
    plt.ylabel("Pixel Value Z")
    plt.xlabel("Log Exposure x")
    plt.savefig(output_dir + "responsecurve.png")

def plot_radiance_map(hdr_image):
    plt.figure()
    plt.imshow(np.log(cv2.cvtColor(hdr_image, cv2.COLOR_RGB2GRAY)), cmap='turbo') # convert hrd to grayscale otherwise the result is not heatmap
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir + "radiance_map.png")

def debevec(imgs_list, exp_time_list, P, N=20):
    '''
    1. Sample from each channel of each images
    2. Find function g() of each channel

    imgs_list     : List of 8-bit digital images numpy arrays; shape = P x H x W x C  (Memorial: 16 x 768 x 512 x 3)
    exp_time_list : List of exposure times; length = P  (Memorial: 16)
    N             : Number of sample points per image
    '''
    # Assumptions / Constants
    Zmax = 255
    Zmin = 0
    Lambda = 50
    # Image Dimension
    H, W, C = imgs_list[0].shape  # e.g (768, 512, 3)


    # Components of the Optimization Equation
    B = [math.log(exp_time,2) for exp_time in exp_time_list]  # Log base 2 of exposure time list; length P
    weight = []  # weight function for each pixel value; length 256 (0-255); actual values are (0-127,127-0)
    for z in range(Zmin, Zmax+1):
        if z <= 0.5*(Zmin + Zmax):
            weight.append(z - Zmin)
        else:
            weight.append(Zmax - z)
    
    # Sample pixel points from images
    Z_samples = uniform_img_sampler(imgs_list, P, H, W, N, num_channels=3)  # list of length num_channels, of items shape: P x N x N

    # Solve Ax=b (try single channel)
    g_funcs, lsq_errors = get_response_func(Z_samples, B, Lambda, weight)

    # Plot g Curves
    plot_response_curve(g_funcs)

    # Get HDR image
    hdr_img = get_hdr(
        imgs_list = imgs_list,
        g_funcs   = g_funcs,
        ln_exp_times = B,
        weight = weight, 
        W = W,
        H = H,
        C = C,
        P = P,
    )
    plot_radiance_map(hdr_img)

    return hdr_img



#========================================================================================================================
def main(args):
    
    imgs_name = getDatasetImageName(args.dataset)
    imgs = [np.asarray(Image.open(img_name).convert('RGB')) for img_name in imgs_name] # read image as numpy array

    #mtb = MTB(8)
    #shift_imgs = mtb.run(imgs, True) # list of numpy array, which are the aligned images

    # TODO: finish HDR algorithm
    exps = get_exposure(args.dataset_info)   # list of exposure times for all images
    P    = len(exps)                         # number of images
    hdr_img = debevec(imgs, exps, P, args.N)
    cv2.imwrite(output_dir + "hdr_image.hdr", hdr_img)
        
    return

if __name__ == '__main__':
    parser = ArgumentParser()
    '''add argument here'''
    parser.add_argument("--dataset", type=str, default="debug", choices=["memorial", "debug"]) # currently debug mode is for testing MTB
    parser.add_argument("--dataset_info", type=str, default="./../data/memorial/memorial.hdr_image_list.txt", help="Path to text file with info about dataset exposure times.")
    parser.add_argument("--N", type=int, default=20, help="Number of sample points per image, per channel.")
    args = parser.parse_args()
    main(args)


