# NTU VFX2023 Homework 1: High Dynamic Range Imaging

## Description:
website: https://www.csie.ntu.edu.tw/~cyy/courses/vfx/21spring/assignments/proj1/
### TL;DR: 
1. Taking photographs.
    - Take a series of photographs for a scene under different exposures. (by changing shutter speed)
2. Write a program to assemble an HDR image.
    - we implemented the Paul Debevec's method. (Recovering High Dynamic Range Radiance Maps from Photographs, SIGGRAPH 1997.)
3. Develop your radiance map using tone mapping.
    - we implemented Reinhard's method. (Photographics Tone Reproduction for Digital Images, SIGGRAPH 2002.)
4. Bonuses:
    - we implemented the MTB image alignment method for HDR imaging.

## Member:
- 電信所碩一 顏子鈞
- 電信所碩一 陳詠源

## Dependencies:
please install them using ```pip install -r requirements.txt```. packages used in our code:
- numpy, pillow, opencv-python, matplotlib

## Execution:
Generate HDR Image and Tonemapped LDR Image:
```
$ cd code
$ python main.py --dataset <name_of_dataset> --dataset_info <path_to_textfile> --N <num_of_samplepoints_per_image> --tm_key <key_to_control_tonemapping>
```
Note:  
- to disable MTB image alignment, use ```--disable_mtb``` flag
- choices of ```<name_of_dataset>```: memorial, ntu_sample1, ntu_sample2, ntu_sample3, ntu_sample4, default = ntu_sample1
- ```<num_of_samplepoints_per_image>``` is the sampling point used for calculating Debevec HDR algorithm, default value = 40
- ```<key_to_control_tonemapping>``` is the key parameter of Reinhard's photographic tonemapping algorithm, default value = 0.5
- output files are all save at ```"data/output/"``` directory, including HDR irradiance map, camera response curve, tonemapped LDR image (global and local operation).





## Results:
### ntu_sample1:
- original captured images:  

<img src="data/ntu_sample1/_C4A1568.JPG" height="256"> <img src="data/ntu_sample1/_C4A1571.JPG" height="256">
</br>
<img src="data/ntu_sample1/_C4A1573.JPG" height="256"> <img src="data/ntu_sample1/_C4A1575.JPG" height="256">

- calculated irradiance map:  
<img src="data/output/rad_ntu_sample1.png" height="512">

- reconstructed HDR images:  
<img src="data/output/output_ntu_sample1.png" height="448">

### ntu_sample3:
- original captured images:  

<img src="data/ntu_sample3/_C4A1528.JPG" height="256"> <img src="data/ntu_sample3/_C4A1526.JPG" height="256">
<br/>
<img src="data/ntu_sample3/_C4A1524.JPG" height="256"> <img src="data/ntu_sample3/_C4A1522.JPG" height="256">

- calculated irradiance map:  
<img src="data/output/rad_ntu_sample3.png" height="512">

- reconstructed HDR images:  
<img src="data/output/output_ntu_sample3.png" height="448">