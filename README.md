# NTU VFX2023 Homework 1

### execution:
```
Generate HDR Image:
$ python main.py --dataset <name_of_dataset> --dataset_info <path_to_textfile> --N <num_of_samplepoints_per_image>

Generate Tonemapped LDR Image:
$ python tone_mapping.py 

Ex.
python hdr_generator.py --dataset memorial --dataset_info ./../data/memorial/memorial.hdr_image_list.txt --N 20
```
Note:
- Currently ```<name_of_dataset>``` is set as ```debug```, which was for MTB debugging.
- Output files are all saved at "./../data/output/" directory