# Inpainting Error Maximization

Author implementation of [Inpainting Error Maximization](https://arxiv.org/abs/2012.07287).

## Requirements
The code has been developed and tested with Python 3.8.8 and PyTorch 1.8.0.

## Running IEM

The repo will run IEM on the Flowers dataset by default, so you should download and prepare a folder with the dataset images beforehand.

To run IEM:

```
python main.py PATH_TO_DATASET
```

PATH_TO_DATASET should point to a folder with setid.mat and subdirectories 'jpg' and 'segmin'.

You can also change settings, such as the kernel size (--kernel-size) and the number of convolutions (--reps) for the Gaussian filter, through command-line arguments.

With the default settings (--kernel-size 11 --reps 2 --sigma 5.0), you should get the following output, which yields an IoU of 76.9 with ~77s runtime on a nVidia 1080ti.

![image](https://github.com/pym96/writing-paper/assets/105438207/a8d182c0-dbc9-44ff-9762-52b47007dcb4)

As you see above, training 100+ fundus images and pre-segment them just costs 5 seconds. 

![gdrishtiGS_010](https://github.com/pym96/writing-paper/assets/105438207/2d59e585-5f52-449b-a4da-debe155392d0)

![output_mask7](https://github.com/pym96/writing-paper/assets/105438207/4a3df2d9-acc0-485e-b6bc-d250f69ffcc9)


## Citation
```
@inproceedings{
savarese2021iem,
  title={Information-Theoretic Segmentation by Inpainting Error Maximization},
  author={Savarese, Pedro and Kim, Sunnie SY and Maire, Michael and Shakhnarovich, Greg and McAllester, David},
  booktitle={Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
