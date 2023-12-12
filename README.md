This is a small Project using computer vision and deep learning which tries to Identify if a human face has any occlusions/addons such as sunglasses and masks. 
It tries to infer a close-up image/video of a face and classifies it as a Normal (neutral) face, face with sunglasses on, Face with masks on, or Face with Both. 

The Dataset is collected from a Publicly available source (https://github.com/ekremerakin/RealWorldOccludedFaces) which had images separated into 3 folders which further has sub-folders. The 3 parent folders were Neutral Faces, Faces with Sunglasses and Faces with Mask. For colecting the 4th class for the problem statement, I have collected some images from a different dataset (https://www.kaggle.com/datasets/hungkhoi99121816/rmfrd-for-masked-face-recognition) which had both sunglasses as well as masked images and selected a sample of it to be included to our 4 th class which is face with sunglasses and mask

For performing the training taks, the images for the 4 categories were arranged in such a way where there were only 4 folders inside which all the images of that respective class was placed. This was done using some basic python where we move the images from the subfolders inside the 4 class folders to the parent folders.

For Training I have used the Resnet-50 model from the model_zoo of the pytorch

Finally the 4 classes for training the model were

masked
masked-and-sunglasses
neutral
sunglasses
train images directory - FaceFirst\RealWorldOccludedFaces\images

video dir - FaceFirst\RealWorldOccludedFaces\test_video

test or prediction images directory - FaceFirst\RealWorldOccludedFaces\test_images


At the end of evaluating the trained ResNet-50 architecture model, it is subject to ONNX conversion and quantization as reduction in size and latency of the model while mainitaining the accuracy is a crucial part when it comes to deployment of any model in an edge device. 



# Real World Occluded Faces (ROF)

![Sample images from the ROF](resources/sample_data.png)


Real World Occluded Faces (ROF) dataset contains face image samples with real life upper-face and lower-face occlusions (i.e. face masks and sunglasses). The dataset contains 3195 neutral images, 1686 sunglasses images and 678 masked images.

These images are distributed among identities as such:
- All 180 identities have neutral images.
- 70 of these identities have both types of occlusion image samples.
- 110 of these identities have only sunglasses image samples.

All of the images are from real life scenarios and have large variations in pose and illumination. Images are collected from Google Image Search using the process described in [1] with some modifications.

**`10.09.2021`**: Dataset is now available! We have included a [download.py](https://github.com/ekremerakin/RealWorldOccludedFaces/blob/main/download.py) script to obtain the jpg formated images from pickle files provided.

```bash
$ cd RealWorldOccludedFaces-main
$ python download.py
```

# Pre-processing
For face detection combination of MTCNN and RetinaFace is used. The bounding boxes are extanded by a factor of 0.3 to include the whole head as described in [1].

# Citation
If you find Real World Occluded Faces dataset useful and used in your reasearch, please consider citing the following paper:

```bash
@inproceedings{erakiotan2021recognizing,
  title={On Recognizing Occluded Faces in the Wild},
  author={Erak$\iota$n, Mustafa Ekrem and Demir, U{\u{g}}ur and Ekenel, Haz$\iota$m Kemal},
  booktitle={2021 International Conference of the Biometrics Special Interest Group (BIOSIG)},
  pages={1--5},
  year={2021},
  organization={IEEE}
}
```

## References
[1] Cao, Q., Shen, L., Xie, W., Parkhi, O. M., & Zisserman, A. (2018, May). Vggface2: A dataset for recognising faces across pose and age. In 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018) (pp. 67-74). IEEE.
