# Real World Occluded Faces Detection

This is a small Project using computer vision and deep learning which tries to Identify if a human face has any occlusions/addons such as sunglasses and masks. 
It tries to infer a close-up image/video of a face and classifies it as a Normal (neutral) face, face with sunglasses on, Face with masks on, or Face with Both. 

The Dataset is collected from a Publicly available source (https://github.com/ekremerakin/RealWorldOccludedFaces) which had images separated into 3 folders which further has sub-folders. The 3 parent folders were Neutral Faces, Faces with Sunglasses and Faces with Mask. For colecting the 4th class for the problem statement, I have collected some images from a different dataset (https://www.kaggle.com/datasets/hungkhoi99121816/rmfrd-for-masked-face-recognition) which had both sunglasses as well as masked images and selected a sample of it to be included to our 4 th class which is face with sunglasses and mask

For performing the training taks, the images for the 4 categories were arranged in such a way where there were only 4 folders inside which all the images of that respective class was placed. This was done using some basic python where we move the images from the subfolders inside the 4 class folders to the parent folders.

For Training I have used the Resnet-50 model from the model_zoo of the pytorch

Finally the 4 classes for training the model were

masked, masked-and-sunglasses, neutral and sunglasses.



At the end of evaluating the trained ResNet-50 architecture model, it is subject to ONNX conversion and quantization as reduction in size and latency of the model while mainitaining the accuracy is a crucial part when it comes to deployment of any model in an edge device. 



