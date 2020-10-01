# Bone-Fracture-Detection---MURA
Here we attempt to create an algorithm to classify images for bone fracture, for this an input image is given and output is given as a “Positive” or “Negative” label. The input data is in the form of X Ray images of the bones. Hence, an appropriate supervised learning model is to be trained with the data to give correct label to the input image to predict a fracture. Some preprocessing of the data (converting RGB images to Grayscale) is also necessary here.
This repository contains a Keras implementation of a 169 layer Densenet Model on [MURA dataset](https://stanfordmlgroup.github.io/competitions/mura/)

Here, I trained the Densenet on **XR_HUMERUS** of the dataset for **52 epochs** with a **batch size of 8**.

To load the dataset you can use the function [**data_loader.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/data_loader.py).

You can train the model with [**mura.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/mura.py).

You can load the model with [**model_test.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/model_test.py).

To get these particular graphs run the particular files:
1. Training Accuracy: [**plot_results_train_acc.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/plot_results_train_acc.py)
![Training Accuracy](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/figures/plot_MURA_train_acc.jpg)

2. Training loss: [**plot_results_train_loss.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/plot_results_train_loss.py)
![Training Loss](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/figures/plot_MURA_train_loss.jpg)

3. Validation Accuracy: [**plot_results_valid_acc.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/plot_results_valid_acc.py)
![Validation Accuracy](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/figures/plot_MURA_valid_acc.jpg)

4. Validation Loss: [**plot_results_valid_loss.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/plot_results_valid_loss.py)
![Validation Loss](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/figures/plot_MURA_valid_loss.jpg)






