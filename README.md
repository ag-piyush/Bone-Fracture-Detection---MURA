# Bone-Fracture-Detection---MURA
Here we attempt to create an algorithm to classify images for bone fracture, for this an input image is given and output is given as a “Positive” or “Negative” label. The input data is in the form of X Ray images of the bones. Hence, an appropriate supervised learning model is to be trained with the data to give correct label to the input image to predict a fracture. Some preprocessing of the data (converting RGB images to Grayscale) is also necessary here.
This repository contains a Keras implementation of a 169 layer Densenet Model on [MURA dataset](https://stanfordmlgroup.github.io/competitions/mura/)

- Here, I trained the Densenet on **XR_HUMERUS** of the dataset for **52 epochs** with a **batch size of 8**.
- To load the dataset you can use the function [**data_loader.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/data_loader.py).
- To train the model run the file [**mura.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/mura.py).
- To load the model run the file [**model_test.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/model_test.py).
- To get these particular graphs run the following files:
  1. Training Accuracy: [**plot_results_train_acc.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/plot_results_train_acc.py)
  <p align="center">
    <img width="460" height="300" src="https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/figures/plot_MURA_train_acc.jpg">
  </p>
  
  2. Validation Accuracy: [**plot_results_valid_acc.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/plot_results_valid_acc.py)
  <p align="center">
    <img width="460" height="300" src="https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/figures/plot_MURA_valid_acc.jpg">
  </p>

  3. Training loss: [**plot_results_train_loss.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/plot_results_train_loss.py)
  <p align="center">
    <img width="460" height="300" src="https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/figures/plot_MURA_train_loss.jpg">
  </p>
 
  4. Validation Loss: [**plot_results_valid_loss.py**](https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/plot_results_valid_loss.py)
  <p align="center">
    <img width="460" height="300" src="https://github.com/ag-piyush/Bone-Fracture-Detection---MURA/blob/master/figures/plot_MURA_valid_loss.jpg">
  </p>






