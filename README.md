# Bone-Fracture-Detection---MURA
Here we attempt to create an algorithm to classify images for bone fracture, for this an input image is given and output is given as a “Positive” or “Negative” label. The input data is in the form of X Ray images of the bones. Hence, an appropriate supervised learning model is to be trained with the data to give correct label to the input image to predict a fracture. Some preprocessing of the data (converting RGB images to Grayscale) is also necessary here.
This repository contains a Keras implementation of a 169 layer Densenet Model on [MURA dataset](https://stanfordmlgroup.github.io/competitions/mura/)



You can train the model with mura.py

You can load the model with model_test.py
To get these particular graphs run the particular files:
1. Training Accuracy: plot_results_train_acc.py
2. Training loss: plot_results_train_loss.py
3. Validation Accuracy: plot_results_valid_acc.py
4. Validation Loss: plot_results_valid_loss.py

