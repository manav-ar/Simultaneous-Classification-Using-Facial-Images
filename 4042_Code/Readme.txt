Avoid executing the evaluation file as the runtime is greater than 3 hours. Refer to existing outputs instead.

Download datasets (UTK and Adience) inside the main folder before running any code.

Most of the training was done locally using Tensorflow version 1.4. However, a couple of UTK models were trained on Google Colab with different Tensorflow version 2.0. The saved model outputs for TF 2.0 had to be converted to Onnx objects for standardization and compatibility with the local environment. 

The code files are configured for local training and migration to Google Colab may require configuration to path elements.
