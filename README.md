# Traffic Sign Detection 

## Overview
This project focuses on implementing and comparing two powerful convolutional neural network (CNN) architectures, **VGG-16** and **ResNet-101**, for traffic sign detection. Using a comprehensive dataset of traffic sign images, the models are trained, validated, and evaluated to analyze their performance. The goal is to enhance road safety and optimize intelligent transportation systems through accurate traffic sign recognition.

## Objectives
- Develop and compare the performance of **VGG-16** and **ResNet-101** for traffic sign detection.
- Preprocess and augment the dataset to ensure uniformity and robustness.
- Train the models using **TensorFlow** and **Keras**, optimizing hyperparameters for best results.
- Evaluate and compare the models based on accuracy, training time, and computational efficiency.
- Deploy the trained models for real-time detection and explore practical applications in traffic surveillance and autonomous vehicles.

## Dataset
The project utilizes the **German Traffic Sign Benchmark (GTSRB)**, consisting of over 50,000 images of various traffic signs under different conditions. The dataset ensures comprehensive testing of model robustness and adaptability.

## Key Features
- **CNN Architectures**: Implemented both **VGG-16** and **ResNet-101** using TensorFlow and Keras for in-depth comparison.
- **Data Augmentation**: Employed techniques like rotation, flipping, and scaling to enrich the dataset.
- **Training & Validation**: Models trained with appropriate optimizers and loss functions.
- **Performance Analysis**: Metrics such as accuracy and loss visualized for comparison.
- **Model Deployment**: Models evaluated on sample images for real-time prediction.

## System Requirements
- **Software**:
  - Python 3.x
  - Visual Studio Code
  - TensorFlow, Keras, NumPy, Pandas, Matplotlib
  - Kaggle for data exploration

## Implementation Steps
1. **Data Preparation**:
   - Load and preprocess the dataset.
   - Split into training and validation sets.
2. **Model Creation**:
   - Build VGG-16 and ResNet-101 architectures.
   - Integrate dropout and batch normalization for improved performance.
3. **Training**:
   - Train each model using the `Adam` and `RMSprop` optimizers.
   - Visualize training and validation accuracy/loss.
4. **Evaluation**:
   - Compare model performance using metrics and graphs.
5. **Testing**:
   - Test models on unseen traffic sign images for final prediction.

## Results
- **VGG-16** and **ResNet-101** models showed high accuracy in traffic sign detection.
- **ResNet-101** demonstrated better accuracy in real-world scenarios, attributed to its deeper architecture.
- The comparison chart revealed **ResNet-101** outperformed **VGG-16** in terms of validation accuracy.

## Conclusion
The project successfully implemented and evaluated **VGG-16** and **ResNet-101** for traffic sign detection, highlighting their strengths and areas for optimization. The insights gained contribute to the development of reliable traffic sign recognition systems, paving the way for enhanced road safety and advanced intelligent transportation solutions.

## References
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
