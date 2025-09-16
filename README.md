# 🐾 Animal Classification using CNN

This project is a **Deep Learning based Animal Image Classification System** that uses **Convolutional Neural Networks (CNN)** to classify images of animals such as Dog, Cat, Tiger, Lion, Elephant, Zebra, Panda, Bear, and more.  

It is built using **TensorFlow, Keras, and Streamlit** and includes an interactive web app where users can upload images and get real-time predictions with confidence scores.

## ✨ Features
- Multi-class Animal Image Classification (15+ species)
- Pretrained CNN Models: **MobileNetV2, VGG16, ResNet50**
- Model Comparison & Accuracy Charts
- Interactive **Streamlit Web App**
- Custom Image Upload & Prediction
- Trained on Animal Dataset with Augmentation

## 🛠️ Tech Stack
- **Python**
- **TensorFlow / Keras**
- **NumPy, Matplotlib**
- **Streamlit (UI)**
- **Jupyter Notebook**

## 🚀 Live Demo
👉 [Try it on Streamlit Cloud](https://animal-classification-cnn-36kbbwxe8axrr8sgssxufx.streamlit.app/)

## 📂 Dataset
The dataset contains images of multiple animals including:
- Dog 🐶
- Cat 🐱
- Tiger 🐯
- Lion 🦁
- Elephant 🐘
- Zebra 🦓
- Panda 🐼
- Giraffe 🦒
- and more…

## 📊 Results
The best performing model is **MobileNetV2** achieving **~87% accuracy** on validation data.

## 🔧 Installation
```bash
git clone git@github.com:Faran-samra/animal-classification-cnn.git
cd animal-classification-cnn
pip install -r requirements.txt
streamlit run app.py
