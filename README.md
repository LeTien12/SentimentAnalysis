# 📊 SentimentAnalysis - Emotion Detection with BERT & DistilBERT

![GitHub stars](https://img.shields.io/github/stars/LeTien12/SentimentAnalysis?style=social)
![GitHub forks](https://img.shields.io/github/forks/LeTien12/SentimentAnalysis?style=social)
![GitHub issues](https://img.shields.io/github/issues/LeTien12/SentimentAnalysis)
![License](https://img.shields.io/github/license/LeTien12/SentimentAnalysis)

## 🌟 Overview

**SentimentAnalysis** is an NLP model for emotion detection trained on the **`dair-ai/emotion`** dataset. It supports **BERT (`bert-base-uncased`)** and **DistilBERT (`distilbert-base-uncased`)**, balancing accuracy and efficiency.

## 🚀 Features

- 🧠 **Supports BERT & DistilBERT** for flexible trade-off between speed and accuracy  
- 📊 **Trained on `dair-ai/emotion`**, detecting emotions like joy, anger, sadness, etc.  
- ⚡ **Efficient inference with DistilBERT** for real-time applications  
- 🔄 **Pretrained models available via Hugging Face**  

## 📦 Installation

Clone the repository and install dependencies:

1️⃣ **Clone the repository**  
   ```bash
   git clone https://github.com/LeTien12/SentimentAnalysis.git
   ```

2️⃣ **Navigate to the project directory**
   ```bash
    cd SentimentAnalysis
   ```

3️⃣ **Activate the Poetry virtual environment**  
   ```bash
   poetry shell
   ```

4️⃣ **Install dependencies**  
   ```bash
   poetry install
   ```
5️⃣ **run project**  
   ```bash
   poe run
   ```



## 🔥 Benchmarks
Both BERT and DistilBERT are fine-tuned on dair-ai/emotion, detecting emotions across six categories:
😃 Joy | 😡 Anger | 😢 Sadness | 😱 Fear | 🤢 Disgust | 😯 Surprise

## 🤝 Contributing
We welcome contributions! Follow these steps to contribute:

Fork the repo and create a new branch
Make your changes and test them
Open a Pull Request
For major changes, please open an issue first to discuss your proposal.


## 📄 License
- This project is licensed under the MIT License - see the LICENSE file for details.

## 🌟 Acknowledgments
## 🤗 Hugging Face Transformers for BERT & DistilBERT
## 📚 PyTorch for model training
## 📊 dair-ai/emotion dataset for emotion classification

