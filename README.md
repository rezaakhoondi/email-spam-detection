# 📧 Email Spam Detection with Machine Learning  

This repository contains a Jupyter Notebook that demonstrates how to build a machine learning model to classify emails as **spam** or **ham (not spam)**.  

The project walks through the full machine learning pipeline: from loading and preprocessing the dataset, to training and evaluating a classification model.  

---

## 🚀 Features
- Clean and well-documented Jupyter Notebook.  
- Text preprocessing (lowercasing, punctuation removal, optional stopword removal).  
- Feature extraction using **TF-IDF vectorization**.  
- Baseline model: **Multinomial Naive Bayes** for text classification.  
- Evaluation with metrics such as Accuracy, Precision, Recall, F1-Score.  
- Visualizations: **Confusion Matrix** and optional **ROC Curve**.  

---

## 📂 Project Structure
├── spam_detection.ipynb # Main Jupyter Notebook
├── spam.csv # Dataset (if included)
├── requirements.txt # Dependencies
└── README.md # Project documentation

---

## 📊 Dataset
We use the **Spam SMS Collection Dataset**, which contains 5,574 labeled messages.  

- **v1** → Label (`ham` or `spam`)  
- **v2** → Email/SMS message text  

📌 Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)  

---

## ⚙️ Installation & Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/email-spam-detection.git
   cd email-spam-detection
2. Install dependencies:
    pip install -r requirements.txt
3. Run the Jupyter Notebook:
    jupyter notebook spam_detection.ipynb
   
## 🧪 Results

    Naive Bayes achieved strong performance on the dataset.
    
    The confusion matrix and classification report are included inside the notebook.
    
    Results may vary slightly depending on random train-test splits.
## 🔮 Future Work
    
    Experiment with other algorithms (Logistic Regression, SVM, Random Forest, Deep Learning).
    
    Hyperparameter tuning for improved accuracy.
    
    Apply additional preprocessing (n-grams, lemmatization, etc.).
## 📜 License

    This project is licensed under the MIT License — feel free to use, modify, and share.
## 🙌 Acknowledgments

    Dataset: UCI SMS Spam Collection Dataset
    
    Inspired by common approaches in natural language processing and machine learning.