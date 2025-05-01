# Fake News Detection System 📰

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-sklearn-orange) ![Web App](https://img.shields.io/badge/Web%20App-Flask-green) ![License](https://img.shields.io/badge/License-MIT-blue)

This project implements a **Fake News Detection System** using Machine Learning (ML) algorithms, including Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest Classifiers. The system is trained on a dataset of fake and real news articles and predicts whether a given article is fake or not.

The application consists of the following components:
- 🧹 **Data Preprocessing**: Cleaning and preparing text data for analysis.
- 🤖 **Model Training**: Training multiple ML models on preprocessed data.
- 📊 **Model Evaluation**: Assessing model performance using classification metrics.
- 🌐 **Web Application**: A Flask-based interface for testing the model with user input.

---

## Table of Contents 📑

- [Project Overview](#project-overview) ℹ️
- [Technologies Used](#technologies-used) 🛠️
- [Data Preprocessing](#data-preprocessing) 🧹
- [Model Training](#model-training) 🤖
- [Web Application](#web-application) 🌐
- [Installation](#installation) ⚙️
- [Usage](#usage) 🚀
- [License](#license) 📜

---

## Project Overview ℹ️

The goal is to classify news articles as **Fake** or **Real** using machine learning models. The system is trained on labeled datasets, and trained models are saved for later use.

### Models Used
- 📈 **Logistic Regression**: A linear model for binary classification.
- 🌳 **Decision Tree**: A non-linear model for classification.
- 🚀 **Gradient Boosting**: An ensemble method combining weak learners.
- 🌲 **Random Forest**: An ensemble method using multiple decision trees.

### Project Workflow
1. 📂 **Data Collection**: Uses `Fake.csv` and `True.csv` for fake and real news.
2. 🧹 **Data Preprocessing**: Cleans text by removing punctuation, URLs, and special characters.
3. 🤖 **Model Training**: Trains models using TF-IDF vectorization.
4. 📊 **Model Evaluation**: Generates accuracy and classification reports.
5. 🌐 **Model Deployment**: Provides a web interface for testing with new inputs.

---

## Technologies Used 🛠️

- 🐍 **Programming Language**: Python 3.x
- 📚 **Libraries**:
  - `pandas`: Data manipulation and analysis.
  - `sklearn`: Machine learning and evaluation.
  - `flask`: Web application framework.
  - `pickle`: Model serialization.
  - `re`, `string`: Text preprocessing.
  - `matplotlib`, `seaborn`: Data visualization.
- 🤖 **Machine Learning Algorithms**:
  - Logistic Regression
  - Decision Tree
  - Gradient Boosting
  - Random Forest
- 📝 **Text Vectorization**: TF-IDF Vectorizer

---

## Data Preprocessing 🧹

The preprocessing pipeline prepares text data for training:

- ✂️ **Text Cleaning**:
  - Converts text to lowercase.
  - Removes special characters, URLs, HTML tags, punctuation, and numbers.
- 🔗 **Dataset Merging**:
  - Combines `Fake.csv` and `True.csv` into a single dataset.
  - Labels fake news as `0` and real news as `1`.
- 📊 **Data Splitting**:
  - Shuffles and splits data into training and test sets using `train_test_split`.

---

## Model Training 🤖

Four machine learning models are trained to classify news articles:

1. 📈 **Logistic Regression**: Baseline linear model for binary classification.
2. 🌳 **Decision Tree Classifier**: Non-linear model with a tree structure.
3. 🚀 **Gradient Boosting**: Ensemble method improving weak learners.
4. 🌲 **Random Forest**: Ensemble method combining multiple decision trees.

### Model Evaluation 📊
Models are evaluated using:
- ✅ **Accuracy**: Percentage of correctly classified articles.
- 📋 **Classification Report**: Precision, recall, and F1-score for detailed insights.

---

## Web Application 🌐

The system includes a **Flask-based web application** for users to input news articles and receive predictions.

### Application Flow
1. ✍️ Users enter a news article in a text box.
2. 🧹 Text is preprocessed using the training pipeline.
3. 🤖 Preprocessed text is passed to trained models for predictions.
4. 📄 Results display predictions from all four models.

### Running the Web Application
- 🖥️ **Flask Server**: Runs locally using Flask.
- 🔍 **Prediction**: Users input articles and view model predictions.

---

## Installation ⚙️

### Prerequisites
- 🐍 Python 3.x
- 📦 Required libraries (listed in `requirements.txt`)

### Installation Steps
1. 📥 Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git

2. 📂 Navigate to the project directory:
   cd fake-news-detection
   
4. 📦 Install dependencies:
   pip install -r requirements.txt
   
6. 🚀 Run the Flask application:
   python app.py

### Usage 🚀
🌐 Open your browser and navigate to http://127.0.0.1:5000/.
✍️ Enter a news article in the text box.
✅ Click Submit to get predictions from the four models.
📄 View results showing predictions from Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest.
