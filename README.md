# Fake News Detection System

This project implements a Fake News Detection system using Machine Learning (ML) algorithms, including Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest Classifiers. The system is trained on a dataset containing both fake and real news, and it predicts whether a given news article is fake or not.

The application consists of the following components:
1. **Data Preprocessing**: Cleaning the dataset and preparing the text for analysis.
2. **Model Training**: Training multiple ML models using the preprocessed data.
3. **Model Evaluation**: Evaluating the performance of models using classification metrics.
4. **Web Application**: A Flask-based web application for testing the model with user input.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Overview

The goal of this project is to build a system that classifies news articles as "Fake" or "Real" using machine learning models. The system is trained on the dataset containing news articles labeled as fake or true, and the trained models are stored for later use.

Models used in this project:
- **Logistic Regression**: A linear model for binary classification.
- **Decision Tree**: A non-linear model used for classification.
- **Gradient Boosting**: A boosting algorithm that combines the predictions of weak learners.
- **Random Forest**: An ensemble method using multiple decision trees.

### Steps in the project:
1. **Data Collection**: The project uses two CSV files, `Fake.csv` and `True.csv`, which contain fake and real news articles, respectively.
2. **Data Preprocessing**: The text data is cleaned to remove irrelevant information such as punctuation, URLs, and special characters.
3. **Model Training**: Models are trained on the preprocessed text data using the TF-IDF vectorization technique.
4. **Model Evaluation**: The models' accuracy and classification reports are generated to evaluate their performance.
5. **Model Deployment**: A web interface is created to allow users to manually test the models with new news inputs.

## Technologies Used

- **Python**: Primary programming language.
- **Libraries**:
  - `pandas`: For data manipulation.
  - `sklearn`: For machine learning and model evaluation.
  - `flask`: For web application.
  - `pickle`: For saving and loading trained models.
  - `re`, `string`: For text preprocessing.
  - `matplotlib`, `seaborn`: For data visualization.
- **Machine Learning Algorithms**:
  - Logistic Regression
  - Decision Tree
  - Gradient Boosting
  - Random Forest
- **Vectorization**: TF-IDF Vectorizer for converting text data into numerical form.

## Data Preprocessing

Data preprocessing involves the following steps:
- **Text Cleaning**: We use a custom function to preprocess the text, which includes:
  - Lowercasing the text.
  - Removing special characters, URLs, and HTML tags.
  - Removing punctuation and numbers.
- **Dataset Merging**: The fake and true news datasets are combined into one, with a class label assigned to each article:
  - Fake news articles are labeled as `0`.
  - Real news articles are labeled as `1`.
  
The data is shuffled and split into training and test sets using `train_test_split` from `sklearn`.

## Model Training

We train four machine learning models to predict whether a news article is fake or real:
1. **Logistic Regression**: This is the baseline linear model used for classification.
2. **Decision Tree Classifier**: A non-linear model that builds a tree structure to classify data.
3. **Gradient Boosting**: An ensemble method that builds a strong model from a sequence of weak models.
4. **Random Forest**: An ensemble learning method that builds multiple decision trees to improve classification accuracy.

### Model Evaluation

After training the models, we evaluate them using the following metrics:
- **Accuracy**: Percentage of correctly classified articles.
- **Classification Report**: Detailed performance metrics, including precision, recall, and F1-score.

## Web Application

The system includes a Flask-based web application where users can input news articles, and the system will predict whether the article is fake or real based on the trained models.

### Application Flow:
1. The user enters a news article in a text box on the web page.
2. The text is preprocessed using the same steps as in the model training phase.
3. The preprocessed text is passed to the trained models, which return predictions for each model.
4. The results are displayed on the web page, showing whether the news is fake or real for each model.

### Running the Web Application
1. **Flask Server**: The app is served using Flask and can be run locally.
2. **Prediction**: Users can manually input a news article and get predictions from multiple models.

## Installation

### Prerequisites:
1. Python 3.x
2. Required libraries (listed in `requirements.txt`)

### Steps to Install:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
2. Navigate to the project folder:

    cd fake-news-detection
3. Install the required libraries:
   pip install -r requirements.txt
4. Run the Flask app:
   python app.py

   
The application will be accessible at http://127.0.0.1:5000/.

Usage
Once the application is running, follow these steps:

Open the application in your browser (http://127.0.0.1:5000/).

Enter a news article in the text box.

Click the "Submit" button to get predictions from the four models.

The results will be displayed on the web page, showing the predictions from the Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest models.
