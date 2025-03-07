
# Fake News Detector

This project is a web application for detecting fake news using machine learning models. It is built with Flask and utilizes four different machine learning models to predict whether a given news article is fake or not.

## Features

- **Multiple Models**: The application uses four machine learning models:
  - Logistic Regression
  - Decision Tree
  - Gradient Boosting
  - Random Forest

- **Web Interface**: A simple and interactive web interface to input news articles and view predictions.

- **Text Preprocessing**: The application preprocesses the input text to improve model accuracy.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
Install dependencies:
Make sure you have Python installed. Then, install the required packages:

BASH

pip install -r requirements.txt
Download Models and Vectorizer:
Ensure you have the following files in the project directory:

vectorizer.pkl
logistic_regression_model.pkl
decision_tree_model.pkl
gradient_boosting_model.pkl
random_forest_model.pkl

##Usage
Run the Flask application:
python app.py
Access the application:
Open your web browser and go to http://127.0.0.1:5000/.

##Test the application:
Enter a news article in the text area.
Click the "Check" button to see predictions from all four models.

##Project Structure
app.py: The main Flask application file.
templates/index.html: The HTML template for the web interface.
static/: Directory for static files (e.g., CSS, JavaScript).
requirements.txt: List of Python dependencies.

##How It Works
Text Input: Users input a news article into the web form.
Preprocessing: The text is preprocessed to remove noise and standardize the input.
Vectorization: The preprocessed text is transformed into numerical features using a pre-trained vectorizer.
Prediction: The vectorized text is fed into four different machine learning models to predict whether the news is fake or not.
Results: The predictions from each model are displayed on the web page.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
