from flask import Flask, request, render_template
import pandas as pd
import re
import string
import pickle

app = Flask(__name__)

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\$\$.*?\$\$', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

# Load the vectorizer and models
with open('vectorizer.pkl', 'rb') as f:
    vectorization = pickle.load(f)
with open('logistic_regression_model.pkl', 'rb') as f:
    LR = pickle.load(f)
with open('decision_tree_model.pkl', 'rb') as f:
    DT = pickle.load(f)
with open('gradient_boosting_model.pkl', 'rb') as f:
    GB = pickle.load(f)
with open('random_forest_model.pkl', 'rb') as f:
    RF = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        news = request.form['news']
        testing_news = {"text": [news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test['text'] = new_def_test["text"].apply(wordopt)
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)

        # Make predictions
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_GB = GB.predict(new_xv_test)
        pred_RF = RF.predict(new_xv_test)

        # Prepare the results
        results = {
            "Logistic_Regression Prediction  : ": output_label(pred_LR[0]),
            "Decision_Tree Prediction        : ": output_label(pred_DT[0]),
            "Gradient_Boost Class Prediction : ": output_label(pred_GB[0]),
            "Random_Forest Class Prediction  : ": output_label(pred_RF[0])
        }

        return render_template('index.html', results=results, news=news)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)