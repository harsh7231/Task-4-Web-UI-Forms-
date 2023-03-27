from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(_name_)
bootstrap = Bootstrap(app)

# Load the dataset
df = pd.read_csv('customer_data.csv')

# Split the dataset into training and testing sets
X = df.drop('purchase', axis=1)
y = df['purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle form submission
        age = int(request.form['age'])
        gender = request.form['gender']
        income = int(request.form['income'])
        gender_binary = 1 if gender == 'Male' else 0
        new_data = np.array([[age, gender_binary, income]])
        prediction = lr_model.predict(new_data)[0]
        prediction_text = 'will' if prediction == 1 else 'will not'
        return render_template('index.html', prediction_text=prediction_text)
    else:
        # Show the form
        return render_template('index.html')

if _name_ == '_main_':
    app.run(debug=True)
