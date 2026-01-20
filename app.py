from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('model/titanic_survival_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get inputs matching our 5 features
            pclass = int(request.form['Pclass'])
            sex = int(request.form['Sex']) # 0 for male, 1 for female
            age = float(request.form['Age'])
            sibsp = int(request.form['SibSp'])
            fare = float(request.form['Fare'])

            # Create array
            features = np.array([[pclass, sex, age, sibsp, fare]])
            
            # Predict
            prediction = model.predict(features)
            
            # Logic for result text
            result_text = "Survived! 🚢✨" if prediction[0] == 1 else "Did Not Survive 😔"
            color_class = "success" if prediction[0] == 1 else "danger"

            return render_template('index.html', prediction_text=result_text, color_class=color_class)
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)