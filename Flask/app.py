import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import datetime
import matplotlib.pyplot as plt
import io
import base64

# Create the Flask app
app = Flask(__name__)

# Load the trained machine learning model and the column data
model = pickle.load(open('model_rf.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))


# Define the route for the home page
@app.route('/')
def home():
    return render_template('home.html')


# Define the route for the prediction form page
@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')


# Define the route for the about page
@app.route('/about')
def about():
    return render_template('about.html')


# Define the route that handles the prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    # Use the current month and year for simplicity
    now = datetime.datetime.now()

    # Create a dictionary from the form data
    form_data = {
        'day': int(request.form['day']),
        'quarter': f"Quarter{request.form['quarter']}",
        'department': request.form['department'],
        'team': int(request.form['team']),
        'targeted_productivity': float(request.form['targeted_productivity']),
        'smv': float(request.form['smv']),
        'wip': float(request.form['wip']),
        'over_time': int(request.form['over_time']),
        'incentive': int(request.form['incentive']),
        'idle_time': float(request.form['idle_time']),
        'idle_men': int(request.form['idle_men']),
        'no_of_style_change': int(request.form['no_of_style_change']),
        'no_of_workers': float(request.form['no_of_workers']),
        'month': now.month, # Use current month
        'year': now.year    # Use current year
    }
    
    # Create a DataFrame from the form data
    input_df = pd.DataFrame([form_data])
    
    # Perform one-hot encoding
    input_df_encoded = pd.get_dummies(input_df)
    
    # Realign the columns to exactly match the model's training data
    input_df_realigned = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Make a prediction
    prediction = model.predict(input_df_realigned)
    output_percentage = round(prediction[0] * 100, 2)

    # Create a qualitative assessment message
    if output_percentage >= 80:
        assessment = "This employee is projected to be Highly Productive."
    elif output_percentage >= 60:
        assessment = "This employee is projected to have Good Productivity."
    else:
        assessment = "This employee's productivity may require attention."

    # Combine assessment and score into one final string
    final_prediction_text = f"{assessment} (Prediction Score: {output_percentage}%)"


    # --- Create the Visualization ---
    # Create a dictionary of only the numerical features for plotting
    numerical_features_for_plot = {k: v for k, v in form_data.items() if isinstance(v, (int, float))}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_names = list(numerical_features_for_plot.keys())
    feature_values = list(numerical_features_for_plot.values())
    
    ax.bar(feature_names, feature_values, color='#00c6ff')
    ax.set_xlabel("Employee Data Features")
    ax.set_ylabel("Value")
    ax.set_title("Input Data Snapshot")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot to a BytesIO object and encode to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('submit.html', prediction_text=final_prediction_text, plot_url=plot_url)


# This block runs the app when the script is executed
if __name__ == "__main__":
    app.run(debug=True)