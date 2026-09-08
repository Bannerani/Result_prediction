import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "svm.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# HTML & CSS Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            --card-bg: rgba(255, 255, 255, 0.04);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --accent-hover: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --shadow-glow: 0 0 25px rgba(99, 102, 241, 0.15);
            --shadow-card: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: var(--shadow-card), var(--shadow-glow);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .container:hover {
            box-shadow: 0 25px 50px -12px rgba(99, 102, 241, 0.25);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.25rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-sub);
            font-size: 0.95rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.25s ease;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        .input-group input:focus, .input-group select:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25), inset 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        .btn-submit {
            margin-top: 2rem;
            width: 100%;
            padding: 1rem;
            background: var(--accent-gradient);
            border: none;
            border-radius: 12px;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:hover {
            background: var(--accent-hover);
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.5);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            text-align: center;
            animation: fadeIn 0.4s ease-in-out;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .result-card h2 {
            font-size: 1.1rem;
            color: var(--text-sub);
            margin-bottom: 0.25rem;
        }

        .result-card .score {
            font-size: 2.25rem;
            font-weight: 700;
            color: #a855f7;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Model Prediction Dashboard</h1>
        <p>Enter student metrics to compute predicted performance output</p>
    </div>

    <form method="POST" action="/predict">
        <div class="grid">
            <div class="input-group">
                <label>Age</label>
                <input type="number" name="age" step="any" placeholder="e.g. 20" required>
            </div>
            <div class="input-group">
                <label>Gender</label>
                <input type="number" name="gender" step="any" placeholder="Numeric value" required>
            </div>
            <div class="input-group">
                <label>Course</label>
                <input type="number" name="course" step="any" placeholder="Numeric value" required>
            </div>
            <div class="input-group">
                <label>Study Hours</label>
                <input type="number" name="study_hours" step="any" placeholder="e.g. 5.5" required>
            </div>
            <div class="input-group">
                <label>Class Attendance</label>
                <input type="number" name="class_attendance" step="any" placeholder="e.g. 85" required>
            </div>
            <div class="input-group">
                <label>Internet Access</label>
                <input type="number" name="internet_access" step="any" placeholder="Numeric value" required>
            </div>
            <div class="input-group">
                <label>Sleep Hours</label>
                <input type="number" name="sleep_hours" step="any" placeholder="e.g. 7" required>
            </div>
            <div class="input-group">
                <label>Sleep Quality</label>
                <input type="number" name="sleep_quality" step="any" placeholder="Numeric value" required>
            </div>
            <div class="input-group">
                <label>Study Method</label>
                <input type="number" name="study_method" step="any" placeholder="Numeric value" required>
            </div>
            <div class="input-group">
                <label>Facility Rating</label>
                <input type="number" name="facility_rating" step="any" placeholder="Numeric value" required>
            </div>
            <div class="input-group">
                <label>Exam Difficulty</label>
                <input type="number" name="exam_difficulty" step="any" placeholder="Numeric value" required>
            </div>
        </div>

        <button type="submit" class="btn-submit">Generate Prediction</button>
    </form>

    {% if prediction_text %}
    <div class="result-card">
        <h2>Predicted Score / Outcome</h2>
        <div class="score">{{ prediction_text }}</div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction_text="Model file missing or failed to load.")
    
    try:
        # Extract features in exact order as required by svm.pkl
        feature_keys = [
            'age', 'gender', 'course', 'study_hours', 'class_attendance',
            'internet_access', 'sleep_hours', 'sleep_quality', 
            'study_method', 'facility_rating', 'exam_difficulty'
        ]
        
        input_data = [float(request.form.get(key, 0)) for key in feature_keys]
        features_array = np.array([input_data])
        
        # Predict using SVR model
        prediction = model.predict(features_array)[0]
        result = f"{prediction:.2f}"
        
        return render_template_string(HTML_TEMPLATE, prediction_text=result)
    
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
