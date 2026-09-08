import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the SVR model
MODEL_PATH = "svm.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Categorical options map matching model feature expectations
CATEGORICAL_OPTIONS = {
    "gender": ["Male", "Female", "Other"],
    "course": ["Computer Science", "Engineering", "Business", "Arts", "Medicine", "Law"],
    "internet_access": ["Yes", "No"],
    "sleep_quality": ["Poor", "Average", "Good", "Excellent"],
    "study_method": ["Self Study", "Group Study", "Online Lectures", "Tutor"],
    "exam_difficulty": ["Easy", "Medium", "Hard"]
}

# Simple manual encoding dictionary for ordinal/categorical fields
ENCODING_MAPS = {
    "gender": {"Male": 0, "Female": 1, "Other": 2},
    "course": {"Computer Science": 0, "Engineering": 1, "Business": 2, "Arts": 3, "Medicine": 4, "Law": 5},
    "internet_access": {"No": 0, "Yes": 1},
    "sleep_quality": {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3},
    "study_method": {"Self Study": 0, "Group Study": 1, "Online Lectures": 2, "Tutor": 3},
    "exam_difficulty": {"Easy": 0, "Medium": 1, "Hard": 2}
}

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVR Model Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 20px 0;
        }
        .card-custom {
            background: #ffffff;
            border: none;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12), 0 4px 8px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .card-custom:hover {
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15), 0 6px 12px rgba(0, 0, 0, 0.08);
        }
        .btn-primary-custom {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 8px;
            padding: 12px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(118, 75, 162, 0.3);
            transition: all 0.3s ease;
        }
        .btn-primary-custom:hover {
            opacity: 0.95;
            box-shadow: 0 6px 16px rgba(118, 75, 162, 0.4);
            transform: translateY(-1px);
        }
        .form-control, .form-select {
            border-radius: 8px;
            padding: 10px 14px;
            border: 1px solid #e2e8f0;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.03);
        }
        .form-control:focus, .form-select:focus {
            border-color: #764ba2;
            box-shadow: 0 0 0 3px rgba(118, 75, 162, 0.15);
        }
        .result-box {
            background: #f8fafc;
            border-left: 4px solid #764ba2;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        }
    </style>
</head>
<body>
<div class="container my-5">
    <div class="row justify-content-center">
        <div class="col-lg-8 col-md-10">
            <div class="card card-custom p-4 p-md-5">
                <h2 class="text-center mb-4 font-weight-bold" style="color: #2d3748;">Performance Predictor</h2>
                
                {% if prediction is not none %}
                    <div class="result-box mb-4 text-center">
                        <h4 class="m-0 text-muted">Predicted Score / Value</h4>
                        <span class="display-5 fw-bold" style="color: #764ba2;">{{ prediction }}</span>
                    </div>
                {% endif %}

                <form action="/predict" method="POST">
                    <div class="row g-3">
                        <div class="col-md-6">
                            <label class="form-label">Age</label>
                            <input type="number" step="any" name="age" class="form-control" placeholder="e.g. 20" required>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Gender</label>
                            <select name="gender" class="form-select" required>
                                {% for opt in categorical_options['gender'] %}
                                    <option value="{{ opt }}">{{ opt }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Course</label>
                            <select name="course" class="form-select" required>
                                {% for opt in categorical_options['course'] %}
                                    <option value="{{ opt }}">{{ opt }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Study Hours</label>
                            <input type="number" step="any" name="study_hours" class="form-control" placeholder="e.g. 5.5" required>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Class Attendance (%)</label>
                            <input type="number" step="any" name="class_attendance" class="form-control" placeholder="e.g. 85" required>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Internet Access</label>
                            <select name="internet_access" class="form-select" required>
                                {% for opt in categorical_options['internet_access'] %}
                                    <option value="{{ opt }}">{{ opt }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Sleep Hours</label>
                            <input type="number" step="any" name="sleep_hours" class="form-control" placeholder="e.g. 7" required>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Sleep Quality</label>
                            <select name="sleep_quality" class="form-select" required>
                                {% for opt in categorical_options['sleep_quality'] %}
                                    <option value="{{ opt }}">{{ opt }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Study Method</label>
                            <select name="study_method" class="form-select" required>
                                {% for opt in categorical_options['study_method'] %}
                                    <option value="{{ opt }}">{{ opt }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Facility Rating (1-5)</label>
                            <input type="number" step="any" name="facility_rating" class="form-control" placeholder="e.g. 4" required>
                        </div>
                        <div class="col-12">
                            <label class="form-label">Exam Difficulty</label>
                            <select name="exam_difficulty" class="form-select" required>
                                {% for opt in categorical_options['exam_difficulty'] %}
                                    <option value="{{ opt }}">{{ opt }}</option>
                                {% endfor %}
                            </select>
                        </div>
                    </div>
                    <button type="submit" class="btn btn-primary-custom text-white w-100 mt-4">Generate Prediction</button>
                </form>
            </div>
        </div>
    </div>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_LAYOUT, categorical_options=CATEGORICAL_OPTIONS, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not loaded.", 500

    form_data = request.form
    
    # Process and encode feature vector according to model sequence:
    # ['age', 'gender', 'course', 'study_hours', 'class_attendance', 'internet_access', 
    #  'sleep_hours', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
    
    features = [
        float(form_data.get("age")),
        ENCODING_MAPS["gender"].get(form_data.get("gender"), 0),
        ENCODING_MAPS["course"].get(form_data.get("course"), 0),
        float(form_data.get("study_hours")),
        float(form_data.get("class_attendance")),
        ENCODING_MAPS["internet_access"].get(form_data.get("internet_access"), 0),
        float(form_data.get("sleep_hours")),
        ENCODING_MAPS["sleep_quality"].get(form_data.get("sleep_quality"), 0),
        ENCODING_MAPS["study_method"].get(form_data.get("study_method"), 0),
        float(form_data.get("facility_rating")),
        ENCODING_MAPS["exam_difficulty"].get(form_data.get("exam_difficulty"), 0)
    ]

    prediction = model.predict(np.array([features]))[0]
    formatted_prediction = round(float(prediction), 2)

    return render_template_string(
        HTML_LAYOUT, 
        categorical_options=CATEGORICAL_OPTIONS, 
        prediction=formatted_prediction
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
