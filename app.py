from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
try:
    with open("d_m.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print(" Model and scaler loaded successfully!")
except Exception as e:
    print(f" Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model or scaler not loaded properly"}), 500

    try:
        # Extract input data from form
        data = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["bp"]),
            float(request.form["skin_thickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"])
        ]
        
        # Convert data to numpy array and apply scaling
        input_data = np.array([data])
        input_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
