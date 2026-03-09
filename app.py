from flask import Flask, request, render_template
import os
from src.poultry_disease.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

model_path = "artifacts/training/model"
predictor = PredictionPipeline(model_path)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    result = predictor.predict(filepath)

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000)