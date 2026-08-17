from flask import Flask, jsonify, render_template, request

from src.predictor import predict

app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "healthy", "service": "credit-risk-api"})


@app.post("/predict")
def predict_route():
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Request body must be JSON."}), 400
        return jsonify(predict(payload))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "detail": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
