from flask import Flask, request, render_template
from flask_cors import CORS
from markupsafe import escape
import phonenumbers
from phonenumbers import geocoder
import joblib
import numpy as np
import re

app = Flask(__name__)
CORS(app)

# Load the trained ML model
model = joblib.load("spam_classifier2.joblib")  # updated model file name

# Raw known spam numbers (with formatting)
raw_known_spam_numbers = [
    "(202) 221-7923", "(469) 709-7630", "(805) 637-7243", "(878) 877-1402",
    "(865) 630-4266", "(863) 532-7969", "(858) 605-9622", "(312) 339-1227",
    "(904) 495-2559", "(917) 540-7996", "(301) 307-4601", "(347) 437-1689"
]

# Function to clean numbers to digits only
def clean_number(number):
    return re.sub(r'\D', '', number)

# Validation: check for valid length
def is_valid_number(number):
    digits = clean_number(number)
    return 10 <= len(digits) <= 15

# Clean and store known spam numbers as digits-only strings
known_spam_numbers = set(clean_number(num) for num in raw_known_spam_numbers)

# Feature extraction from phone number string
def extract_features(phone_number):
    number = clean_number(phone_number)
    length = len(number)
    starts_with_plus = 1 if phone_number.strip().startswith("+") else 0

    digit_pattern_1 = 0
    digit_pattern_2 = 0

    match1 = re.search(r'(\d{3})', number)
    if match1:
        digit_pattern_1 = int(match1.group(1))
    match2 = re.search(r'\d{3}(\d{2})', number)
    if match2:
        digit_pattern_2 = int(match2.group(1))

    features = np.array([length, starts_with_plus, digit_pattern_1, digit_pattern_2]).reshape(1, -1)
    return features

# Analyze and predict phone number spam status
def analyze_call(phone_number):
    number = clean_number(phone_number)

    # Validate number length
    if not is_valid_number(phone_number):
        return "Invalid Number", ["Invalid number length."], "Unknown"

    # Check known spam numbers first
    if number in known_spam_numbers:
        return "Spam Detected", [], "Unknown"

    suspicious = False
    reasons = []

    # Detect country using phonenumbers
    try:
        parsed_number = phonenumbers.parse(phone_number, "IN")
        country = geocoder.description_for_number(parsed_number, "en")
    except Exception:
        country = "Unknown"

    # Pattern: All digits same
    if len(set(number)) == 1:
        return "Suspicious", ["All digits are the same."], country

    # Pattern: Service/sales or suspicious prefixes
    if number.startswith(("140", "1800")):
        suspicious = True
        reasons.append("Starts with sales/service prefix (140 or 1800).")

    if any(number.startswith(prefix) for prefix in ["900", "700", "888"]):
        suspicious = True
        reasons.append("Known spam prefix")

    # ML model prediction
    try:
        features = extract_features(phone_number)
        model_pred = model.predict(features)[0]  # 0 or 1
        if model_pred == 1:
            prediction = "Spam Detected"
        else:
            prediction = "Safe"
    except Exception as e:
        prediction = "Unknown"
        reasons.append(f"Model error: {str(e)}")

    if prediction == "Unknown" and suspicious:
        prediction = "Suspicious"

    return prediction, reasons, country

# Flask routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/check_call", methods=["POST"])
def check_call():
    phone_number = request.form.get("phone_number", "")
    phone_number = escape(phone_number)

    prediction, reasons, country = analyze_call(phone_number)

    result = {
        "phone_number": phone_number,
        "prediction": prediction,
        "reasons": reasons,
        "country": country
    }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
