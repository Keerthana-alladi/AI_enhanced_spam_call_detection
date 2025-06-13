# 📞 Spam Call Classifier

A machine learning-based project to classify phone calls as **spam** or **not spam** based on call attributes. The model is trained and saved using `scikit-learn`.

---

## 🔍 Project Overview

This project aims to detect spam calls using classification algorithms. It involves:

- Data preprocessing
- Model training
- Evaluation
- Saving the trained model for deployment

---

## 🧠 Model Details

- Trained using **scikit-learn**
- Model saved using `joblib` as `spam_classifier2.joblib`
- Input features may include:
  - Call duration
  - Frequency
  - Number of past spam reports
  - Whether the number is saved or unknown

---

## 🛠️ Technologies & Libraries

- Python 3.x
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/spam-call-classifier.git
   cd spam-call-classifier
Install dependencies
pip install -r requirements.txt
Load and use the model
import joblib

# Load model
model = joblib.load('spam_classifier2.joblib')

# Example prediction
sample_input = [[120, 3, 0]]  # Example: duration, frequency, saved_number
prediction = model.predict(sample_input)
print("Prediction:", prediction)
![WhatsApp Image 2025-06-07 at 11 54 21_9535f105](https://github.com/user-attachments/assets/fabf5e7d-1791-45c7-87b8-e4d7d628ef99)
![WhatsApp Image 2025-06-07 at 11 55 29_40ebf94d](https://github.com/user-attachments/assets/96f56ab0-13fa-4a58-b857-59ab5cf7245a)

