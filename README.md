📊 Customer Churn Prediction Web App

This project is a Simple Customer Churn Prediction web application built using Streamlit and a trained Deep Learning model (TensorFlow/Keras).
The app predicts whether a bank customer is likely to churn (leave the bank) based on customer details.

🚀 Features

Interactive web interface built with Streamlit

Takes customer information as input

Uses a trained neural network model to predict churn probability

Displays churn probability and final prediction result

🧠 Model Used

Deep Learning model built using TensorFlow / Keras

Preprocessing includes:

Label Encoding (Gender)

One-Hot Encoding (Geography)

Standard Scaling (Numerical Features)

Saved files used:

model.h5 → Trained model

label_encoder_gender.pkl → Gender label encoder

onehot_encoder_geo.pkl → Geography one-hot encoder

scaler.pkl → Feature scaler

🖥️ Tech Stack

Python

Streamlit

TensorFlow / Keras

Scikit-learn

Pandas

NumPy

📂 Project Structure
churn-prediction-app/
│
├── app.py                  # Streamlit application
├── model.h5                # Trained neural network model
├── scaler.pkl              # StandardScaler object
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── requirements.txt
└── README.md

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py


The app will open in your browser at:

http://localhost:8501

📝 How It Works

User enters customer details

Input data is encoded and scaled

Processed data is passed to the trained model

Model returns churn probability

App displays:

Churn Probability

Final Prediction (Likely to churn / Not likely)

📈 Example Output
Churn Probability: 0.73
The customer is likely to churn.

🙌 Future Improvements

Deploy on cloud (Streamlit Cloud / Render / Hugging Face Spaces)

Add batch prediction

Improve model accuracy

Store prediction history

👤 Author

Rohit Yadav