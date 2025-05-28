# 🏏 IPL Score Predictor

A machine learning-based web application built with Streamlit to predict the final score of an IPL (Indian Premier League) match using match progress and pitch conditions as input features.

---

## 📌 Features

* Predict final match score based on current runs, wickets, overs, and last 5 overs data.
* Venue-wise pitch conditions integrated: pace-friendly, spin-friendly, and detailed pitch descriptions.
* User-friendly web interface powered by **Streamlit**.
* Supports multiple ML models: Linear Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost.
* Integrated pitch behavior dataset for context-aware predictions.

---

## 🧠 Models Used

* **Linear Regression**
* **Decision Tree Regressor**
* **Random Forest Regressor**
* **Support Vector Regressor (SVR)**
* **K-Nearest Neighbors (KNN)**
* **XGBoost Regressor**

Each model was evaluated on MAE, MSE, RMSE, and R² score, and the best-performing one was saved for deployment.

---

## 🗂️ Dataset

* **Match Data**: Extracted from [Kaggle IPL datasets](https://www.kaggle.com/datasets).
* **Pitch Conditions**: Custom CSV file created based on historical data and expert insights for each stadium.

---

## 🏗️ Project Structure

```
📁 IPL_Score_Predictor/
│
├── pitch_conditions.csv            # Dataset with pitch type and description
├── ml_model.pkl                    # Trained ML model
├── ipl_score_predictor.py         # Streamlit app file
├── IPL_Score_Prediction.ipynb     # Model training and evaluation
├── README.md                       # This file
```

---

## ⚙️ How to Run Locally

1. Clone the repo:

   ```bash
   git clone https://github.com/msrvarry/IPL_Score_Pred.git
   cd IPL_Score_Pred
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run ipl_score_predictor.py
   ```

---

## 📊 Input Parameters

* Batting Team
* Bowling Team
* Current Runs
* Current Wickets
* Overs (min: 5.1)
* Runs and Wickets in Last 5 Overs
* Selected Venue (which loads pitch info)

---

## 📝 Output

* Predicted final score for the innings
* Display of pitch type, friendliness (spin/pace), and description

---

## 🔍 Future Work

* Include real-time player form and match context
* Live integration with APIs for auto-updating match data
* Use deep learning (RNN/LSTM) for sequence-based score forecasting
* Add ball-by-ball prediction granularity

---

## 🤝 Contributors

* [MSR Varshadh](https://github.com/msrvarry)

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---
