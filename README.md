# 🧠 Student Mental Health & Depression Prediction

> A machine learning system for early identification of depression risk in students, built on a dataset of 27,901 records with 6 classification models and an interactive web application.

**Author:** Huma Aslam
**Live Demo:** [Student Depression Predictor on Hugging Face](https://huggingface.co/spaces/thehmfpk/student-depression-predictor)
**Dataset:** [Student Depression Dataset — Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)

---

## 📌 Project Overview

Mental health among students is a growing concern globally. This project applies supervised machine learning to predict depression risk based on academic, lifestyle, and demographic factors — enabling early intervention before conditions worsen.

The system outputs a **risk classification (Low / High)** along with a **Risk Quotient score (%)**, rather than a clinical diagnosis, reflecting responsible and ethical use of ML in healthcare contexts.

---

## 🌐 Live Application

The app is deployed on Hugging Face Spaces and accessible permanently at:

🔗 **[https://huggingface.co/spaces/thehmfpk/student-depression-predictor](https://huggingface.co/spaces/thehmfpk/student-depression-predictor)**

### App Features
| Tab | Description |
|-----|-------------|
| 🔬 Screening & Diagnostics Dashboard | Input student profile, get instant risk classification |
| 📊 Analytics & Risk Visualisation | Visual breakdown of risk factors and scores |
| 💊 Wellness Remedies & Action Plan | Personalised recommendations based on risk level |
| 📚 Educational Infrastructure & Clinical Metadata | Background on methodology and data sources |

---

## 📂 Dataset

| Property | Detail |
|----------|--------|
| Source | Kaggle — Student Depression Dataset |
| Records | 27,901 students |
| Features | 17 input features + 1 target |
| Target | Depression (0 = No, 1 = Yes) |
| Class Balance | 58.5% Depressed / 41.5% Not Depressed |

### Key Features Used
- Age, Gender, CGPA
- Academic Pressure, Work Pressure, Financial Stress
- Study Satisfaction, Job Satisfaction
- Sleep Duration, Dietary Habits
- Work/Study Hours per Day
- Suicidal Thoughts History
- Family History of Mental Illness
- City, Profession, Degree

---

## ⚙️ Methodology

### Preprocessing
- **Ordinal Encoding** for Sleep Duration and Dietary Habits (natural order preserved)
- **Binary Encoding** for Gender, Suicidal Thoughts, Family History
- **One-Hot Encoding** for City, Profession, Degree
- **Feature Engineering** — 4 new interaction features:
  - `risk_score` — composite of strongest depression signals
  - `pressure_sleep` — Academic Pressure × Sleep Duration
  - `stress_hours` — Financial Stress × Work/Study Hours
  - `pressure_satisfy` — Academic Pressure × Study Satisfaction

### Train / Test Split
- 80% training / 20% testing
- Stratified split to preserve class balance
- Standard scaling applied for distance-based models (SVM, MLP, Logistic Regression)

---

## 🤖 Models & Results

Six models were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 84.27% | 85.56% | 87.97% | 86.75% |
| Random Forest | 84.14% | 84.87% | 88.74% | 86.76% |
| XGBoost | 84.05% | 85.59% | 87.48% | 86.53% |
| SVM | 84.11% | 84.98% | 88.49% | 86.70% |
| MLP Neural Network | 82.06% | 84.25% | 85.31% | 84.78% |
| **Stacking Ensemble** | **84.20%** | **85.51%** | **87.91%** | **86.69%** |

### Key Finding
All models converged at ~84% accuracy, indicating the performance ceiling lies in **feature expressiveness** rather than model complexity. Logistic Regression matched XGBoost and the Stacking Ensemble — suggesting the depression signals in this dataset are largely **linearly separable**, which supports the use of lightweight, interpretable models for clinical screening tools.

**Recall was prioritised** as the primary metric — in a mental health context, missing a depressed student (false negative) is a more serious error than a false alert (false positive).

---

## 📁 Repository Structure

```
├── app.py                          # Gradio web application
├── requirements.txt                # Python dependencies
├── student_depression_improved.ipynb  # Full training notebook
├── README.md                       # This file
```

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://huggingface.co/spaces/thehmfpk/student-depression-predictor
cd student-depression-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the app
python app.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML Libraries | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Web App | Gradio |
| Deployment | Hugging Face Spaces |
| Notebook | Google Colab |

---

## ⚠️ Ethical Disclaimer

This tool is intended for **research and educational purposes only**. It does not constitute a clinical diagnosis. Depression is a complex medical condition that requires professional evaluation. The risk classification output should be used as a **screening aid**, not a definitive assessment.

If you or someone you know is struggling, please reach out to a qualified mental health professional.

---

## 👩‍💻 Author

**Huma Aslam**
🔗 [Hugging Face Profile](https://huggingface.co/thehmfpk)

---

## 📄 License

This project is licensed under the MIT License.
