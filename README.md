# Student Academic Performance Early Warning System (CRISP-MLQ)

## Overview
An automated machine learning system built following the CRISP-MLQ framework.

## 📁 Repository Structure
```
├── student-mat.csv         # Baseline dataset (UCI Student Performance)
├── Predict Student Pass or Fail Early.pynb                 # Full CRISP-MLQ pipeline & embedded Gradio Web App
├── model.pkl               # Serialized winning K-Nearest Neighbors (KNN) model
|
└── README.md               # Project documentation
```

## 📊 Dataset & Selected Features
* **Source:** UCI Machine Learning Repository (`student-mat.csv`)
* **Total Records:** 395 students (0 missing values)
* **Target Variable:** `Pass_Fail` (Binary: 1 if G3 >= 10 else 0)
* **Input Features:**
  1. `G2` (Second Period Grade): 0-20 scale (r = +0.7254)
  2. `G1` (First Period Grade): 0-20 scale (r = +0.6649)
  3. `failures` (Past Class Failures): 0-4 count (r = -0.3377)
  4. `absences` (School Absences): Count of missed days (r = -0.0922)

---

## ⚙️ CRISP-MLQ Workflow & Model Evaluation

1. **Business & Data Understanding:** Formalized early intervention objective and verified dataset integrity (395 records, 0 missing entries).
2. **Feature Selection:** Streamlined feature space by >87% using Pearson correlation to focus on core academic metrics.
3. **Train-Test Split:** Partitioned data into an 80% training set (316 students) and a 20% holdout test set (79 students) with fixed random seed (`random_state=42`).
4. **Model Performance Comparison:**

| Algorithm | Model Architecture | Test Set Accuracy | Selection Status |
| :--- | :--- | :---: | :--- |
| **K-Nearest Neighbors (KNN)** | Instance-Based Classifier | **92.41%** | **Winning Model (Selected)** |
| **Logistic Regression** | Linear Classification | **89.87%** | Benchmark Candidate |
| **Decision Tree Classifier** | Non-Linear Decision Tree | **89.87%** | Benchmark Candidate |

5. **Serialization & Deployment:** Saved winning KNN model binary as `model.pkl` via `pickle` and launched an interactive Gradio web application interface.

---

## 💻 Code Example & Prediction Function

```python
import pickle
import gradio as gr

# Load trained KNN classifier
model = pickle.load(open("model.pkl", "rb"))

def predict_student_status(G1, G2, failures, absences):
    # Model inference
    prediction = model.predict([[G1, G2, failures, absences]])[0]
    return "On Track (Pass)" if prediction == 1 else "Needs Support (At-Risk)"

# Launch Gradio Web Interface
interface = gr.Interface(
    fn=predict_student_status,
    inputs=[
        gr.Number(label="G1 Period 1 Grade (0-20)"),
        gr.Number(label="G2 Period 2 Grade (0-20)"),
        gr.Number(label="Past Failures (0-4)"),
        gr.Number(label="School Absences")
    ],
    outputs=gr.Textbox(label="Status Prediction"),
    title="Student Early Intervention Predictor",
    description="Early warning ML system following the CRISP-MLQ framework."
)

if __name__ == "__main__":
    interface.launch(share=True)
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn gradio
```

### 2. Running the Notebook
```bash
Predict Student Pass or Fail Early.pynb
```
Executing this file processes `student-mat.csv`, trains and evaluates candidate models, serializes `model.pkl`, and starts the live Gradio interface.
