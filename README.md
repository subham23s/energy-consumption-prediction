# ⚡ Energy Consumption Prediction using ANN

A machine learning project that predicts the **heating load** of buildings using an Artificial Neural Network (ANN), built with TensorFlow/Keras and deployed as a live interactive dashboard using Streamlit.

---

## 📌 Project Overview

Buildings consume a significant amount of energy for heating and cooling. Predicting energy consumption in advance helps in:
- Reducing electricity costs
- Efficient energy planning
- Building smarter, eco-friendly systems

This project trains an ANN on the **UCI Energy Efficiency Dataset** to predict heating load based on 8 building parameters. Users can interact with a live dashboard and get instant predictions by adjusting input sliders.

---

## 🎯 Features

- ✅ Data preprocessing with MinMax Normalization
- ✅ ANN model with 3 hidden layers + Dropout (anti-overfitting)
- ✅ Early stopping during training
- ✅ Model evaluation with MAE, RMSE, R² Score
- ✅ Training loss curve visualization
- ✅ Actual vs Predicted chart
- ✅ Live interactive Streamlit dashboard
- ✅ Real-time prediction on slider change

---

## 🗂️ Project Structure

```
energy-consumption-prediction/
│
├── ENB2012_data.xlsx       # UCI Energy Efficiency Dataset
├── train_model.py          # Train ANN and save model
├── dashboard.py            # Streamlit live dashboard
├── .gitignore
├── LICENSE
└── README.md
```

> After running `train_model.py`, these files are generated locally:
> - `energy_model.keras` — saved trained model
> - `scaler.pkl` — saved MinMax scaler
> - `loss_curve.png` — training vs validation loss chart
> - `actual_vs_predicted.png` — prediction comparison chart

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Energy Efficiency Dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency)

| Column | Feature | Description |
|--------|---------|-------------|
| X1 | Relative Compactness | Shape of building |
| X2 | Surface Area | Total surface in m² |
| X3 | Wall Area | Wall surface in m² |
| X4 | Roof Area | Roof surface in m² |
| X5 | Overall Height | Height of building (m) |
| X6 | Orientation | Direction building faces |
| X7 | Glazing Area | Window/glass area ratio |
| X8 | Glazing Area Distribution | Window placement |
| **Y1** | **Heating Load** | **Target variable (kWh)** |
| Y2 | Cooling Load | Not used in this project |

- **Rows:** 768
- **Missing values:** None
- **Format:** Excel (.xlsx)

---

## 🧠 ANN Architecture

```
Input Layer     →  8 neurons  (X1 to X8)
Hidden Layer 1  →  64 neurons (ReLU) + Dropout(0.2)
Hidden Layer 2  →  32 neurons (ReLU) + Dropout(0.2)
Hidden Layer 3  →  16 neurons (ReLU)
Output Layer    →  1 neuron   (Predicted Heating Load)
```

- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Epochs:** 200 (with Early Stopping)
- **Batch Size:** 32
- **Train/Test Split:** 80/20

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/subham23s/energy-consumption-prediction.git
cd energy-consumption-prediction
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install required libraries
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn streamlit openpyxl
```

---

## 🚀 How to Run

### Step 1 — Train the model (run once)
```bash
python train_model.py
```
This will:
- Load and preprocess the dataset
- Train the ANN model
- Save `energy_model.keras` and `scaler.pkl`
- Generate loss and prediction charts
- Print MAE, RMSE, R² scores

### Step 2 — Launch the live dashboard
```bash
streamlit run dashboard.py
```
Browser opens at `http://localhost:8501` automatically.

---

## 📈 Model Performance

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination (closer to 1 = better) |

---

## 🖥️ Dashboard Preview

The live dashboard allows users to:
- Adjust 8 building parameter sliders
- Get **instant prediction** of heating load
- View normalized input feature chart
- See training loss and prediction charts

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| TensorFlow / Keras | ANN model |
| Scikit-learn | Preprocessing & metrics |
| Pandas & NumPy | Data handling |
| Matplotlib | Visualization |
| Streamlit | Live dashboard |

---

## 👨‍💻 Author

**Subham** — AI/ML Engineering Student  
GitHub: [@subham23s](https://github.com/subham23s)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
