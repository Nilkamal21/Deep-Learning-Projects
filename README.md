# 🚀 Deep Learning Projects Portfolio (PyTorch & TensorFlow / Keras)

🔗 **GitHub Repository:** [https://github.com/Nilkamal21/Deep-Learning-Projects.git](https://github.com/Nilkamal21/Deep-Learning-Projects.git)

Welcome to the **Deep Learning Projects Portfolio**! This repository contains end-to-end, beginner-friendly implementations of fundamental deep learning architectures built side-by-side in both **PyTorch** and **TensorFlow / Keras** using **Scikit-Learn** and **Kagglehub** datasets.

Each project is self-contained inside its respective directory and structured with detailed markdown explanations, visual diagnostic plots, metrics evaluation, and clean framework code.

---

## 📁 Repository Structure

```text
Deep Learning Projects/
│
├── 📂 ANN/
│   ├── 📄 Loan_Default_Prediction_ANN.ipynb           # PyTorch Implementation
│   └── 📄 Loan_Default_Prediction_ANN_TensorFlow.ipynb# TensorFlow / Keras Implementation
│
├── 📂 CNN/
│   ├── 📄 CNN_and_Transfer_Learning_Project.ipynb     # PyTorch Implementation
│   └── 📄 CNN_and_Transfer_Learning_Project_TensorFlow.ipynb # TensorFlow / Keras Implementation
│
├── 📂 RNN/
│   ├── 📄 RNN_Time_Series_Project.ipynb              # PyTorch Implementation
│   └── 📄 RNN_Time_Series_Project_TensorFlow.ipynb   # TensorFlow / Keras Implementation
│
└── 📂 LSTM/
    ├── 📄 LSTM_Power_Consumption_Project.ipynb       # PyTorch Implementation
    └── 📄 LSTM_Power_Consumption_Project_TensorFlow.ipynb # TensorFlow / Keras Implementation
```

---

## 📌 Project Summaries & Framework Implementations

### 1️⃣ Artificial Neural Network (ANN) — Loan Default Prediction
* **Directory:** [`ANN/`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/ANN)
* **Notebooks:**
  - 📄 **PyTorch:** [`ANN/Loan_Default_Prediction_ANN.ipynb`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/ANN/Loan_Default_Prediction_ANN.ipynb)
  - 📄 **TensorFlow / Keras:** [`ANN/Loan_Default_Prediction_ANN_TensorFlow.ipynb`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/ANN/Loan_Default_Prediction_ANN_TensorFlow.ipynb)
* **Problem Type:** Binary Tabular Classification (Predicting whether a borrower will default on a loan).
* **Key Deep Learning Concepts:**
  - **Feature Scaling**: `StandardScaler` normalization of continuous features.
  - **Handling Class Imbalance**: Applying `SMOTE` (Synthetic Minority Over-sampling Technique) to balance default vs. non-default target classes.
  - **PyTorch Stack**: `nn.Module`, `nn.Linear`, `nn.BatchNorm1d`, `nn.Dropout`, `nn.BCELoss`, `optim.Adam`.
  - **TensorFlow / Keras Stack**: `tf.keras.Sequential`, `Dense`, `BatchNormalization`, `Dropout`, `binary_crossentropy`, `Adam`.
  - **Evaluation & Diagnostics**: Train vs. Validation loss curves, Threshold Tuning (optimizing classification cutoffs for precision/recall balance), Confusion Matrix, and F1-score evaluation.

---

### 2️⃣ Convolutional Neural Network (CNN) — Rock-Paper-Scissors Classification
* **Directory:** [`CNN/`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/CNN)
* **Notebooks:**
  - 📄 **PyTorch:** [`CNN/CNN_and_Transfer_Learning_Project.ipynb`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/CNN/CNN_and_Transfer_Learning_Project.ipynb)
  - 📄 **TensorFlow / Keras:** [`CNN/CNN_and_Transfer_Learning_Project_TensorFlow.ipynb`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/CNN/CNN_and_Transfer_Learning_Project_TensorFlow.ipynb)
* **Dataset:** Kaggle Rock-Paper-Scissors Dataset (`sanikamal/rock-paper-scissors-dataset`) via `kagglehub`.
* **Problem Type:** Multi-Class Image Classification (Rock, Paper, Scissors hand gestures).
* **Key Deep Learning Concepts:**
  - **Data Preprocessing & Augmentation**: Image resizing (`64x64`), Random Flips, Rotations, Color Jitter, and Normalization.
  - **PyTorch CNN Architecture**: 3x3 `nn.Conv2d` layers (Filters/Kernels & Feature Maps), `nn.ReLU`, `nn.MaxPool2d(2, 2)`, `nn.BatchNorm2d`, `nn.Dropout(0.4)`.
  - **TensorFlow CNN Architecture**: `tf.keras.layers.Conv2D`, `MaxPooling2D`, `BatchNormalization`, `Dropout(0.4)`, `Dense`.
  - **Early Stopping Mechanism**: Custom PyTorch callback / `tf.keras.callbacks.EarlyStopping` to stop training when validation loss spikes and restore best model weights.
  - **Error Analysis**: Visualizing misclassified test set images with `True Label vs. Predicted Label` titles.

---

### 3️⃣ Recurrent Neural Network (RNN) — Daily Climate Forecasting
* **Directory:** [`RNN/`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/RNN)
* **Notebooks:**
  - 📄 **PyTorch:** [`RNN/RNN_Time_Series_Project.ipynb`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/RNN/RNN_Time_Series_Project.ipynb)
  - 📄 **TensorFlow / Keras:** [`RNN/RNN_Time_Series_Project_TensorFlow.ipynb`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/RNN/RNN_Time_Series_Project_TensorFlow.ipynb)
* **Dataset:** Kaggle Daily Delhi Climate Dataset (`sumanthvrao/daily-climate-time-series-data`) via `kagglehub`.
* **Problem Type:** Time Series Regression (Forecasting mean daily temperature).
* **Key Deep Learning Concepts:**
  - **Sequential Data Preparation**: Scaling values between `0` and `1` using `MinMaxScaler`.
  - **Sliding Window Sequences**: Creating 30-day historical lookback sequence windows ($X_{t-30 \dots t-1} \to y_t$).
  - **PyTorch Stack**: `nn.RNN(input_size=1, hidden_size=32, num_layers=1, batch_first=True)` with hidden memory state ($h_t$).
  - **TensorFlow Stack**: `tf.keras.layers.SimpleRNN(32, input_shape=(30, 1))` with `Dense(1)` output layer.
  - **Inverse Transformation & Visualization**: Converting scaled forecasts back into °C with `scaler.inverse_transform()` and plotting Actual vs. Predicted Temperature curves.
  - **Regression Metrics**: Calculating MSE, RMSE, and MAE scores.

---

### 4️⃣ Long Short-Term Memory (LSTM) — Household Power Consumption
* **Directory:** [`LSTM/`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/LSTM)
* **Notebooks:**
  - 📄 **PyTorch:** [`LSTM/LSTM_Power_Consumption_Project.ipynb`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/LSTM/LSTM_Power_Consumption_Project.ipynb)
  - 📄 **TensorFlow / Keras:** [`LSTM/LSTM_Power_Consumption_Project_TensorFlow.ipynb`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/LSTM/LSTM_Power_Consumption_Project_TensorFlow.ipynb)
* **Dataset:** Kaggle Individual Household Electric Power Consumption (`uciml/electric-power-consumption-data-set`) via `kagglehub`.
* **Problem Type:** Time Series Regression (Forecasting daily active power consumption in kW).
* **Key Deep Learning Concepts:**
  - **Data Resampling**: Aggregating minute-by-minute electricity readings into daily average power consumption (`Global_active_power`).
  - **Feature Scaling**: `MinMaxScaler` normalization to stabilize gradient flow inside LSTM memory gates.
  - **PyTorch Stack**: `nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)` with cell state ($c_t$) and hidden state ($h_t$).
  - **TensorFlow Stack**: `tf.keras.layers.LSTM(64, input_shape=(30, 1))` leveraging cell and hidden states.
  - **Forecasting & Evaluation**: Inverse transform back to kilowatts (`kW`), plotting Actual vs. LSTM Predicted Power consumption over time, and computing MSE, RMSE, and MAE.

---

## 🛠️ Required Packages & Dependencies

To run any of the project notebooks, install the required Python packages:

```bash
pip install torch torchvision tensorflow numpy pandas matplotlib seaborn scikit-learn imbalanced-learn kagglehub nbformat
```

---

## 💡 How to Run the Notebooks

1. Clone the repository:
   ```bash
   git clone https://github.com/Nilkamal21/Deep-Learning-Projects.git
   cd Deep-Learning-Projects
   ```
2. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
3. Open any project notebook in **PyTorch** or **TensorFlow** and run all cells sequentially! Datasets download automatically via `kagglehub`.

---

## 🎯 Framework & Architecture Comparison Summary

| Architecture | Primary Use Case | PyTorch Module | TensorFlow / Keras Layer | Key Features & Advantages | Loss / Metrics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ANN** | Tabular Classification | `nn.Linear` | `tf.keras.layers.Dense` | Dense layers, ReLU, Batch Normalization, Dropout | Binary Crossentropy, F1-Score |
| **CNN** | Image Classification | `nn.Conv2d` | `tf.keras.layers.Conv2D` | Spatial filters/kernels, MaxPool, Early Stopping | CrossEntropy, Confusion Matrix |
| **RNN** | Short Sequential Data | `nn.RNN` | `tf.keras.layers.SimpleRNN` | Hidden state memory ($h_t$) across temporal sequences | MSELoss, RMSE (°C) |
| **LSTM** | Long Time Series Data | `nn.LSTM` | `tf.keras.layers.LSTM` | Cell state ($c_t$) + Forget, Input, Output memory gates | MSELoss, RMSE (kW) |

---
*Created as part of the Deep Learning & Neural Networks Project Portfolio.* 🎉
