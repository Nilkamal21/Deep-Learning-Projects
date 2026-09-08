# 🚀 Deep Learning Projects Portfolio (PyTorch & Data Science)

🔗 **GitHub Repository:** [https://github.com/Nilkamal21/Deep-Learning-Projects.git](https://github.com/Nilkamal21/Deep-Learning-Projects.git)

Welcome to the **Deep Learning Projects Portfolio**! This repository contains end-to-end, beginner-friendly implementations of fundamental deep learning architectures using **PyTorch**, **Scikit-Learn**, and **Kagglehub** datasets.

Each project is self-contained inside its respective directory and structured with detailed markdown explanations, visual diagnostic plots, metrics evaluation, and clean PyTorch code.

---

## 📁 Repository Structure

```text
Deep Learning Projects/
│
├── 📂 ANN/
│   └── 📄 Loan_Default_Prediction_ANN.ipynb      # Tabular Classification using PyTorch ANN
│
├── 📂 CNN/
│   └── 📄 CNN_and_Transfer_Learning_Project.ipynb # Image Classification using Custom PyTorch CNN
│
├── 📂 RNN/
│   └── 📄 RNN_Time_Series_Project.ipynb          # Temperature Forecasting using PyTorch RNN
│
└── 📂 LSTM/
    └── 📄 LSTM_Power_Consumption_Project.ipynb   # Electricity Consumption Forecasting using PyTorch LSTM
```

---

## 📌 Project Summaries & Key Learning Outcomes

### 1️⃣ Artificial Neural Network (ANN) — Loan Default Prediction
* **Directory:** [`ANN/`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/ANN)
* **Notebook:** [`ANN/Loan_Default_Prediction_ANN.ipynb`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/ANN/Loan_Default_Prediction_ANN.ipynb)
* **Problem Type:** Binary Tabular Classification (Predicting whether a borrower will default on a loan).
* **Key Deep Learning Concepts:**
  - **Feature Scaling**: `StandardScaler` normalization of continuous features.
  - **Handling Class Imbalance**: Applying `SMOTE` (Synthetic Minority Over-sampling Technique) to balance default vs. non-default target classes.
  - **Network Architecture**: Fully Connected (Linear/Dense) layers, `nn.ReLU` activation functions, and `nn.Sigmoid` output layer.
  - **Regularization & Optimization**: `nn.BatchNorm1d`, `nn.Dropout`, `nn.BCELoss` (Binary Cross-Entropy Loss), and `torch.optim.Adam`.
  - **Evaluation & Diagnostics**: Train vs. Validation loss curves, Threshold Tuning (optimizing classification cutoffs for precision/recall balance), Confusion Matrix, and F1-score evaluation.

---

### 2️⃣ Convolutional Neural Network (CNN) — Rock-Paper-Scissors Classification
* **Directory:** [`CNN/`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/CNN)
* **Notebook:** [`CNN/CNN_and_Transfer_Learning_Project.ipynb`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/CNN/CNN_and_Transfer_Learning_Project.ipynb)
* **Dataset:** Kaggle Rock-Paper-Scissors Dataset (`sanikamal/rock-paper-scissors-dataset`) via `kagglehub`.
* **Problem Type:** Multi-Class Image Classification (Rock, Paper, Scissors hand gestures).
* **Key Deep Learning Concepts:**
  - **Data Preprocessing & Augmentation**: Image resizing (`64x64`), `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter`, and ImageNet channel normalization.
  - **Custom CNN Architecture**: 3x3 `nn.Conv2d` layers (Filters/Kernels & Feature Map extraction), `nn.ReLU`, `nn.MaxPool2d(2, 2)`, `nn.BatchNorm2d`, and `nn.Dropout(0.4)`.
  - **Early Stopping Mechanism**: Custom PyTorch callback monitoring validation loss, stopping training when loss spikes, and restoring the best model weights.
  - **Overfitting Diagnostics**: Analysis of validation loss spikes, confusion matrix heatmaps, macro Accuracy / Precision / Recall / F1-scores.
  - **Error Analysis**: Visualizing misclassified test set images with `True Label vs. Predicted Label` titles.

---

### 3️⃣ Recurrent Neural Network (RNN) — Daily Climate Forecasting
* **Directory:** [`RNN/`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/RNN)
* **Notebook:** [`RNN/RNN_Time_Series_Project.ipynb`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/RNN/RNN_Time_Series_Project.ipynb)
* **Dataset:** Kaggle Daily Delhi Climate Dataset (`sumanthvrao/daily-climate-time-series-data`) via `kagglehub`.
* **Problem Type:** Time Series Regression (Forecasting mean daily temperature).
* **Key Deep Learning Concepts:**
  - **Sequential Data Preparation**: Scaling values between `0` and `1` using `MinMaxScaler`.
  - **Sliding Window Sequences**: Creating 30-day historical lookback sequence windows ($X_{t-30 \dots t-1} \to y_t$).
  - **RNN Architecture**: `nn.RNN(input_size=1, hidden_size=32, num_layers=1, batch_first=True)` with hidden memory states.
  - **Loss & Optimization**: `nn.MSELoss` (Mean Squared Error) and `torch.optim.Adam`.
  - **Inverse Transformation & Visualization**: Converting scaled forecasts back into °C with `scaler.inverse_transform()` and plotting Actual vs. Predicted Temperature curves.
  - **Regression Metrics**: Calculating MSE, RMSE, and MAE scores.

---

### 4️⃣ Long Short-Term Memory (LSTM) — Household Power Consumption
* **Directory:** [`LSTM/`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/LSTM)
* **Notebook:** [`LSTM/LSTM_Power_Consumption_Project.ipynb`](file:///C:/Users/adhik/OneDrive/Desktop/Deep%20Leaning%20Projects/LSTM/LSTM_Power_Consumption_Project.ipynb)
* **Dataset:** Kaggle Individual Household Electric Power Consumption (`uciml/electric-power-consumption-data-set`) via `kagglehub`.
* **Problem Type:** Time Series Regression (Forecasting daily active power consumption in kW).
* **Key Deep Learning Concepts:**
  - **Data Resampling**: Aggregating minute-by-minute electricity readings into daily average power consumption (`Global_active_power`).
  - **Feature Scaling**: `MinMaxScaler` normalization to stabilize gradient flow inside LSTM memory gates.
  - **LSTM Architecture**: `nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)` leveraging cell state ($c_t$) and hidden state ($h_t$) memory gates (Forget Gate, Input Gate, Output Gate) to solve vanishing gradients.
  - **Loss & Optimization**: `nn.MSELoss` and `torch.optim.Adam(lr=0.001)`.
  - **Forecasting & Evaluation**: Inverse transform back to kilowatts (`kW`), plotting Actual vs. LSTM Predicted Power consumption over time, and computing MSE, RMSE, and MAE.

---

## 🛠️ Required Packages & Dependencies

To run any of the project notebooks, install the following Python packages:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn imbalanced-learn kagglehub nbformat
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
3. Open any project notebook (e.g. `CNN/CNN_and_Transfer_Learning_Project.ipynb`) and run all cells sequentially! Datasets download automatically via `kagglehub`.

---

## 🎯 Architecture Comparison Summary

| Architecture | Primary Use Case | Key Features & Advantages | Loss / Metrics |
| :--- | :--- | :--- | :--- |
| **ANN** | Tabular Classification | Dense layers, ReLU, Batch Normalization, Dropout | BCELoss, F1-Score |
| **CNN** | Image Classification | Spatial filters/kernels, Conv2d, MaxPool2d, Early Stopping | CrossEntropy, Confusion Matrix |
| **RNN** | Short Sequential Data | Hidden state memory ($h_t$) across temporal sequences | MSELoss, RMSE (°C) |
| **LSTM** | Long Time Series Data | Cell state ($c_t$) + Forget, Input, Output gating mechanisms | MSELoss, RMSE (kW) |

---
*Created as part of the Deep Learning & Neural Networks Project Portfolio.* 🎉
