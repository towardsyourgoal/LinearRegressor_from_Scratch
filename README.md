# 📈 Linear Regression from Scratch (Python)

This project implements **Linear Regression from scratch** using pure Python and NumPy, without relying on machine learning libraries like `scikit-learn`.
The goal is to understand the **math, optimization, and training process** behind linear regression.

---

## 🚀 Features

* Implements **simple & multivariate linear regression**
* Uses **Gradient Descent** for optimization
* Custom implementation of:

  * Cost function (Mean Squared Error)
  * Parameter updates
* Clear, readable, and beginner-friendly code
* Comparison-ready with `sklearn` results

---

## 🧠 Concepts Covered

* Hypothesis function
* Mean Squared Error (MSE)
* Gradient Descent
* Learning Rate
* Convergence behavior

---

## 📂 Project Structure

```
LinearRegressor_from_Scratch/
│
├── LinearRegressor_Custom.py   # Core implementation
├── data/
│   └── sample_data.csv       # Sample dataset
├── notebooks/
│   └── demo.ipynb            # Step-by-step explanation
├── README.md
└── requirements.txt
```

---

## 📐 Mathematical Formulation

**Hypothesis**

```
ŷ = Xθ + b
```

**Cost Function (MSE)**

```
J(θ) = (1 / 2m) Σ (ŷ − y)²
```

**Gradient Descent Update**

```
θ = θ − α · ∂J/∂θ
b = b − α · ∂J/∂b
```

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/LinearRegressor_from_Scratch.git
cd LinearRegressor_from_Scratch
pip install -r requirements.txt
```

