# Task7
# Breast Cancer Classification using Support Vector Machine (SVM)

This project implements **Support Vector Machines (SVM)** with both linear and RBF kernels on the Breast Cancer Wisconsin dataset to classify tumors as benign or malignant.  
It includes preprocessing, dimensionality reduction for visualization, hyperparameter tuning, model evaluation, and decision boundary plots.

---

## Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## Steps and Implementation

### 1. Load and Prepare a Dataset for Binary Classification
- Loaded the `breast-cancer.csv` dataset
- Dropped unnecessary columns (`id`, `Unnamed: 32`)
- Encoded the `diagnosis` column:  
  - `M` (Malignant) → 1  
  - `B` (Benign) → 0
- Normalized features using `StandardScaler`
- Reduced dimensionality to 2D using `PCA` for visualization

### 2. Train an SVM with Linear and RBF Kernel
- Trained two models using `SVC`:
  - `kernel='linear'`
  - `kernel='rbf'` with `gamma=0.1`
- Used 80/20 train-test split

### 3. Visualize Decision Boundary Using 2D Data
- Created meshgrid over the 2D PCA-reduced feature space
- Plotted decision boundaries for both linear and RBF models using `matplotlib`

### 4. Tune Hyperparameters Like C and Gamma
- Used `GridSearchCV` with 5-fold cross-validation to find optimal:
  - `C` values: `[0.1, 1, 10]`
  - `gamma` values: `[0.01, 0.1, 1]` (for RBF)
- Printed best parameters from the grid search

### 5. Use Cross-Validation to Evaluate Performance
- Evaluated both models using 5-fold cross-validation
- Reported mean accuracy for:
  - Linear SVM
  - RBF SVM

---

## Example Output

```text
Linear SVM Accuracy: 0.947
RBF SVM Accuracy: 0.956

Cross-validation Accuracy (Linear SVM): 0.952
Cross-validation Accuracy (RBF SVM): 0.960

Best Parameters (RBF SVM): {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
