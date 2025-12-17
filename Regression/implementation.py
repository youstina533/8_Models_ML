import math
from statistics import mean
from collections import defaultdict

def load_csv(insurance_fixed_clean):
    with open(insurance_fixed_clean, 'r') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        raise ValueError("Empty CSV file")
    
    header = lines[0].strip().split(',')
    data = []
    
    for line in lines[1:]:
        if line.strip():  # skip empty lines
            values = [float(v.strip()) for v in line.strip().split(',')]
            data.append(values)
    
    print(f"Loaded {len(data)} rows and {len(header)} columns from {insurance_fixed_clean}")
    return header, data

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])

        # Add bias column (column of 1s)
        X_b = [[1.0] + row for row in X]

        # Compute X^T * X
        XtX = [[0.0 for _ in range(n_features + 1)] for _ in range(n_features + 1)]
        for i in range(n_features + 1):
            for j in range(n_features + 1):
                for k in range(n_samples):
                    XtX[i][j] += X_b[k][i] * X_b[k][j]

        # Compute X^T * y
        Xty = [0.0 for _ in range(n_features + 1)]
        for i in range(n_features + 1):
            for k in range(n_samples):
                Xty[i] += X_b[k][i] * y[k]

        # Solve using Gaussian elimination
        theta = self._gaussian_elimination(XtX, Xty)

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def _gaussian_elimination(self, A, b):
        n = len(A)
        aug = [row[:] + [val] for row, val in zip(A, b)]

        # Forward elimination with partial pivoting
        for i in range(n):
            # Pivoting
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]

            # Eliminate
            pivot = aug[i][i]
            if abs(pivot) < 1e-10:
                pivot = 1e-10  # avoid division by zero
            for k in range(i + 1, n):
                factor = aug[k][i] / pivot
                for j in range(i, n + 1):
                    aug[k][j] -= factor * aug[i][j]

        # Back substitution
        x = [0.0 for _ in range(n)]
        for i in range(n - 1, -1, -1):
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            x[i] /= aug[i][i] if abs(aug[i][i]) > 1e-10 else 1

        return x

    def predict(self, X):
        """Predict target values for X (list of lists)"""
        predictions = []
        for row in X:
            pred = self.intercept_
            for coef, val in zip(self.coef_, row):
                pred += coef * val
            predictions.append(pred)
        return predictions

    def mae(self, y_true, y_pred):
        return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

    def rmse(self, y_true, y_pred):
        mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
        return math.sqrt(mse)

    def r2_score(self, y_true, y_pred):
        ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
        ss_tot = sum((yt - mean(y_true)) ** 2 for yt in y_true)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def cross_validate(self, X, y, k=5, verbose=True):
        """
        Simple k-fold cross-validation.
        Returns list of R², MAE, RMSE for each fold.
        """
        n = len(X)
        fold_size = n // k
        results = []

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size if fold < k - 1 else n

            X_test = X[start:end]
            y_test = y[start:end]
            X_train = X[:start] + X[end:]
            y_train = y[:start] + y[end:]

            # Train on training fold
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)

            # Compute metrics
            r2 = self.r2_score(y_test, y_pred)
            mae_val = self.mae(y_test, y_pred)
            rmse_val = self.rmse(y_test, y_pred)

            fold_result = {
                "fold": fold + 1,
                "R2": r2,
                "MAE": mae_val,
                "RMSE": rmse_val,
                "samples": len(y_test)
            }
            results.append(fold_result)

            if verbose:
                print(f"Fold {fold + 1}: R² = {r2:.4f}, MAE = ${mae_val:.2f}, RMSE = ${rmse_val:.2f}")

        return results
        
if __name__ == "__main__":
    # Load the insurance dataset
    header, data = load_csv(r"D:\BI Sec\BI_project\Regression\insurance_fixed_clean.csv")

    # Identify target column index
    try:
        target_idx = header.index("charges")
    except ValueError:
        print("Column 'charges' not found. Check header:", header)
        exit()

    # Prepare features (X) and target (y)
    X = [row[:target_idx] + row[target_idx + 1:] for row in data]  # all columns except charges
    y = [row[target_idx] for row in data]

    print(f"Features: {len(X[0])} columns")
    print(f"Target column: charges (index {target_idx})")
    print()   

    # Initialize model
    model = LinearRegression()

    # Option 1: Simple train-test split evaluation
    print("=== Train-Test Split Evaluation ===")
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Test R² Score : {model.r2_score(y_test, y_pred):.4f}")
    print(f"Test MAE      : ${model.mae(y_test, y_pred):.2f}")
    print(f"Test RMSE     : ${model.rmse(y_test, y_pred):.2f}")
    print(f"Intercept     : ${model.intercept_:.2f}")
    print()

    # Option 2: 5-Fold Cross-Validation
    print("=== 5-Fold Cross-Validation ===")
    cv_results = model.cross_validate(X, y, k=5)

    avg_r2 = mean(res["R2"] for res in cv_results)
    avg_mae = mean(res["MAE"] for res in cv_results)
    avg_rmse = mean(res["RMSE"] for res in cv_results)

    print("\n--- Cross-Validation Summary ---")
    print(f"Average R²   : {avg_r2:.4f}")
    print(f"Average MAE  : ${avg_mae:.2f}")
    print(f"Average RMSE : ${avg_rmse:.2f}")
    print()

    # Predict on a new sample
    print("=== Prediction on New Sample ===")
    new_sample = [[35, 28.5, 2, 1, 0, 1, 0, 0]]  # age, bmi, children, sex_male, smoker_yes, region_nw, se, sw
    predicted_charge = model.predict(new_sample)[0]
    print(f"Predicted insurance charge: ${predicted_charge:.2f}")
    