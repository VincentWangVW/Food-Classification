import numpy as np
import pandas as pd

def set_seed(seed=42):
    np.random.seed(seed)

def tanh(Z):
    return np.tanh(Z)

def tanh_derivative(A):
    return 1 - A**2

def softmax(Z):
    exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def categorical_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

class MLPClassifierMulticlass:
    """
    MLP with one hidden layer + early stopping on a validation set.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=120,
        alpha=0.0227940227,
        learning_rate_init=0.0106098289,
        max_iter=500,
        random_state=42,
        n_iter_no_change=10,
        tol=1e-4
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.learning_rate = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

        set_seed(self.random_state)
        self._initialize_weights()

    def _initialize_weights(self):
        limit_W1 = np.sqrt(6.0 / (self.input_dim + self.hidden_dim))
        self.W1 = np.random.uniform(-limit_W1, limit_W1, (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim))

        limit_W2 = np.sqrt(6.0 / (self.hidden_dim + self.output_dim))
        self.W2 = np.random.uniform(-limit_W2, limit_W2, (self.hidden_dim, self.output_dim))
        self.b2 = np.zeros((1, self.output_dim))

    def _forward(self, X):
        Z1 = X.dot(self.W1) + self.b1
        A1 = tanh(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = softmax(Z2)
        return A1, A2

    def _compute_loss(self, X, y):
        y_one_hot = one_hot_encode(y, self.output_dim)
        _, A2 = self._forward(X)
        ce = categorical_cross_entropy(y_one_hot, A2)
        l2 = 0.5 * self.alpha * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return ce + l2

    def fit(self, X_train, y_train, X_val, y_val):
        n_samples = X_train.shape[0]
        y_train_one_hot = one_hot_encode(y_train, self.output_dim)

        best_val_loss = float("inf")
        best_weights = None
        epochs_no_improve = 0

        for epoch in range(self.max_iter):
            # Forward on train
            Z1 = X_train.dot(self.W1) + self.b1
            A1 = tanh(Z1)
            Z2 = A1.dot(self.W2) + self.b2
            A2 = softmax(Z2)

            train_loss = categorical_cross_entropy(y_train_one_hot, A2)
            train_loss += 0.5 * self.alpha * (np.sum(self.W1**2) + np.sum(self.W2**2))

            # Backprop
            dZ2 = (A2 - y_train_one_hot)
            dW2 = A1.T.dot(dZ2) / n_samples
            db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples
            dW2 += self.alpha * self.W2

            dA1 = dZ2.dot(self.W2.T)
            dZ1 = dA1 * tanh_derivative(A1)
            dW1 = X_train.T.dot(dZ1) / n_samples
            db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples
            dW1 += self.alpha * self.W1

            # Update
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            # Validation loss
            val_loss = self._compute_loss(X_val, y_val)

            if val_loss < best_val_loss - self.tol:
                best_val_loss = val_loss
                best_weights = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.max_iter}: "
                      f"Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

            if epochs_no_improve >= self.n_iter_no_change:
                print(f"Early stopping at epoch={epoch+1}, Val Loss={val_loss:.4f}")
                break

        # Restore best
        if best_weights:
            self.W1, self.b1, self.W2, self.b2 = best_weights

    def predict_proba(self, X):
        _, A2 = self._forward(X)
        return A2

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

def main():
    """
    1) Force the categories: Pizza -> 0, Sushi -> 1, Shawarma -> 2
    2) Train on train.csv, val on validation.csv, final check on test.csv
    3) Save the best model
    """
    forced_categories = ["Pizza", "Sushi", "Shawarma"]

    # Read train
    df_train = pd.read_csv("train1.csv")
    # Force label ordering
    df_train["Label"] = pd.Categorical(
        df_train["Label"],
        categories=forced_categories,
        ordered=True
    )
    X_train = df_train.drop(columns=["Label"]).values
    y_train = df_train["Label"].cat.codes.values  

    # Read validation
    df_val = pd.read_csv("validation1.csv")
    df_val["Label"] = pd.Categorical(
        df_val["Label"],
        categories=forced_categories,
        ordered=True
    )
    X_val = df_val.drop(columns=["Label"]).values
    y_val = df_val["Label"].cat.codes.values

    # Read test
    df_test = pd.read_csv("test1.csv")
    df_test["Label"] = pd.Categorical(
        df_test["Label"],
        categories=forced_categories,
        ordered=True
    )
    X_test = df_test.drop(columns=["Label"]).values
    y_test = df_test["Label"].cat.codes.values

    # Basic checks
    input_dim = X_train.shape[1]
    if X_val.shape[1] != input_dim or X_test.shape[1] != input_dim:
        raise ValueError("Train/Val/Test feature dimension mismatch.")
    output_dim = len(forced_categories)

    # Create MLP
    model = MLPClassifierMulticlass(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=120,
        alpha=0.0227940227,
        learning_rate_init=0.0106098289,
        max_iter=500,
        random_state=42,
        n_iter_no_change=10,
        tol=1e-4
    )

    # Fit
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate on test
    y_test_pred = model.predict(X_test)
    test_accuracy = np.mean(y_test_pred == y_test)
    print("Final Test Accuracy:", test_accuracy)

    # Save
    np.savez(
        "my_model.npz",
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2,
        input_dim=model.input_dim,
        output_dim=model.output_dim,
        hidden_dim=model.hidden_dim,
        alpha=model.alpha,
        learning_rate_init=model.learning_rate,
        max_iter=model.max_iter,
        random_state=model.random_state,
        n_iter_no_change=model.n_iter_no_change,
        tol=model.tol,
        categories=np.array(forced_categories, dtype=object)
    )
    print("Saved best model to my_model.npz")

if __name__ == "__main__":
    main()
