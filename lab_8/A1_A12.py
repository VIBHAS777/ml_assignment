"""
23CSE301 - Lab Session 08
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from numpy.linalg import pinv

# A1: Utility functions
def summation_unit(inputs, weights):
    x = np.asarray(inputs)
    w = np.asarray(weights)
    return w[0] + np.dot(w[1:], x)

def activation_unit(net, kind='step', bipolar=False, leaky_slope=0.01):
    if kind == 'step':
        return 1 if net >= 0 else 0
    if kind == 'bipolar_step':
        return 1 if net >= 0 else -1
    if kind == 'sigmoid':
        return 1.0/(1.0 + np.exp(-net))
    if kind == 'tanh':
        return np.tanh(net)
    if kind == 'relu':
        return net if net > 0 else 0.0
    if kind == 'leaky_relu':
        return net if net > 0 else leaky_slope*net
    raise ValueError(f"Unknown activation kind: {kind}")

def comparator_unit_error(targets, outputs):
    t = np.asarray(targets)
    y = np.asarray(outputs)
    return 0.5 * np.sum((t - y) ** 2)

# -----------------------------
# A2: Perceptron training
def perceptron_train(X, y, weights_init, lr=0.05, activation='step', bipolar=False,
                     max_epochs=1000, convergence_error=0.002):
    n_samples, n_features = X.shape
    w = np.array(weights_init, dtype=float).copy()
    errors = []

    for epoch in range(1, max_epochs + 1):
        outputs = []
        for i in range(n_samples):
            net = summation_unit(X[i], w)
            out = activation_unit(net, kind=activation, bipolar=bipolar)
            outputs.append(out)

            if activation in ('step', 'bipolar_step', 'relu', 'leaky_relu'):
                delta = lr * (y[i] - out)
                w[0] += delta * 1.0
                w[1:] += delta * X[i]
            elif activation in ('sigmoid', 'tanh'):
                if activation == 'sigmoid':
                    out_cont = activation_unit(net, kind='sigmoid')
                    deriv = out_cont * (1 - out_cont)
                else:
                    out_cont = activation_unit(net, kind='tanh')
                    deriv = 1 - out_cont**2
                delta = lr * (y[i] - out_cont) * deriv
                w[0] += delta
                w[1:] += delta * X[i]
            else:
                delta = lr * (y[i] - out)
                w[0] += delta
                w[1:] += delta * X[i]

        epoch_error = comparator_unit_error(y, outputs)
        errors.append(epoch_error)
        if epoch_error <= convergence_error:
            return w, errors, epoch
    return w, errors, epoch

def perceptron_predict(X, weights, activation='step', bipolar=False):
    preds = []
    for i in range(X.shape[0]):
        net = summation_unit(X[i], weights)
        preds.append(activation_unit(net, kind=activation, bipolar=bipolar))
    return np.array(preds)

def plot_epochs_error(errors, title, filename=None):
    plt.figure()
    plt.plot(range(1, len(errors)+1), errors)
    plt.xlabel('Epoch')
    plt.ylabel('Sum-Squared-Error')
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename)
    plt.close()

# A4: Vary learning rates
def vary_learning_rates_experiment(X, y, weights_init, lr_list, activation='step', bipolar=False,
                                   max_epochs=1000, conv_err=0.002):
    results = {}
    for lr in lr_list:
        _, _, epochs = perceptron_train(X, y, weights_init, lr=lr,
                                        activation=activation, bipolar=bipolar,
                                        max_epochs=max_epochs, convergence_error=conv_err)
        results[lr] = epochs
    return results

# A6: Customer dataset (robust loader)
def prepare_customer_data_from_excel(path):
    df = pd.read_excel(path)
    print("DEBUG: Columns in Excel:", df.columns.tolist())

    # Standardize names
    df.columns = df.columns.str.strip().str.lower().str.replace('[^a-z0-9]', '', regex=True)

    # Try to auto-match
    feature_cols = []
    for col in df.columns:
        if 'cand' in col or 'mango' in col or 'milk' in col or 'pay' in col:
            feature_cols.append(col)

    if len(feature_cols) < 4:
        raise ValueError(f"Could not find enough feature columns. Found: {feature_cols}")

    X = df[feature_cols].values

    # Last column = target
    target_col = df.columns[-1]
    y = df[target_col].apply(lambda x: 1 if str(x).strip().lower() in ('yes','y','true','1') else 0).values
    return X, y, df

# A7: Pseudo-inverse
def pseudo_inverse_solution(X, y):
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    return pinv(Xb).dot(y)

# A8–A9: Two-layer NN (manual)
def sigmoid(x): return 1.0/(1.0 + np.exp(-x))
def train_two_layer_nn(X, y, hidden_neurons=2, lr=0.05, max_epochs=1000, conv_err=0.002):
    rng = np.random.RandomState(1)
    W1 = rng.normal(scale=0.5, size=(hidden_neurons, X.shape[1]+1))
    W2 = rng.normal(scale=0.5, size=(1, hidden_neurons+1))
    errors = []
    for epoch in range(1, max_epochs+1):
        outputs = []
        for i in range(X.shape[0]):
            xi_b = np.concatenate(([1.0], X[i]))
            net_h = W1.dot(xi_b); out_h = sigmoid(net_h)
            out_h_b = np.concatenate(([1.0], out_h))
            net_o = W2.dot(out_h_b); out_o = sigmoid(net_o)[0]
            outputs.append(out_o)
            delta_o = (y[i] - out_o) * out_o * (1 - out_o)
            W2 += lr * delta_o * out_h_b.reshape(1, -1)
            delta_h = (W2[:,1:].T * delta_o).flatten() * out_h * (1 - out_h)
            W1 += lr * np.outer(delta_h, xi_b)
        err = comparator_unit_error(y, outputs)
        errors.append(err)
        if err <= conv_err: return W1, W2, errors, epoch
    return W1, W2, errors, epoch

def predict_two_layer_nn(X, W1, W2):
    preds = []
    for i in range(X.shape[0]):
        xi_b = np.concatenate(([1.0], X[i]))
        out_h = sigmoid(W1.dot(xi_b))
        out_h_b = np.concatenate(([1.0], out_h))
        preds.append(sigmoid(W2.dot(out_h_b))[0])
    return np.array(preds)

# A11–A12: sklearn MLP
def sklearn_mlp_train(X, y, hidden_layer_sizes=(2,), lr=0.05, max_iter=1000):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        activation='logistic',
                        learning_rate_init=lr,
                        max_iter=max_iter,
                        solver='sgd')
    clf.fit(X, y)
    return clf
# Main
if __name__ == '__main__':
    excel_path = 'AirQualityUCI.xlsx'

    # A1 & A2
    X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_and = np.array([0,0,0,1])
    weights_init = np.array([10.0, 0.2, -0.75])
    _, _, ep = perceptron_train(X_and, y_and, weights_init)
    print(f"A1 & A2: AND (step): epochs = {ep}")

    # A3
    _, _, ep_b = perceptron_train(X_and, (y_and*2-1), weights_init, activation='bipolar_step', bipolar=True)
    print(f"A3: AND (bipolar): epochs = {ep_b}")

    _, _, ep_s = perceptron_train(X_and, y_and, weights_init, activation='sigmoid')
    print(f"A3: AND (sigmoid): epochs = {ep_s}")

    _, _, ep_r = perceptron_train(X_and, y_and, weights_init, activation='relu')
    print(f"A3: AND (ReLU): epochs = {ep_r}")

    # A4
    lr_results = vary_learning_rates_experiment(X_and, y_and, weights_init, [0.1*i for i in range(1,11)])
    print("A4: Learning rate vs epochs:", lr_results)

    # A5
    X_xor = X_and.copy(); y_xor = np.array([0,1,1,0])
    _, _, ep_xor = perceptron_train(X_xor, y_xor, weights_init)
    print(f"A5: XOR (step): epochs = {ep_xor} (may not converge)")

    # A6–A7
    try:
        X_cust, y_cust, df_cust = prepare_customer_data_from_excel(excel_path)
        Xc_norm = (X_cust - X_cust.mean(axis=0)) / (X_cust.std(axis=0)+1e-8)
        w_init_cust = np.zeros(Xc_norm.shape[1]+1)
        _, _, ep_cust = perceptron_train(Xc_norm, y_cust, w_init_cust, activation='sigmoid')
        print(f"A6: Customer (sigmoid perceptron): epochs={ep_cust}")

        w_pinv = pseudo_inverse_solution(X_cust, y_cust)
        Xc_b = np.hstack([np.ones((X_cust.shape[0],1)), X_cust])
        preds_pinv = (Xc_b.dot(w_pinv) >= 0.5).astype(int)
        print(f"A7: Pseudo-inverse acc = {accuracy_score(y_cust, preds_pinv):.3f}")
    except Exception as e:
        print("A6–A7: Customer data error:", e)

    # A8
    W1, W2, _, ep_nn = train_two_layer_nn(X_and.astype(float), y_and.astype(float))
    print(f"A8: Two-layer NN (AND): epochs={ep_nn}")

    # A9
    W1x, W2x, _, ep_nn_x = train_two_layer_nn(X_xor.astype(float), y_xor.astype(float))
    print(f"A9: Two-layer NN (XOR): epochs={ep_nn_x}")

    # A11
    clf_and = sklearn_mlp_train(X_and, y_and, hidden_layer_sizes=())
    print("A11: Sklearn MLP AND preds:", clf_and.predict(X_and).tolist())
    clf_xor = sklearn_mlp_train(X_xor, y_xor, hidden_layer_sizes=(2,))
    print("A11: Sklearn MLP XOR preds:", clf_xor.predict(X_xor).tolist())

    # A12
    try:
        if 'X_cust' in locals():
            clf_cust = sklearn_mlp_train(Xc_norm, y_cust, hidden_layer_sizes=(5,))
            print("A12: Sklearn MLP Customer acc:", accuracy_score(y_cust, clf_cust.predict(Xc_norm)))
    except Exception as e:
        print("A12: Customer MLP error:", e)

