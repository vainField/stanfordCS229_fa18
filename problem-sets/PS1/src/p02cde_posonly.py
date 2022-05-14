import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** TODO: START CODE HERE ***
    x_valid, t_valid = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    _, y_valid = util.load_dataset(valid_path, label_col='y')
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    _, y_train = util.load_dataset(train_path, label_col='y')
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    model = LogisticRegression()
    model.fit(x_train, t_train)
    t_pred = model.predict(x_valid)
    print('(c): Prediction accuracy is', np.sum((t_pred > 0.5) == t_valid) / t_valid.shape[0])
    util.plot(x_test, t_test, model.theta)
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    print('(d): Prediction accuracy is', np.sum((y_pred > 0.5) == t_valid) / t_valid.shape[0])
    util.plot(x_test, t_test, model.theta)
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    alpha = np.sum(y_pred[y_valid == 1]) / np.sum(y_valid)
    y_pred = y_pred / alpha
    print('(e): Prediction accuracy is', np.sum((y_pred > 0.5) == t_valid) / t_valid.shape[0])
    util.plot(x_test, t_test, model.theta, correction=alpha)
    # *** END CODER HERE
