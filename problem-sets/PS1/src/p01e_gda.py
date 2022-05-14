import numpy as np
import util

from linear_model import LinearModel

from numpy import ndarray


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    model = GDA()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x: ndarray, y: ndarray):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        
        phi = np.sum(y) / m
        mu_0 = np.sum(np.reshape(1 - y, (m, 1)) * x, axis=0) / np.sum(1 - y)
        mu_1 = np.sum(np.reshape(y, (m, 1)) * x, axis=0) / np.sum(y)
        mu_y = np.dot((1 - y).reshape((m, 1)), mu_0.reshape((1, n))) \
               + np.dot(y.reshape((m, 1)), mu_1.reshape((1, n)))
        sigma = np.dot((x - mu_y).T, x - mu_y) / m
        
        theta = np.linalg.inv(sigma).dot(mu_1 - mu_0)
        theta_0 = (mu_0 + mu_1).T.dot(-theta) / 2 - np.log((1 - phi) / phi)
        
        self.theta = np.concatenate([[theta_0,], theta])
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE
