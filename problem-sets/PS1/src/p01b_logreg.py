import numpy as np
import util

from linear_model import LinearModel

from numpy import ndarray

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x: ndarray, y: ndarray):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = np.shape(x)
        self.theta = np.zeros(n)
        
        while True:
            score: ndarray = 1 / (1 + np.exp(-np.dot(x, self.theta)))
            # likelihood: int = y * np.log(score) + (1 - y) * np.log(1 - score)
            gradient = -np.dot(x.T, y - score) / m
            hessian = np.dot(x.T * score * (1 - score), x) / m
            
            theta_old = self.theta.copy()
            self.theta -= np.linalg.inv(hessian).dot(gradient)
            if np.linalg.norm(self.theta - theta_old) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x: ndarray):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE ***
