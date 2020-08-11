# Do not use anything outside of the standard distribution of python
# when implementing this class
import math 

class LogisticRegressionSGD:
    """
    Logistic regression with stochastic gradient descent
    """

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.eta = eta
        self.weight = [0.0] * n_feature
        self.mu = mu

    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """
        s = 0
        #calculate w_(t-1)*x_t
        for tup in X:
            s+= tup[1]*self.weight[tup[0]]

        for tup in X:
            self.weight[tup[0]] += self.eta*tup[1]*(y-(1/(1+math.exp(-s))))
        

        for i in range(len(self.weight)):
            self.weight[i] += -2*self.eta*self.mu*self.weight[i]

    def predict(self, X):
        """
        Predict 0 or 1 given X and the current weights in the model
        """
        return 1 if predict_prob(X) > 0.5 else 0

    def predict_prob(self, X):
        """
        Sigmoid function
        """
        return 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))
