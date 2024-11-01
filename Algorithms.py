import numpy as np



class SingleLayerPerceptron:
    def __init__(self, learning_rate=0.01, epochs=100, add_bias=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.add_bias = add_bias
        self.weights = None
        self.bias = None

    def signum(self, x):
        return  np.where(x >= 0, 1, -1)  

    def fit(self, X, y):  # train
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.random.rand(num_features)
        print(f"The Weigted intialized: {self.weights}")
        self.bias = np.random.rand(1)
        print(f"The bias intialized: {self.bias}")

        for i in range(self.epochs):
            for j in range(num_samples):
                if self.add_bias:
                    netInput = np.dot( self.weights , X[j]) + self.bias
                else:
                    netInput = np.dot( self.weights , X[j])

                y_predicted = self.signum(netInput)

                if y_predicted != y[j]:
                    error = (y[j] - y_predicted)
                    self.weights += self.learning_rate *error * X[j]
                    if self.add_bias: 
                        self.bias += self.learning_rate *error

       # return (self.weights, self.bias)

    def predict(self,x):  # test
        if self.add_bias:
            netInput = np.dot(x,self.weights) + self.bias
        else:
            netInput = np.dot(x, self.weights)

        return self.signum(netInput)  

    def confusion_matrix(self, y_true, y_pred):
        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == -1) & (y_true == -1))
        FP = np.sum((y_pred == 1) & (y_true == -1))
        FN = np.sum((y_pred == -1) & (y_true == 1))
        
        return np.array([[TP, FP],
                        [FN, TN]])   
    





class Adaline:
    def __init__(self, learning_rate=0.01, epochs=100, mse_threshold=3,add_bias=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.add_bias = add_bias
        self.mse_threshold = mse_threshold
        self.weights = None
        self.bias = None
        self.costs = []

    def signum(self, x):
        return  np.where(x >= 0, 1, -1)  
    
    
    def fit(self, X, y):  # train
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.random.rand(num_features)
        print(f"The initialized weights = {self.weights}")
        self.bias = np.random.rand(1)
        print(f"The initialized bias = {self.bias}")

        for i in range(self.epochs):
            epoch_errors = []
            for j in range(num_samples):
                if self.add_bias:
                    netInput = np.dot( self.weights , X[j]) + self.bias
                else:
                    netInput = np.dot( self.weights , X[j])
                
                error = (netInput - y[j])
                epoch_errors.append(error)

                # Update weights and bias for each sample
                self.weights -= self.learning_rate * (X[j] * error)
                self.bias -= self.learning_rate * error

            # Calculate and store cost (MSE)
            mse = (1/(2 * num_samples)) * np.sum(np.square(epoch_errors))
            self.costs.append(mse)

            # Stop if MSE is below the threshold
            if self.mse_threshold is not None and mse < self.mse_threshold:
                print(f"Stopping early at iteration {i + 1} with MSE: {mse}")
                break

    def predict(self,x):  # test
        if self.add_bias:
            netInput = np.dot(x,self.weights) + self.bias
        else:
            netInput = np.dot(x, self.weights)

        return self.signum(netInput)

    def confusion_matrix(self, y_true, y_pred):
        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == -1) & (y_true == -1))
        FP = np.sum((y_pred == 1) & (y_true == -1))
        FN = np.sum((y_pred == -1) & (y_true == 1))
        
        return np.array([[TP, FP],
                        [FN, TN]])        