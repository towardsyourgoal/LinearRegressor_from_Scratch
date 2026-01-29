import numpy as np

class LinearRegressor:
    """
    A custom implementation of Linear Regression using Gradient Descent
    1. Initialize weights and bias
    2. Compute cost function
    3. Compute gradients
    4. Fit the model using gradient descent
    5. Make predictions
    6. Evaluate model performance using MSE, RMSE, R-squared, Adjusted R-squared, and MBE
    """
    def __init__(self, learning_rate=0.01, iterations=10000):
        self.learning_rate=learning_rate
        self.iterations=iterations
        self.weights=None
        self.bias=None
        self.j_history=[]
        pass

    def compute_cost(self, X, y, w, b, m):
        """we are calculating the cost or error for the given w and b 
        we will try to minimize this cost in our fit function for better accuracy
        """
        cost = 0
        for i in range(m):
            f_wb_i = np.dot(X[i], w) + b
            error_i = f_wb_i - y[i]
            cost += error_i ** 2
        cost /= (2 * m)
        return cost

    def compute_gradient(self, X, y, w, b, m, n):
        """
        we will compute the gradient of cost function with respect to w and b
        to update w and b using gradient descent
        """
        dj_dw = np.zeros((n,))
        dj_db = 0
        for i in range(m):
            f_wb_i = np.dot(X[i], w) + b
            error_i = f_wb_i - y[i]
            for j in range(n):
                dj_dw[j] += error_i * X[i][j]
            dj_db += error_i
        dj_dw /= m
        dj_db /= m
        return dj_dw, dj_db


    def fit(self, X, y):
        """
        This function will train the linear regression model using gradient descent
        and update the weights and bias accordingly
        we are also storing the cost history for analysis
        """
        X_np=X.to_numpy()
        y_np=y.to_numpy().reshape(-1,1)
        assert X_np.shape[0]==y_np.shape[0], "Number of samples in X and y must be the same"
        m=len(X_np)
        n=X_np.shape[1]
        self.weights=np.zeros((n,))
        self.bias=0
        for i in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(X_np, y_np, self.weights, self.bias, m,n)
            self.j_history.append(self.compute_cost(X_np, y_np, self.weights, self.bias, m))
            self.weights -= self.learning_rate*dj_dw
            self.bias -= self.learning_rate*dj_db


    def predict(self, X):
        """
        This function will make predictions using the trained weights and bias
        """
        X_np=X.to_numpy()
        #we wil use the final w and b to make predictions
        m=len(X_np)
        y_pred=np.zeros((m,))
        for i in range(m):
            y_pred[i]=np.dot(X_np[i], self.weights)+self.bias
        return y_pred

    def mse(self, y_true, y_pred):
        """
        This function will calculate Mean Squared Error between true and predicted values
        """
        assert len(y_true)==len(y_pred), "Length of true values and predicted values must be the same"
        y_true_np=y_true.to_numpy().reshape(-1,1)
        m=len(y_true_np)
        mse=0
        for i in range(m):
            mse+=(y_pred[i]-y_true_np[i])**2
        mse/=m
        return mse
    
    def rmse(self, y_true, y_pred):
        """
        This function will calculate Root Mean Squared Error between true and predicted values
        """
        mse=self.mse(y_true, y_pred)
        rmse=np.sqrt(mse)
        return rmse


    def r_squared(self, X, y_true, y_pred):
        """
        This function will calculate coefficient of determination, which represents proportion of variance explained by the model
        """
        X_np=X.to_numpy()
        y_true_np=y_true.to_numpy().reshape(-1,1)
        m=len(X_np)
        n=X_np.shape[1]
        p=n+1
        y_mean=np.mean(y_true_np)
        rss=0
        tss=0
        for i in range(m):
            rss+=(y_true_np[i]-y_pred[i])**2
            tss+=(y_true_np[i]-y_mean)**2
        r_squared = 1 - (rss/tss)
        adj_r_squared = 1 - (((1-(r_squared)**2)*(m-1))/(m-p-1))
        return r_squared, adj_r_squared

    def mbe(self, X, y_true, y_pred):
        m=len(X)
        mbe=0
        for i in range(m):
            mbe+=(y_pred[i]-y_true[i])
        mbe/=m
        return mbe
