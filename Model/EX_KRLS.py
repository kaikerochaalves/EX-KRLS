# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np

class EX_KRLS:
    def __init__(self, alpha = 0.999, beta = 0.995, lambda1 = 1E-2, q = 1E-3, M = 100, kernel_width = 1):
        #self.hyperparameters = pd.DataFrame({})
        self.parameters = pd.DataFrame(columns = ['alpha', 'rho', 'Q', 'Dict'])
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
        # Hyperparameters and parameters
        self.alpha = alpha # State forgetting factor
        self.beta = beta # Data forgetting factor
        self.lambda1 = lambda1 # Regularization
        self.q = q # Trade-off between modeling variation and measurement disturbance
        self.M = M # Maximum dictionary size
        self.kernel_width = 1 # Kernel width
        self.i = 0 # Iteration number;

    def fit(self, X, y):

        # Compute the number of samples
        n = X.shape[0]
        
        # Initialize the first input-output pair
        x0 = X[0,].reshape(-1,1)
        y0 = y[0]
        
        # Initialize the consequent parameters
        self.Initialize_EX_KRLS(x0, y0)

        for k in range(1, n):

            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
                      
            # Update the consequent parameters
            kn1 = self.EX_KRLS(x, y[k])
            
            # Compute the output
            Output = self.parameters.loc[0, 'alpha'].T @ kn1
            
            # Store the results
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output )
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(y[k]) - Output )
        return self.OutputTrainingPhase
            
    def predict(self, X):

        for k in range(X.shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T

            # Compute k
            kn1 = np.array(())
            for ni in range(self.parameters.loc[0, 'Dict'].shape[1]):
                kn1 = np.append(kn1, [self.Kernel(self.parameters.loc[0, 'Dict'][:,ni].reshape(-1,1), x)])
            kn1 = kn1.reshape(-1,1)
            # Compute the output
            Output = self.parameters.loc[0, 'alpha'].T @ kn1
            # Storing the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output )

        return self.OutputTestPhase

    def Kernel(self, x1, x2):
        k = np.exp( - ( 1/2 ) * ( (np.linalg.norm( x1 - x2 ))**2 ) / ( self.kernel_width**2 ) )
        return k
    
    def Initialize_EX_KRLS(self, x, y):
        self.i += 1
        ktt = self.Kernel(x, x)
        alpha = np.ones((1,1)) * self.alpha * y / ( self.lambda1 * self.beta + ktt)
        rho = self.lambda1 * self.beta / ( self.alpha**2 * self.beta + self.lambda1 * self.q )
        Q = np.ones((1,1)) * self.alpha**2 / ( ( self.beta * self.lambda1 + ktt ) * ( self.alpha**2 + self.beta * self.lambda1 * self.q ) )
        NewRow = pd.DataFrame([[alpha, rho, Q, x]], columns = ['alpha', 'rho', 'Q', 'Dict'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
        # Initialize first output and residual
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def EX_KRLS(self, x, y):
        i = 0              
        if self.parameters.loc[i, 'Dict'].shape[1] < self.M:
            self.i += 1
            # Update dictionary
            self.parameters.at[i,  'Dict'] = np.hstack([self.parameters.loc[i,  'Dict'], x])
            # Compute k
            k = np.array(())
            for ni in range(self.parameters.loc[i,  'Dict'].shape[1]):
                k = np.append(k, [self.Kernel(self.parameters.loc[i,  'Dict'][:,ni].reshape(-1,1), x)])
            kt = k[:-1].reshape(-1,1)
            ktt = self.Kernel(x, x)
            # Compute z
            z = self.parameters.loc[i,  'Q'] @ kt
            # Compute r
            r = self.beta**self.i * self.parameters.loc[i,  'rho']  + ktt - kt.T @ z
            # Estimate the error
            err = y - kt.T @ self.parameters.loc[i,  'alpha']
            # Update alpha
            self.parameters.at[i,  'alpha'] = self.alpha * ( self.parameters.loc[i,  'alpha'] - z * err / r )
            self.parameters.at[i,  'alpha'] = np.vstack([self.parameters.loc[i,  'alpha'], self.alpha * err / r]) # Update the size of alpha
            # Parcel to update Q
            dummy = self.alpha**2 + self.beta**self.i * self.q * self.parameters.loc[i,  'rho'] 
            self.parameters.at[i,  'rho'] = self.parameters.loc[i,  'rho'] / dummy
            # Update Q
            self.parameters.at[i,  'Q'] = self.parameters.loc[i,  'Q'] * r + z @ z.T
            self.parameters.at[i,  'Q'] = np.lib.pad(self.parameters.loc[i,  'Q'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeQ = self.parameters.loc[i,  'Q'].shape[0] - 1
            self.parameters.at[i, 'Q'][sizeQ,sizeQ] = 1.
            self.parameters.at[i, 'Q'][0:sizeQ,sizeQ] = -z.flatten()
            self.parameters.at[i, 'Q'][sizeQ,0:sizeQ] = -z.flatten()
            self.parameters.at[i,  'Q'] = self.alpha**2 / ( r * dummy ) * self.parameters.loc[i,  'Q']
        else:
            # Compute k
            k = np.array(())
            for ni in range(self.parameters.loc[i,  'Dict'].shape[1]):
                k = np.append(k, [self.Kernel(self.parameters.loc[i,  'Dict'][:,ni].reshape(-1,1), x)])
 
        return k.reshape(-1,1)