# EX-KRLS (extended kernel recursive least squares)

The extended kernel recursive least squares (EX-KRLS) is a model proposed by Liu et al. [1].

- [EX-KRLS](https://github.com/kaikerochaalves/EX-KRLS/blob/2761a22ba438595c2d31403c064d05df69d9c548/Model/EX_KRLS.py) is the EX-KRLS model.

- [GridSearch_AllDatasets](https://github.com/kaikerochaalves/EX-KRLS/blob/2761a22ba438595c2d31403c064d05df69d9c548/GridSearch_AllDatasets.py) is the file to perform a grid search for all datasets and store the best hyper-parameters.

- [Runtime_AllDatasets](https://github.com/kaikerochaalves/EX-KRLS/blob/2761a22ba438595c2d31403c064d05df69d9c548/Runtime_AllDatasets.py) perform 30 simulations for each dataset and compute the mean runtime and the standard deviation.

- [MackeyGlass](https://github.com/kaikerochaalves/EX-KRLS/blob/2761a22ba438595c2d31403c064d05df69d9c548/MackeyGlass.py) is the script to prepare the Mackey-Glass time series, perform simulations, compute the results and plot the graphics. 

- [Nonlinear](https://github.com/kaikerochaalves/EX-KRLS/blob/2761a22ba438595c2d31403c064d05df69d9c548/Nonlinear.py) is the script to prepare the nonlinear dynamic system identification time series, perform simulations, compute the results and plot the graphics.

- [LorenzAttractor](https://github.com/kaikerochaalves/EX-KRLS/blob/2761a22ba438595c2d31403c064d05df69d9c548/LorenzAttractor.py) is the script to prepare the Lorenz Attractor time series, perform simulations, compute the results and plot the graphics. 

[1] W. Liu, I. Park, Y. Wang, J. C. Principe, Extended kernel recursive least squares algorithm, IEEE Transactions on Signal Processing 57 (10) (2009) 3801–3814. 
doi:https://doi.org/10.1109/TSP.2009.2022007.
