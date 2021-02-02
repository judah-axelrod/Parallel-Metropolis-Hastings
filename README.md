# Parallel-Metropolis-Hastings
This repository contains an implementation of the Metropolis-Hastings algorithm parallelized using the Python Multiprocessing module. Provided in **MH-Parallel.py** is an example of the Metropolis-Hastings algorithm in a Bayesian context, where the user can generate samples for the posterior distributions of multiple parameters, in this case the mean and standard deviation of a normal distribution. A few additional notes:
- The user can choose to run the algorithm either non-parallelized or parallelized and compare the runtimes of each implementation to see that the parallel version will execute faster in most cases.
- The user can tweak any of the parameters, distributional assumptions (such as the prior, the simulated dataset, etc.), number of samples, and burn-in time. The code is fairly flexible and can be generalized to other contexts besides the one presented here. 
- The user can specify a different number of cores (e.g. 2, 4, 8) to use for parallelization depending on how many processors their computer has. The current default is 4.
- The user can visualize the output in three ways:
  1. Trace plots of the generated samples for the mean and standard deviation
  2. An acceptance/rejection plot to see which proposal points were accepted into the Markov chain
  3. Histograms of the estimated posterior distributions

Finally, for those who want further background and a more complete discussion on parallel computing in Python or Markov chain Monte Carlo (MCMC) methods such as the Metropolis Hastings algorithm, please refer to the included full report **MH-Parallel-Report.pdf**.

This project was completed in collaboration with Anil Bhattarai, Priyanshi Gupta, and Yufeng Fu, as part of LSE ST444, a graduate-level Computational Data Science course in LSE's Department of Statistics. Thanks very much to my fellow group members for all of their hard work in putting this together.
