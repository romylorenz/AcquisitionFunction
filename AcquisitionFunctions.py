### PI, EI and GP-UCB acquistion functions as described in [Brochu et al. 2010]

import scipy
import numpy


def PIacquisition(muNew, stdNew, fMax, epsilon):
	"""
	Probability of improvement acquisition function

	INPUT:
		- muNew: mean of predicted point in grid
		- stdNew: sigma (square root of variance) of predicted point in grid
		- fMax: observed or predicted maximum value (depending on noise p.19 [Brochu et al. 2010])
		- epsilon: trade-off parameter (>=0)

	OUTPUT:
		- PI: probability of improvement for candidate point

	As describend in:
		E Brochu, VM Cora, & N de Freitas (2010): 
		A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning,
		arXiv:1012.2599, http://arxiv.org/abs/1012.2599.
	"""

	Z = (muNew - fMax - epsilon)/stdNew

	return scipy.stats.norm.cdf(Z) 


def EIacquisition(muNew, stdNew, fMax, epsilon):
	"""
	Expected improvement acquisition function

	INPUT:
		- muNew: mean of predicted point in grid
		- stdNew: sigma (square root of variance) of predicted point in grid
		- fMax: observed or predicted maximum value (depending on noise p.19 Brochu et al. 2010)
		- epsilon: trade-off parameter (>=0) 
			[Lizotte 2008] suggest setting epsilon = 0.01 (scaled by the signal variance if necessary)  (p.14 [Brochu et al. 2010])		

	OUTPUT:
		- EI: expected improvement for candidate point

	As describend in:
		E Brochu, VM Cora, & N de Freitas (2010): 
		A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning, 
		arXiv:1012.2599, http://arxiv.org/abs/1012.2599.
	"""

	Z = (muNew - fMax - epsilon)/stdNew

	return (muNew - fMax - epsilon)* scipy.stats.norm.cdf(Z) + stdNew*scipy.stats.norm.pdf(Z)


def UCBacquisition(muNew, stdNew, t, d, v=1, delta=.1):
	"""
	Upper confidence bound acquisition function

	INPUT:
		- muNew: predicted mean
		- stdNew: sigma (square root of variance) of predicted point in grid
		- t: number of iteration
		- d: dimension of optimization space
		- v: hyperparameter v = 1*
		- delta: small constant (prob of regret)

		*These bounds hold for reasonably smooth kernel functions.
		[Srinivas et al., 2010]

		OUTPUT:
		- UCB: upper confidence bound for candidate point

	As describend in:
		E Brochu, VM Cora, & N de Freitas (2010): 
		A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning, 
		arXiv:1012.2599, http://arxiv.org/abs/1012.2599.
	"""

	Kappa = numpy.sqrt( v* (2*  numpy.log( (t**(d/2. + 2))*(numpy.pi**2)/(3. * delta)  )))

	return muNew + Kappa * stdNew	
