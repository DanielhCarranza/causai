"""Idea based on https://github.com/robertness/causalML/tree/master/projects/causal%20OOP Implementation  """

import abc
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import pyro
import pyro.distributions as dist

class CausalEngine(metaclass=abc.ABCMeta):
    """
    The CausalEngine class is not one of the entities we model in our domain. It serves as a convenient 
    meta class to capture the probabilities for the random variables in the inheriting class, abstract methods 
    to be implemented by the inheriting class and utility methods shared by the inheriting class. All the domain 
    entities inherit the CausalEngine class.
    """
    
    # Values the random variables can take
    values = {
    }
    
    # Conditional probability tables
    cpts = {}
    
    # Dictionary containing the connections sampled in the current model run
    existing_connections = dict()
    
    @abc.abstractmethod
    def observe(self):
        """
        Abstract method to make sure subclasses implement the observe function which allows the user 
        to set evidence pertaining to a class instance. 
        """
        pass
    
    @abc.abstractmethod
    def infer(self):
        """
        Abstract method to make sure subclasses implement the infer function which allows the user 
        infer a particular value for a class instance.
        """
        pass
    
    @staticmethod
    def evidence(conditions):
        """
        Forms a dictionary to be passed to pyro from a list of conditions formed by calling
        the observe function of each class object with the given observation. 
        
        :param list(dict) conditions: List of dictionaries, each dictionary having key as the 
            trace object related to the class object and its observed value.
        
        :return: Dictionary where the keys are trace objects and values are the observed values. 
        :rtype: dict(str, torch.tesnor)
        """
        
        cond_dict = {}
        for c in conditions:
            cond_dict[list(c.keys())[0]] = list(c.values())[0]
        
        return cond_dict
    
    @staticmethod
    def condition(model, evidence, infer, val, num_samples = 1000):
        """
        Uses pyro condition function with importance sampling to get the conditional probability 
        of a particular value for the random variable under inference. 
        
        :param func model: Probabilistic model defined with pyro sample methods.
        :param dict(str, torch.tensor) evidence: Dictionary with trace objects and their observed values.
        :param str infer: Trace object which needs to be inferred.
        :param int val: Value of the trace object for which the probabilities are required.
        :param int num_samples: Number of samples to run the inference alogrithm.
        
        :return: Probability of trace object being the value provided.
        :rtype: int
        """
        
        conditioned_model = pyro.condition(model, data = evidence)
        posterior = pyro.infer.Importance(conditioned_model, num_samples=num_samples).run()
        marginal = pyro.infer.EmpiricalMarginal(posterior, infer)
        samples = np.array([marginal().item() for _ in range(num_samples)])
        
        return sum([1 for s in samples if s.item() == val])/num_samples
    
    @staticmethod
    def intervention(model, evidence, infer, val, num_samples = 1000):
        """
        Uses pyro condition function with importance sampling to get the intervention probability 
        of a particular value for the random variable under inference.
        
        :param func model: Probabilistic model defined with pyro sample methods.
        :param dict(str, torch.tensor) evidence: Dictionary with trace objects and their observed values.
        :param str infer: Trace object which needs to be inferred.
        :param int val: Value of the trace object for which the probabilities are required.
        :param int num_samples: Number of samples to run the inference alogrithm.
        
        :return: Probability of trace object being the value provided.
        :rtype: int
        """
        
        intervention_model = pyro.do(model, data = evidence)
        posterior = pyro.infer.Importance(intervention_model, num_samples=num_samples).run()
        marginal = pyro.infer.EmpiricalMarginal(posterior, infer)
        samples = np.array([marginal().item() for _ in range(num_samples)])
        
        return sum([1 for s in samples if s.item() == val])/num_samples