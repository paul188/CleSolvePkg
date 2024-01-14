import pytest
import numpy as np
import math

from cle_solve_func import solve_cle
from cle_solve_func import Model
from cle_solve_func import euler_maruyama
from cle_solve_func import f
from cle_solve_func import G

# Test f and G defined in the package
def test_f():
    # test f in the simple mRNA degradation model from assignment 1
    rates = np.array([0.5,0.1])
    reactants = np.array([0,1])
    products = np.array([1,0])
    def get_propensities_mRNA_deg(y,rates):
        return np.array([rates[0],rates[1]*y[0]])
    
    degradation_model = Model(reactants,products,np.array([0]),rates,get_propensities_mRNA_deg)
    assert f(np.array([0]),0.0,degradation_model) == degradation_model.rates[0]
    assert f(np.array([1]),0.0,degradation_model) == degradation_model.rates[0]-degradation_model.rates[1]
    assert f(np.array([2]),0.0,degradation_model) == degradation_model.rates[0]-2*degradation_model.rates[1]

def test_G():
    # test G in the mRNA degradation model:
    rates = np.array([0.5,0.1])
    reactants = np.array([[0,1]])
    products = np.array([[1,0]])
    def get_propensities_mRNA_deg(y,rates):
        return np.array([rates[0],rates[1]*y[0]])
    
    degradation_model = Model(reactants,products,np.array([0]),rates,get_propensities_mRNA_deg)
    assert np.array_equal(G(np.array([0]),0.0,degradation_model),np.array([[math.sqrt(0.5),-0.0]]))
    assert np.array_equal(G(np.array([1]),0.0,degradation_model),np.array([[math.sqrt(rates[0]),-math.sqrt(rates[1])]]))
    assert np.array_equal(G(np.array([2]),0.0,degradation_model),np.array([[math.sqrt(rates[0]),-math.sqrt(rates[1]*2)]]))

def test_solve_cle():
    # Test CLE with both integrators for degradation reaction
    # Degradation reaction with high rate over extremely long time 
    # certain to degrade once.
    y0 = np.array([1])
    reactants = np.array([[1]])
    products = np.array([[0]])
    rates = np.array([0.5])

    def get_propensities(y,rates):
        return np.array([y[0]*rates[0]])

    model = Model(reactants,products,y0,rates,get_propensities)
    sol = solve_cle(model,np.linspace(0,1000000,100),0)
    assert sol[-1] == np.array([0])

    sol2 = solve_cle(model,np.linspace(0,1000000,100),1)
    assert sol[-1] == np.array([0])

def test_euler_maruyama():
    # Test the Euler Maruyama integrator
    # First, test that deterministic ODE integrated correctly for f(x) = x
    # (here integration should be exactly correct independent of stepsize)
    def f(t,y):
        return np.array([1])
    
    def G(t,y):
        return np.array([[0]])
    
    sol = euler_maruyama(np.array([0]),np.linspace(0,100,2),f,G)
    assert sol[-1][0] == 100

def test_euler_maruyama_2():
    # Test the stochastic part of Euler Maruyama:
    # Average over many Wiener processes should be small 
    def f(t,y):
        return np.array([0])
    
    def G(t,y):
        return np.array([[1]])
    
    sols = np.empty(100000)
    for i in range(0,100000):
        sols[i] = euler_maruyama(np.array([0]),np.linspace(0,100,2),f,G)[-1]
    
    assert abs(np.average(sols)) <= 10
