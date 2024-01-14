from functools import partial
import numpy as np
import sdeint
import matplotlib.pyplot as plt

class Model:
    """ class to define (bio-)chemical reaction network

        Members:
        products -- The products occuring in the respective reactions
        reactants -- The reactants "..."
        initial_vals -- Initial values of species
        rates -- The rates at which the reactions take place
        get_propensities -- function that returns array of propensities for the reactions
        """
    def __init__(self,reactants,products,initial_vals,rates,get_propensities):
        self.products = products
        self.reactants = reactants
        self.initial_vals = initial_vals
        self.rates = rates
        self.get_propensities = get_propensities

def euler_maruyama(y0,t_span:np.array,f,G,generator:np.random.Generator):
    """ Run the Euler Maruyama method to solve an SDE

        Keyword arguments:
        y0 -- Initial value of the state variable(s)
        t_span -- Timepoints for the Integrator
        f -- deterministic part of the r.h.s. of the SDE
        G -- stochastic part of the "..."
        """
    nb_rows = len(y0)
    nb_columns = len(G(y0,0)[0])
    nb_time_points = len(t_span)
    y = np.empty((nb_time_points,nb_rows))
    y[0] = y0
    for t in range(0,nb_time_points-1):
        t_k = t_span[t]
        t_kplus = t_span[t+1]
        dt = t_kplus - t_k
        dW = np.empty(nb_columns)
        for j in range(0,nb_columns):
            dW[j] = generator.normal(0,np.sqrt(dt))
        y[t+1] = y[t] + f(y[t],t)*dt + G(y[t],t).dot(dW)
    return y

# I have to move the functions f and G outside of solve_cle in order to be able to test them directly.
def f(y,t,model):
        for i in range(0,len(y)):
            if y[i] < 0:
                y[i] = 0
        S = model.products - model.reactants
        propensities = model.get_propensities(y,model.rates)
        ode_rhs = S.dot(propensities)
        return ode_rhs
    
def G(y,t,model):
    num_species = len(model.initial_vals)
    num_reactions = len(model.reactants[0])
    for i in range(0,len(y)):
        if y[i] < 0:
            y[i] = 0
    S = model.products - model.reactants
    propensities = model.get_propensities(y,model.rates)
    sqrt_propensities = np.sqrt(propensities)
    equations = np.empty((num_species,num_reactions))
    for i in range(0,num_species):
        for j in range(0,num_reactions):
            equations[i][j] = S[i][j]*sqrt_propensities[j]
    return equations

def solve_cle(model:Model,t_span:np.array,num_solver:int,generator:np.random.Generator):
    """ Simulate the CLE
         
        Keyword arguments:
        model -- The Model object containing information about the reaction network
        t_span -- np.array containing the timepoints for which CLE should be simulated
        num_solver -- The solver to be used (0<-> Euler Maruyama, 1<-> itoint)
        
        Returns array values of biochemical species at timepoints in t_span
        """
    num_species = len(model.initial_vals)

    f2 = partial(f,model=model)
    G2 = partial(G,model=model)
    
    sol = np.empty((len(t_span),num_species))
    if num_solver == 0:
        sol = euler_maruyama(model.initial_vals,t_span,f2,G2,generator=generator)

    if num_solver == 1:
        sol = sdeint.itoint(f2,G2,model.initial_vals,t_span,generator=generator)

    return sol
    
def plot_cle(model:Model,t_span:np.array,num_solver:int,legend:list,generator:np.random.Generator):
    """ Plot the simulation of the CLE.
        
    Keyword arguments:
    model -- The Model object containing information about the reaction network
    t_span -- np.array containing the timepoints for which CLE should be simulated
    num_solver -- The solver to be used (0<-> Euler Maruyama, 1<-> itoint)
    legend -- A list of labels for the biochemical species 
    """
    sol = solve_cle(model,t_span,num_solver,generator=generator)
    if num_solver == 0:
        plt.title("CLE of the Model Simulated with Euler Maruyama")
    if num_solver == 1:
        plt.title("CLE of the Model Simulated with Itoint")
    for i in range(len(model.initial_vals)):
        plt.plot(t_span,sol[:,i],label=legend[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Molecules (#)")
    plt.legend()
    plt.show()
    plt.clf()