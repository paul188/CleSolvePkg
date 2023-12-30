import numpy as np
import sdeint

class Model:
    def __init__(self,reactants,products,initial_vals,rates,get_propensities):
        self.products = products
        self.reactants = reactants
        self.initial_vals = initial_vals
        self.rates = rates
        self.get_propensities = get_propensities

def euler_maruyama(y0,t_span:np.array,f,G):
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
            dW[j] = np.random.normal(0,np.sqrt(dt))
        y[t+1] = y[t] + f(y[t],t)*dt + G(y[t],t).dot(dW)
    return y

def solve_cle(model:Model,t_span:np.array,num_solver:int):
    # model is an instance of class Model, the model to be simulated
    # t_span timepoints to be used in simulation
    # num_solver describes solver to be used: 
    # 0 <-> Euler-Maruyama, 1 <-> sdeint.itoint
    num_species = len(model.initial_vals)
    num_reactions = len(model.reactants[0])
    def f(y,t):
        for i in range(0,len(y)):
            if y[i] < 0:
                y[i] = 0
        S = model.products - model.reactants
        propensities = model.get_propensities(y,model.rates)
        ode_rhs = S.dot(propensities)
        return ode_rhs
    
    def G(y,t):
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
    
    sol = np.empty((len(t_span),num_species))
    if num_solver == 0:
        sol = euler_maruyama(model.initial_vals,t_span,f,G)

    if num_solver == 1:
        sol = sdeint.itoint(f,G,model.initial_vals,t_span)

    return sol