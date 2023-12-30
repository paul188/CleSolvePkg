from cle_solve_func import Model
from cle_solve_func import solve_cle

import numpy as np
import matplotlib.pyplot as plt

labels = ['C8','C8a','C3','C3a','CARP','IAP','C8a ~ CARP', 'C3a ~ IAP']
x01 = np.array([130000,0,21000,0,40000,40000,0,0])
x02 = np.array([130000,2000,21000,0,40000,40000,0,0])
x03 = np.array([130000,3000,21000,0,40000,40000,0,0])

def get_propensities(x:np.array,k:np.array):
    #x is the array containing concentrations
    #k is the reaction rate array
    #rates are ordered: k_1 <-> k[0], k_2 <-> 1, k_3 <-> 2, k_{-3} <-> 3, k_{4} <-> 4, k_5 <-> 5, k_6 <-> 6, k_7 <-> 7, k_8 <-> 8, k_{-8} <-> 9
    #k_{9} <-> 10, k_{-9} <-> 11, k_{10} <-> 12, k_{-10} <-> 13, k_{11} <-> 14, k_{-11} <-> 15, k_{12} <-> 16
    return np.array([k[0] * x[2] * x[1],
        k[1] * x[0] * x[3],
        k[2] * x[3] * x[5],
        k[3] * x[7],
        k[4] * x[5] * x[3],
        k[5] * x[1],
        k[6] * x[3],
        k[7] * x[7],
        k[8] * x[5],
        k[9],
        k[10] * x[0],
        k[11],
        k[12] * x[2],
        k[13],
        k[14] * x[1] * x[4],
        k[15] * x[6],
        k[16] * x[4],
        k[17],
        k[18]*x[6]])

# The rates given and used in the last sheet
table_rates = np.array([
    5.8 * (10**(-5)),
    1.0 * (10**(-5)),
    5.0 * (10**(-4)),
    0.21,
    3.0 * (10**(-4)),
    5.8 * (10**(-3)),
    5.8 * (10**(-3)),
    1.73 * (10**(-2)),
    1.16 * (10**(-2)),
    464,
    3.9 * (10**(-3)),
    507,
    3.9 * (10**-3),
    81.9,
    5.0 * (10**(-4)),
    0.21,
    1.0 * (10**(-3)),
    40,
    1.16 * (10**-2)
])

reactants = np.array([[0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                      [1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
                      [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                      [0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0],
                      [0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
                      [0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]])

products  =    np.array([[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                      [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                      [1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
                      [0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                      [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])


model = Model(reactants,products,x01,table_rates,get_propensities)

plt.title("Apoptosis with itoint (1st initial condition)")
plt.xlabel("Time (s)")
plt.ylabel("Molecules (#)")
result_1 = solve_cle(model,np.linspace(0,5000.0,100000),0)
plt.plot(np.linspace(0,5000,100000),result_1[:,0],label="C8",color="b")
plt.plot(np.linspace(0,5000,100000),result_1[:,1],label="C8a",color="orange")
plt.plot(np.linspace(0,5000,100000),result_1[:,2],label="C3",color="yellow")
plt.plot(np.linspace(0,5000,100000),result_1[:,3],label="C3a",color="green")
plt.plot(np.linspace(0,5000,100000),result_1[:,4],label="CARP",color="red")
plt.plot(np.linspace(0,5000,100000),result_1[:,5],label="IAP",color="purple")
plt.plot(np.linspace(0,5000,100000),result_1[:,6],label="C8a ~ CARP",color="pink")
plt.plot(np.linspace(0,5000,100000),result_1[:,7],label="C3a ~ IAP",color="black")
plt.legend()
plt.show()
plt.clf()