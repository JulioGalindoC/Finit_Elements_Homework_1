from numpy import array, pi, zeros, ix_, sqrt
from beam_column_element import beam_element
from scipy.linalg import solve

xy = array([
    [0,0],      # nodo 0
    [0,3],      # nodo 1
    [0,5],      # nodo 2
    [0,6],      # nodo 3
    [6,5.5],    # nodo 4
    [6,5],      # nodo 5
    [6,3],      # nodo 6
    [6,0]       # nodo 7
    ])

conec = array([
    [0,1],      # elemento 0
    [1,2],      # elemento 1
    [2,3],      # elemento 2
    [3,4],      # elemento 3
    [4,5],      # elemento 4
    [5,6],      # elemento 5
    [6,7],      # elemento 6
    [2,5],      # elemento 7
    [1,6]       # elemento 8
    ],
    dtype=int
    )

densidad_H = 2500   #kg/m^3
g = 9.8             #m/s^2

# Roof beam.
w_roof_beam = 20e-2
h_roof_beam = 20e-2

# beams.
w_beams = 20e-2
h_beams = 40e-2

# columns
w_columns = 30e-2
h_columns = 30e-2

properties_roof_beam = {}
properties_beams = {}
properties_columns = {}

properties_roof_beam["E"] = 21e9
properties_roof_beam["A"] = w_roof_beam * h_roof_beam
properties_roof_beam["I"] = w_roof_beam * h_roof_beam**3 / 12
properties_roof_beam["qx"] = densidad_H * g * properties_roof_beam["A"] * 6/sqrt(36.25)
properties_roof_beam["qy"] = densidad_H * g * properties_roof_beam["A"] * 0.5/sqrt(36.25)

properties_beams["E"] = 21e9
properties_beams["A"] = w_beams * h_beams
properties_beams["I"] = w_beams * h_beams**3 /12
properties_beams["qx"] = 0
properties_beams["qy"] = densidad_H * g * properties_beams["A"]


properties_columns["E"] = 21e9
properties_columns["A"] = w_columns * h_columns
properties_columns["I"] = w_columns * h_columns**3 / 12
properties_columns["qx"] = densidad_H * g * properties_columns["A"]
properties_columns["qy"] = 0

elements = [properties_columns, properties_columns, properties_columns,         # left columns
            properties_roof_beam,                                               # beam sup.
            properties_columns, properties_columns, properties_columns,         # right columns
            properties_beams,properties_beams]                                  # beam mid.

Nnodes = xy.shape[0]
Nelems = conec.shape[0]

NDOFs_per_node = 3
NDOFs = 3*Nnodes

K = zeros((NDOFs, NDOFs))
f = zeros((NDOFs, 1))

for e in range(Nelems) :
    ni = conec[e,0]
    nj = conec[e,1]

    print(f"e = {e} ni = {ni} nj = {nj}")

    xy_e = xy[[ni, nj], :]

    #print(f"xy_e = {xy_e}")

    ke, fe = beam_element(xy_e, elements[e])

    #print(f"ke = {ke}")

    d = [3*ni, 3*ni+1, 3*ni+2, 3*nj, 3*nj+1, 3*nj+2 ] # global DOFs from local dofs

    #Direct stiffnes method
    for i in range(2*NDOFs_per_node) :
        p = d[i]
        for j in range(2*NDOFs_per_node) :
            q = d[j]
            K[p, q] += ke[i,j]
        f[p] += fe[i]

#print(K)
print(f"f = {f}")

# System partitioning and solution

free_DOFs = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
constrain_DOFs = [0,1,2,21,22,23] 

Kff = K[ix_(free_DOFs, free_DOFs)]
Kfc = K[ix_(free_DOFs, constrain_DOFs)]
Kcf = K[ix_(constrain_DOFs, free_DOFs)]
Kcc = K[ix_(constrain_DOFs, constrain_DOFs)]

ff = f[free_DOFs]
fc = f[constrain_DOFs]

# Solve:
u = zeros((NDOFs,1))

u[free_DOFs] = solve(Kff, ff)

# Get reaction forces:

R = Kcf @ u[free_DOFs] + Kcc @ u[constrain_DOFs] -fc

print(f"u = {u}")
print(f"R = {R}")
