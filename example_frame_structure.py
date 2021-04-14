from numpy import array, pi, zeros, ix_
from beam_column_element import beam_element
from scipy.linalg import solve

xy = array([
    [0,0],
    [4,3],
    [9,3],
    [9,0]
    ])

conec = array([
    [0,1],
    [1,2],
    [2,3]
    ],
    dtype=int
    )

t = 20e-3
r = 400e-3

properties_0 = {}
properties_1 = {}
properties_2 = {}

properties_0["E"] = 200e9
properties_0["A"] = pi*(r**2 - (r-t)**2)
properties_0["I"] = pi*(r**4/4 - (r-t)**4/4)
properties_0["qx"] = 0
properties_0["qy"] = 0

properties_1["E"] = 200e9
properties_1["A"] = pi*(r**2 - (r-t)**2)
properties_1["I"] = pi*(r**4/4 - (r-t)**4/4)
properties_1["qx"] = 0
properties_1["qy"] = 5000


properties_2["E"] = 200e9
properties_2["A"] = pi*(r**2 - (r-t)**2)
properties_2["I"] = pi*(r**4/4 - (r-t)**4/4)
properties_2["qx"] = 0
properties_2["qy"] = 0

elements = [properties_0, properties_1, properties_2]

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

free_DOFs = [3,4,5,6,7,8,9,11]
constrain_DOFs = [0,1,2,10] 

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

print(f"Kff = {Kff}")
print(f"R = {R}")
