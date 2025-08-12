import numpy as np
from Elements_Nodes import Elements_Nodes
from Vec_Elements_Bars import Vec_Elements_Bars
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N
from Vec_Elements_CST import Vec_Elements_CST

from Assembly_KirigamiTruss import Assembly_KirigamiTruss
from Plot_KirigamiTruss import Plot_KirigamiTruss
from Solver_NR_Folding_4N import Solver_NR_Folding_4N

# Timing
import time
start = time.time()

# Define Geometry
L = 1
w = 0.1
gap = 0
N = 2

node = Elements_Nodes()

node.coordinates_mat = np.array([
    [-w, 0, 0],
    [-w, L, 0],
    [-w, 0, L],
    [-w, L, L]
])

for i in range(1, N+1):
    base = (w + L) * (i - 1)
    node.coordinates_mat = np.vstack([
        node.coordinates_mat,
        [base, 0, 0], [base, L, 0], [base, 0, L], [base, L, L],
        [base + L/2, 0, 0], [base + L/2, 0, gap],
        [base + L/2, L, 0], [base + L/2, L, gap],
        [base + L/2, 0, L], [base + L/2, gap, L],
        [base + L/2, L, L], [base + L/2, L, L - gap],
        [base + L/2, L/2, 0], [base + L/2, L/2, L],
        [base + L/2, 0, L/2], [base + L/2, L, L/2],
        [base + L, 0, 0], [base + L, L, 0], [base + L, 0, L], [base + L, L, L]
    ])

node.coordinates_mat = np.vstack([
    node.coordinates_mat,
    [(w + L)*N, 0, 0], [(w + L)*N, L, 0],
    [(w + L)*N, 0, L], [(w + L)*N, L, L]
])

# Define assembly
assembly = Assembly_KirigamiTruss()
assembly.node = node

cst = Vec_Elements_CST()
bar = Vec_Elements_Bars()
rotSpr = Vec_Elements_RotSprings_4N()

assembly.cst = cst
assembly.bar = bar
assembly.rotSpr = rotSpr


# Define Triangle Elements
for i in range(N+1):
    if i==0:
        cst.node_ijk_mat=np.array([
            [0, 2, 6], 
            [0, 4, 6], 
            [2, 3, 7], 
            [2, 6, 7],
            [1, 3, 5], 
            [3, 5, 7], 
            [2, 3, 7], 
            [0, 1, 5],
            [0, 4, 5]
        ])
    else:
        idx = 20*i
        cst.node_ijk_mat=np.vstack([cst.node_ijk_mat,
            [idx+0, idx+2, idx+6], 
            [idx+0, idx+4, idx+6], 
            [idx+2, idx+3, idx+7], 
            [idx+2, idx+6, idx+7],
            [idx+1, idx+3, idx+5], 
            [idx+3, idx+5, idx+7], 
            [idx+2, idx+3, idx+7], 
            [idx+0, idx+1, idx+5],
            [idx+0, idx+4, idx+5]
        ])
    
for i in range(N):    
    idx = 20*i
    cst.node_ijk_mat=np.vstack([cst.node_ijk_mat,
        [idx+4, idx+5, idx+16], 
        [idx+4, idx+8, idx+16], 
        [idx+10, idx+5, idx+16],
        [idx+8, idx+20, idx+16], 
        [idx+10, idx+21, idx+16], 
        [idx+20, idx+21, idx+16]
    ])
    
cstNum = len(cst.node_ijk_mat)
cst.t_vec = 0.05 * np.ones(cstNum)
cst.E_vec = 2.0e9 * np.ones(cstNum)
cst.v_vec = 0.2 * np.ones(cstNum)



# Define Bars
for i in range(N):
    if i==0:
        idx=0
        bar.node_ij_mat=np.array([
            [idx+5-1, idx+10-1], 
            [idx+19-1, idx+10-1], 
            [idx+5-1, idx+19-1], 
            [idx+7-1, idx+13-1], 
            [idx+7-1, idx+19-1], 
            [idx+13-1, idx+19-1],
            [idx+13-1, idx+23-1], 
            [idx+19-1, idx+23-1], 
            [idx+19-1, idx+21-1], 
            [idx+10-1, idx+21-1],             
            [idx+7-1, idx+14-1], 
            [idx+7-1, idx+18-1],
            [idx+14-1, idx+18-1], 
            [idx+8-1, idx+15-1], 
            [idx+15-1, idx+18-1], 
            [idx+8-1, idx+18-1], 
            [idx+14-1, idx+23-1], 
            [idx+18-1, idx+23-1],
            [idx+18-1, idx+24-1], 
            [idx+15-1, idx+24-1], 
            [idx+8-1, idx+16-1], 
            [idx+16-1, idx+20-1], 
            [idx+8-1, idx+20-1], 
            [idx+24-1, idx+16-1],
            [idx+24-1, idx+20-1], 
            [idx+6-1, idx+20-1], 
            [idx+6-1, idx+12-1], 
            [idx+12-1, idx+20-1], 
            [idx+22-1, idx+12-1], 
            [idx+22-1, idx+20-1],
            [idx+5-1, idx+9-1], 
            [idx+9-1, idx+17-1], 
            [idx+5-1, idx+17-1], 
            [idx+6-1, idx+11-1], 
            [idx+11-1, idx+17-1], 
            [idx+6-1, idx+17-1],
            [idx+11-1, idx+22-1], 
            [idx+17-1, idx+22-1], 
            [idx+9-1, idx+21-1], 
            [idx+17-1, idx+21-1]
        ])
    else:        
        idx = 20*i
        bar.node_ij_mat=np.vstack([bar.node_ij_mat,
            [idx+5-1, idx+10-1], 
            [idx+19-1, idx+10-1], 
            [idx+5-1, idx+19-1], 
            [idx+7-1, idx+13-1], 
            [idx+7-1, idx+19-1], 
            [idx+13-1, idx+19-1],
            [idx+13-1, idx+23-1], 
            [idx+19-1, idx+23-1], 
            [idx+19-1, idx+21-1], 
            [idx+10-1, idx+21-1], 
            [idx+7-1, idx+14-1], 
            [idx+7-1, idx+18-1],
            [idx+14-1, idx+18-1], 
            [idx+8-1, idx+15-1], 
            [idx+15-1, idx+18-1], 
            [idx+8-1, idx+18-1], 
            [idx+14-1, idx+23-1], 
            [idx+18-1, idx+23-1],
            [idx+18-1, idx+24-1], 
            [idx+15-1, idx+24-1], 
            [idx+8-1, idx+16-1], 
            [idx+16-1, idx+20-1], 
            [idx+8-1, idx+20-1], 
            [idx+24-1, idx+16-1],
            [idx+24-1, idx+20-1], 
            [idx+6-1, idx+20-1], 
            [idx+6-1, idx+12-1], 
            [idx+12-1, idx+20-1], 
            [idx+22-1, idx+12-1], 
            [idx+22-1, idx+20-1],
            [idx+5-1, idx+9-1], 
            [idx+9-1, idx+17-1], 
            [idx+5-1, idx+17-1], 
            [idx+6-1, idx+11-1], 
            [idx+11-1, idx+17-1], 
            [idx+6-1, idx+17-1],
            [idx+11-1, idx+22-1], 
            [idx+17-1, idx+22-1], 
            [idx+9-1, idx+21-1], 
            [idx+17-1, idx+21-1]
        ])
        
        
        
barNum = len(bar.node_ij_mat)
bar.A_vec = 0.01 * np.ones(barNum)
bar.E_vec = 2.0e9 * np.ones(barNum)


# Define Rotational Springs
for i in range(N):
    if i==0:
        rotSpr.node_ijkl_mat=np.array([
        [20*(i)+5-1,  20*(i)+1-1,  20*(i)+7-1,  20*(i)+2-1],
        [20*(i)+1-1,  20*(i)+7-1,  20*(i)+3-1,  20*(i)+8-1],
        [20*(i)+7-1,  20*(i)+3-1,  20*(i)+8-1,  20*(i)+4-1],
        [20*(i)+3-1,  20*(i)+4-1,  20*(i)+8-1,  20*(i)+6-1],
        [20*(i)+2-1,  20*(i)+4-1,  20*(i)+6-1,  20*(i)+8-1],
        [20*(i)+4-1,  20*(i)+2-1,  20*(i)+6-1,  20*(i)+1-1],
        [20*(i)+2-1,  20*(i)+6-1,  20*(i)+1-1,  20*(i)+5-1],
        [20*(i)+6-1,  20*(i)+1-1,  20*(i)+5-1,  20*(i)+7-1],
        [20*(i)+1-1,  20*(i)+7-1,  20*(i)+5-1,  20*(i)+19-1],
        [20*(i)+5-1,  20*(i)+7-1,  20*(i)+19-1,  20*(i)+13-1],
        [20*(i)+7-1,  20*(i)+13-1,  20*(i)+19-1,  20*(i)+23-1],
        [20*(i)+19-1, 20*(i)+23-1, 20*(i)+21-1, 20*(i)+27-1],
        [20*(i)+5-1,  20*(i)+10-1, 20*(i)+19-1, 20*(i)+21-1],
        [20*(i)+7-1,  20*(i)+5-1,  20*(i)+19-1, 20*(i)+10-1],
        [20*(i)+13-1, 20*(i)+19-1, 20*(i)+23-1, 20*(i)+21-1],
        [20*(i)+10-1, 20*(i)+19-1, 20*(i)+21-1, 20*(i)+23-1],
        [20*(i)+3-1,  20*(i)+8-1,  20*(i)+7-1,  20*(i)+18-1],
        [20*(i)+8-1,  20*(i)+7-1,  20*(i)+18-1, 20*(i)+14-1],
        [20*(i)+7-1,  20*(i)+8-1,  20*(i)+18-1, 20*(i)+15-1],
        [20*(i)+8-1,  20*(i)+15-1, 20*(i)+18-1, 20*(i)+24-1],
        [20*(i)+7-1,  20*(i)+14-1, 20*(i)+18-1, 20*(i)+23-1],
        [20*(i)+14-1, 20*(i)+18-1, 20*(i)+23-1, 20*(i)+24-1],
        [20*(i)+15-1, 20*(i)+18-1, 20*(i)+24-1, 20*(i)+23-1],
        [20*(i)+18-1, 20*(i)+23-1, 20*(i)+24-1, 20*(i)+28-1],
        [20*(i)+4-1,  20*(i)+6-1,  20*(i)+8-1,  20*(i)+20-1],
        [20*(i)+6-1,  20*(i)+8-1,  20*(i)+20-1, 20*(i)+16-1],
        [20*(i)+8-1,  20*(i)+6-1,  20*(i)+20-1, 20*(i)+12-1],
        [20*(i)+8-1,  20*(i)+16-1, 20*(i)+20-1, 20*(i)+24-1],
        [20*(i)+6-1,  20*(i)+12-1, 20*(i)+20-1, 20*(i)+22-1],
        [20*(i)+24-1, 20*(i)+20-1, 20*(i)+22-1, 20*(i)+12-1],
        [20*(i)+16-1, 20*(i)+20-1, 20*(i)+24-1, 20*(i)+22-1],
        [20*(i)+26-1, 20*(i)+24-1, 20*(i)+22-1, 20*(i)+20-1],
        [20*(i)+1-1,  20*(i)+5-1,  20*(i)+6-1,  20*(i)+17-1],
        [20*(i)+6-1,  20*(i)+5-1,  20*(i)+17-1, 20*(i)+9-1],
        [20*(i)+5-1,  20*(i)+6-1,  20*(i)+17-1, 20*(i)+11-1],
        [20*(i)+5-1,  20*(i)+9-1,  20*(i)+17-1, 20*(i)+21-1],
        [20*(i)+6-1,  20*(i)+17-1, 20*(i)+11-1, 20*(i)+22-1],
        [20*(i)+11-1, 20*(i)+17-1, 20*(i)+22-1, 20*(i)+21-1],
        [20*(i)+9-1,  20*(i)+17-1, 20*(i)+21-1, 20*(i)+22-1],
        [20*(i)+17-1, 20*(i)+21-1, 20*(i)+22-1, 20*(i)+26-1]
        ]) 
    else:
        rotSpr.node_ijkl_mat=np.vstack([rotSpr.node_ijkl_mat,
                [20*(i)+5-1,  20*(i)+1-1,  20*(i)+7-1,  20*(i)+2-1],
                [20*(i)+1-1,  20*(i)+7-1,  20*(i)+3-1,  20*(i)+8-1],
                [20*(i)+7-1,  20*(i)+3-1,  20*(i)+8-1,  20*(i)+4-1],
                [20*(i)+3-1,  20*(i)+4-1,  20*(i)+8-1,  20*(i)+6-1],
                [20*(i)+2-1,  20*(i)+4-1,  20*(i)+6-1,  20*(i)+8-1],
                [20*(i)+4-1,  20*(i)+2-1,  20*(i)+6-1,  20*(i)+1-1],
                [20*(i)+2-1,  20*(i)+6-1,  20*(i)+1-1,  20*(i)+5-1],
                [20*(i)+6-1,  20*(i)+1-1,  20*(i)+5-1,  20*(i)+7-1],
                [20*(i)+1-1,  20*(i)+7-1,  20*(i)+5-1,  20*(i)+19-1],
                [20*(i)+5-1,  20*(i)+7-1,  20*(i)+19-1,  20*(i)+13-1],
                [20*(i)+7-1,  20*(i)+13-1,  20*(i)+19-1,  20*(i)+23-1],
                [20*(i)+19-1, 20*(i)+23-1, 20*(i)+21-1, 20*(i)+27-1],
                [20*(i)+5-1,  20*(i)+10-1, 20*(i)+19-1, 20*(i)+21-1],
                [20*(i)+7-1,  20*(i)+5-1,  20*(i)+19-1, 20*(i)+10-1],
                [20*(i)+13-1, 20*(i)+19-1, 20*(i)+23-1, 20*(i)+21-1],
                [20*(i)+10-1, 20*(i)+19-1, 20*(i)+21-1, 20*(i)+23-1],
                [20*(i)+3-1,  20*(i)+8-1,  20*(i)+7-1,  20*(i)+18-1],
                [20*(i)+8-1,  20*(i)+7-1,  20*(i)+18-1, 20*(i)+14-1],
                [20*(i)+7-1,  20*(i)+8-1,  20*(i)+18-1, 20*(i)+15-1],
                [20*(i)+8-1,  20*(i)+15-1, 20*(i)+18-1, 20*(i)+24-1],
                [20*(i)+7-1,  20*(i)+14-1, 20*(i)+18-1, 20*(i)+23-1],
                [20*(i)+14-1, 20*(i)+18-1, 20*(i)+23-1, 20*(i)+24-1],
                [20*(i)+15-1, 20*(i)+18-1, 20*(i)+24-1, 20*(i)+23-1],
                [20*(i)+18-1, 20*(i)+23-1, 20*(i)+24-1, 20*(i)+28-1],
                [20*(i)+4-1,  20*(i)+6-1,  20*(i)+8-1,  20*(i)+20-1],
                [20*(i)+6-1,  20*(i)+8-1,  20*(i)+20-1, 20*(i)+16-1],
                [20*(i)+8-1,  20*(i)+6-1,  20*(i)+20-1, 20*(i)+12-1],
                [20*(i)+8-1,  20*(i)+16-1, 20*(i)+20-1, 20*(i)+24-1],
                [20*(i)+6-1,  20*(i)+12-1, 20*(i)+20-1, 20*(i)+22-1],
                [20*(i)+24-1, 20*(i)+20-1, 20*(i)+22-1, 20*(i)+12-1],
                [20*(i)+16-1, 20*(i)+20-1, 20*(i)+24-1, 20*(i)+22-1],
                [20*(i)+26-1, 20*(i)+24-1, 20*(i)+22-1, 20*(i)+20-1],
                [20*(i)+1-1,  20*(i)+5-1,  20*(i)+6-1,  20*(i)+17-1],
                [20*(i)+6-1,  20*(i)+5-1,  20*(i)+17-1, 20*(i)+9-1],
                [20*(i)+5-1,  20*(i)+6-1,  20*(i)+17-1, 20*(i)+11-1],
                [20*(i)+5-1,  20*(i)+9-1,  20*(i)+17-1, 20*(i)+21-1],
                [20*(i)+6-1,  20*(i)+17-1, 20*(i)+11-1, 20*(i)+22-1],
                [20*(i)+11-1, 20*(i)+17-1, 20*(i)+22-1, 20*(i)+21-1],
                [20*(i)+9-1,  20*(i)+17-1, 20*(i)+21-1, 20*(i)+22-1],
                [20*(i)+17-1, 20*(i)+21-1, 20*(i)+22-1, 20*(i)+26-1]
        ])
        
# Add final rotSpr set for end cap
rotSpr.node_ijkl_mat=np.vstack([rotSpr.node_ijkl_mat,
                [20*N+5-1,  20*N+1-1,  20*N+7-1,  20*N+2-1],
                [20*N+1-1,  20*N+7-1,  20*N+3-1,  20*N+8-1],
                [20*N+7-1,  20*N+3-1,  20*N+8-1,  20*N+4-1],
                [20*N+3-1,  20*N+4-1,  20*N+8-1,  20*N+6-1],
                [20*N+2-1,  20*N+4-1,  20*N+6-1,  20*N+8-1],
                [20*N+4-1,  20*N+2-1,  20*N+6-1,  20*N+1-1],
                [20*N+2-1,  20*N+6-1,  20*N+1-1,  20*N+5-1],
                [20*N+6-1,  20*N+1-1,  20*N+5-1,  20*N+7-1]])
    
rotNum = len(rotSpr.node_ijkl_mat)
rotSpr.rot_spr_K_vec = 1.0 * np.ones(rotNum)


factor = 100
for i in range(N+1):
    for offset in [1, 3, 5, 7]:
        rotSpr.rot_spr_K_vec[i*40 + offset] *= factor        
        
# Initialize Assembly
assembly.Initialize_Assembly()                
        
# Define Plotting
plots = Plot_KirigamiTruss()
plots.assembly = assembly

plots.display_range = np.array([-1, L*(N+1), -1, 2, -1, 2])

plots.Plot_Shape_Node_Number()
plots.Plot_Shape_Cst_Number()
plots.Plot_Shape_Bar_Number()        
plots.Plot_Shape_Spr_Number()

Fstart,Kstart=assembly.Solve_FK(np.zeros([48,3]))
Fb,Kb=bar.Solve_FK(node,np.zeros([48,3]))
Fcst,Kcst=cst.Solve_FK(node,np.zeros([48,3]))
Frs,Krs=rotSpr.Solve_FK(node,np.zeros([48,3]))

# Solver Setup
sf = Solver_NR_Folding_4N()
sf.assembly = assembly
sf.supp = [[0,1,1,1],[1,1,1,1],[2,1,1,1],[3,1,1,1]]
sf.targetRot = rotSpr.theta_stress_free_vec[:]
sf.increStep = 100
sf.iterMax = 20
sf.tol = 1e-4
rate = 0.1
for i in range(N):
    sf.targetRot[i*40+10-1]=np.pi-rate*np.pi
    sf.targetRot[i*40+11-1]=np.pi+rate*np.pi
    sf.targetRot[i*40+13-1]=np.pi+rate*np.pi
    sf.targetRot[i*40+14-1]=np.pi-rate*np.pi
    sf.targetRot[i*40+15-1]=np.pi-rate*np.pi
    sf.targetRot[i*40+16-1]=np.pi-rate*np.pi
    
    sf.targetRot[i*40+18-1]=np.pi+rate*np.pi
    sf.targetRot[i*40+19-1]=np.pi+rate*np.pi
    sf.targetRot[i*40+20-1]=np.pi-rate*np.pi
    sf.targetRot[i*40+21-1]=np.pi-rate*np.pi
    sf.targetRot[i*40+22-1]=np.pi+rate*np.pi
    sf.targetRot[i*40+23-1]=np.pi+rate*np.pi
    
    sf.targetRot[i*40+26-1]=np.pi+rate*np.pi
    sf.targetRot[i*40+27-1]=np.pi+rate*np.pi
    sf.targetRot[i*40+28-1]=np.pi-rate*np.pi
    sf.targetRot[i*40+29-1]=np.pi-rate*np.pi
    sf.targetRot[i*40+30-1]=np.pi+rate*np.pi
    sf.targetRot[i*40+31-1]=np.pi+rate*np.pi
    
    sf.targetRot[i*40+34-1]=np.pi-rate*np.pi
    sf.targetRot[i*40+35-1]=np.pi-rate*np.pi
    sf.targetRot[i*40+36-1]=np.pi+rate*np.pi
    sf.targetRot[i*40+37-1]=np.pi+rate*np.pi
    sf.targetRot[i*40+38-1]=np.pi-rate*np.pi
    sf.targetRot[i*40+39-1]=np.pi-rate*np.pi
    

Uhis = sf.Solve()
end = time.time()
print("Execution Time:", end - start)
plots.Plot_Deformed_Shape(Uhis[-1])
# plots.fileName = 'OrigamiTruss_deploy.gif'
# plots.plot_deformed_history(Uhis[::10])
