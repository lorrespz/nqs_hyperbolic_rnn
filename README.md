## Hyperbolic recurrent neural network as the first type of non-Euclidean neural quantum state ansatz 
This is the GitHub repo for the work arXiv: 2505.22083. 

In this work, we introduce hyperbolic GRU as the first type of non-Euclidean neural quantum state (NQS) for quantum many-body systems in condensed matter physics. 
We investigate the viability and performance of hyperbolic GRU NQS ansatz in 4 prototypical settings for many-body quantum physics: one-dimensional & two-dimensional transverse field Ising models, one-dimensional Heisenberg $J_1J_2$ and $J_1J_2J_3$ models. In each of these 4 settings, hyperbolic GRU is benchmarked against different variants of Euclidean RNNs (either RNN or GRU), as well as against the exact results provided by DMRG (Density Matrix Renormalization Group) method. The results of running the Variational Monte Carlo (VMC) experiments using hyperbolic GRU as well as Euclidean RNNs NQS ansatzes for each of the 4 settings are shown graphically below. 

## 1D TFIM
- In the 1D TFIM setting, with the number of spins N being 20, 40, 80 and 100, where the Hamiltonian comprises only the nearest neighbor interaction and a transverse magnetic field, with no hierarchical interaction structure, three out of four times, hyperbolic GRU does not outperform either Euclidean RNN or GRU.
<img src="https://github.com/lorrespz/nqs_hyperbolic_rnn/blob/main/figs/1d_tfim_comparison.png"  style="width:60%; height:auto;">

## 2D TFIM
- In the 2D TFIM setting, with the number of size of the 2D square lattice $(N_x, N_y)$ being (5, 5), (7, 7), (8, 8), (9, 9) corresponding to N = 25, 49, 64, 81 spins, where the Hamitonian comprises the horizontal and vertical nearest neighbor interaction in two dimensions and a transverse magnetic field, 2D Euclidean RNN always emerge as the best ansatz, followed by 1D hyperbolic GRU, which almost always outperforms 1D Euclidean GRU, except for a single instance in the training result for $(N_x,N_y)$ = (8,8). In this case, due to the artificial rearrangement or reshaping/folding of the 1D spin chain to mimic the 2D lattice, a hierarchy in the interaction structure, comprising the first and the Nth nearest neighbor interactions, is introduced when originally faraway spins (at site i and site $i + N$) in the 1D chains become immediate vertical neighbors. The fact that the hyperbolic GRU NQS ansatz outperformed the Euclidean GRU NQS ansatz for this setting is our first clue of the role played by the hyperbolic geometry of the neural network in determining its performance when there is a hierarchical structure in the Hamiltonian.
  
<img src="https://github.com/lorrespz/nqs_hyperbolic_rnn/blob/main/figs/2d_tfim_comparison.png"  style="width:60%; height:auto;">

## 1D Heisenberg J1J2 
- In the 1D Heisenberg J1J2 setting with N = 50 spins, $J_1 = 1.0$ (fixed) and $J_2$ = 0.0, 0.2, 0.5, 0.8, hyperbolic GRU definitively outperforms Euclidean GRU three out of four times when the J1J2 Hamiltonian exhibits a hierarchical interaction structure when $J_2\neq 0$ which takes into account the second nearest neighbor interactions in addition to the first nearest neighbor interactions. This is our second clue of the role played by the hyperbolic geometry of the neural network.
  
<img src="https://github.com/lorrespz/nqs_hyperbolic_rnn/blob/main/figs/1d_j1j2_comparison.png"  style="width:60%; height:auto;">

## 1D Heisenberg J1J2J3
- In the 1D Heisenberg J1J2J3 setting with N = 30 spins, J1 = 1.0 (fixed) and (J2, J3) = (0.0, 0.5), (0.2, 0.2), (0.2,0.5), (0.5,0.2), hyperbolic GRU definitively outperforms Euclidean GRU four of four times. In this setting, the J1J2J3 Hamiltonian exhibits even more pronounced hierarchical interaction structures comprising the first, second and third nearest neighbors.
  
<img src="https://github.com/lorrespz/nqs_hyperbolic_rnn/blob/main/figs/1d_j1j2j3_comparison.png"  style="width:60%; height:auto;">
