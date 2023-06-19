# cKAM

Codes implementing converse KAM results (MacKay'18▯) to near-integrable examples of axisymmetric magnetic fields perturbed by helical one or two terms (Kallinikos et. al 2014◧).
Details can be found in https://arxiv.org/abs/2304.09613 .


There are three versions:
1. 'general_1res.py' for the general case presented in MacKay'18, applied to an integrable example of a magnetic field perturbed by a single helical term.
2. 'general_2r.py' for the general case applied to a near integrable example with helical terms as perturbation. 
3. 'symmetrical.py' that corresponds to the R-symmetric result from MacKay'18. In this case, the symmetry considered is the so called Stellarator symmetry, 
   R: (r, \vartheta, φ) -> (r, −\vartheta, −φ), present in the examples considered (with one or two helical modes).


Codes 1 and 2, consider the integration of a regular grid of initial conditions on the YZ-plane for the given magnetic field. Y and Z here correspond to the
symplectic polar coordinates: (\tilde{y},\tilde{z}) = (\sqrt{2ψ/B0} cos\vartheta, \sqrt{2ψ/B0} sin\vartheta). The codes discriminate when the initial does 
not correspond to a flux surface of the same class as the ones present in the unperturbed case (which corresponds to concentric circles around the origin). 
Code 3 does the same identification for initial conditions taken over the symmetric semi-lines θ = 0 and θ = π on the plane X = 0 (i.e. \tilde{z} = 0). 

Notes: 
- A script to plot the results and a text file with the parameters of each code is included. 'general_plot.py' for case 1, 'plot_general_er.py' for 2 and 'sym_plot.py' for 3. 
- The plot coordinates used by these scripts correspond to symplectic polar: (\tilde{y},\tilde{z}) = (\sqrt{2ψ/B0} cos\vartheta, \sqrt{2ψ/B0} sin\vartheta).
- A script to compute the Poincare section of the initial conditions used for case 3 is included, 'sym_Psec.py'. 
- The script 'sym_plot.py' can be used to plot either the orignal output of 3 or the Poincase section from 'sym_Psec.py'.



References

▯ R.S. MacKay. “Finding the complement of the invariant manifolds transverse to a
given foliation for a 3D flow”. Regular and Chaotic Dynamics 23(8), 2018.

◧ N. Kallinikos, H. Isliker, L. Vlahos, E. Meletlidou. “Integrable perturbed magnetic
fields in toroidal geometry: An exact analytical flux surface label for large aspect
ratio”. Physics of Plasmas 21(6), 2014.
