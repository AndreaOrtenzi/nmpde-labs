# Laboratories

## Lab 01

Solve poisson problem 1D and visualize it with paraview

## Lab 02

Convergence analysis of Lab 01.
    Interested in ||u_h - u_{ex} ||
    ||.|| L2 or H1 norm
Visualize what happens if you change the number of elements or the polynomial degree

It's obvious that h goes to zero with order 1 
With polynomial of order 1 we expect the L2 error goes to zero with order 2 and the H1 error should go to zero with order 1
With polynomial of order 2 we expect the L2 error goes to zero with order 3 and the H1 error should go to zero with order 2
With polynomial of order r we expect the L2 error goes to zero with order r+1 and the H1 error should go to zero with order r

- Space dependent forcing term f = 4pi^2 sin(2pi x)
- Exact solution needed to do convergence analysis ( u_{ex} = sin(2pi x) ). Error computed with compute_error method
- To plot the csv results use the pythonn script but you need to install matplotlib first
    - `pip install matplotlib`
    - `../scripts/plot-convergence.py convergence.csv`