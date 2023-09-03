#include "Poisson1D.hpp"

// Initialization.
void Poisson1D::setup() {
    // Build the mesh: describe the domain and its discretization

    // Construct the information for the FE space that we gonna use
    // We're going to use linear finite elements

    // Initialize the DoF (Degree of freedom) handler. Nothing but the unknowns of our problem
    // It's a dealii class that does a lot of stuff under the hood to manage the numbering of this DoF and the local to global map

    // Build the Linear algebra structure: Initilize the system matrix, the vector. Only allocation not assemble


    // *********** Build the mesh ***********
    
    // Generates the cube or a square or a line ... depending on the dimention
    // the dimention is known from the template variable dim used in the definition of mesh
    // N+1 is the number of elements that I want.
    // 0.0, 1.0 left and right extremes
    // true to label the boundary of this mesh
    GridGenerator::subdivided_hyper_cube(mesh,N+1,0.0, 1.0,true);
    // you can read a mesh from file for more complicated cases

    std::cout << "There are " << mesh.n_active_cells() << " mesh elements" << std::endl;

    // Write a mesh to a file
    {
        const std::string mesh_file_name = "mesh-" + std::to_string(N+1) + ".vtk";
        // Open a stream with this filename
        std::ofstream grid_out_file(mesh_file_name);
        // dealii class that help writing meshes to file
        GridOut grid_out;
        grid_out.write_vtk(mesh,grid_out_file);

        // file closed automatically due to brackets {}
    }


    // *********** FE ***********
    {
        // FE_Q is the concrete class. It represents FE on quadrilateral elements (2D-> squares)
        // r is the polynomial degree
        fe = std::make_unique<FE_Q<dim>>(r);

        // r+1 is the number of points for the qudrature nodes
        quadrature = std::make_unique<QGauss<dim>>(r+1);
    }

    // *********** Initilize the DoF handler ***********
    {
        // Tell it on what mesh it's definied
        dof_handler.reinit(mesh);
        // Tell it what FE space we're using
        dof_handler.distribute_dofs(*fe);

        std::cout << "Number of DoF (unknows): " << dof_handler.n_dofs() << std::endl;
    }

    // *********** Linear algebra ***********
    {
        // Sparsity pattern
        DynamicSparsityPattern dsp(dof_handler.n_dofs()); // different data structure, faster and easier to write in
        // Fill the sparsity pattern
        DoFTools::make_sparsity_pattern(dof_handler,dsp);

        // Faster to read from so copy in it
        sparsity_pattern.copy_from(dsp);

        // Use it to initialize the matrix
        system_matrix.reinit(sparsity_pattern);


        // Initilize the vectors just telling them the size (number of unknowns)
        system_rhs.reinit(dof_handler.n_dofs());
        system_solution.reinit(dof_handler.n_dofs());
    }
}

// System assembly.
void Poisson1D::assemble() {
    std::cout << "Assembling the linear system" << std::endl;

    // Clear out the system matrix and vector to be sure
    system_matrix = 0.0;
    system_rhs = 0.0;

    // Store the local contribution of each element in a small matrix and a small vector
    // Can get how large they are from the fe object:
    const unsigned int dof_per_cell = fe->dofs_per_cell;
    FullMatrix<double> cell_matrix(dof_per_cell*dof_per_cell);
    Vector<double> cell_rhs(dof_per_cell);

    // To assemble the matrix each element is the sum of non zero integral that in a computer are computed with quadratures
    // This means that the integral become a sum from 1 to quadrature size
    const unsigned int n_q_points = quadrature->size();

    // It helps us in walking over mesh elements and computing the formula in the integrals
    // FE space to know which basis functions I need
    // Needs the quadrature rule to know on which points I need to evaluate the basis functions
    // Needs to know what quantities I need to compute with a series of flags
        // update_values: we want to know the values of the shape/basis funtions at the quadrature nodes
        // update_gradients: we want to know the derivative of the basis functions
        // update_quadrature_points: we want to know the fisical location of the quadrature points
        // update_JxW_values: we need the value of the product of the jacobian and the quadrature weight
    FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);

    // It contains the global indeces of the current element
    std::vector<types::global_dof_index> dof_indices(dof_per_cell);

    // loop over all the elements
    for (const auto &cell : dof_handler.active_cell_iterators()) { // range base for loop
        cell_matrix = 0.0;
        cell_rhs = 0.0;

        fe_values.reinit(cell);

        for (unsigned int q = 0; q < n_q_points; ++q) {
            
            // Build the local matrix
            for ( unsigned int i = 0; i < dof_per_cell; ++i){ // rows
                for ( unsigned int j = 0; j < dof_per_cell; ++j){
                    // the Aij is:
                        // fe_values.shape_grad(j,q): derivative of the local basis function on the quadrature node q. 
                            // It already accounts for the jacobian and that we are working with reference configuration
                            
                    cell_matrix(i,j) =    diffusion_coefficient.value(fe_values.quadrature_point(q)) // mu(psi_c(xi_q))
                                        * fe_values.shape_grad(j,q) // In the laboratory notes this is contribution (I)
                                        * fe_values.shape_grad(i,q) // contribution (II)
                                        * fe_values.JxW(q); // contribution (III)
                }
                cell_rhs(i) = forcing_term.value(fe_values.quadrature_point(q)) * fe_values.shape_value(i,q) * fe_values.JxW(q);
            }

            
            cell->get_dof_indices(dof_indices);

            // Add in the positions indicated in dof_inceces add the contribution of cell_matrix
            system_matrix.add(dof_indices,cell_matrix);
            system_rhs.add(dof_indices,cell_rhs);
        }
    }
    // Apply boundary conditions
    // we've labeled our boundaries
    {
        Functions::ZeroFunction<dim> bc_function;

        std::map<types::boundary_id, const Function<dim> *> boundary_functions;

        // Boundary ids are 0 for the left and 1 for the right. I know it from the GridGenerator::subdivided_hyper_cube documentation
        // Boundary ids are NOT the coordinates! they are labels
        boundary_functions[0] = &bc_function;
        boundary_functions[1] = &bc_function;

        // Associate to any unknown of the problem a double that represents the corresponding boundary value
        std::map<types::global_dof_index, double> boundary_values;
        // To fill it we can use a dealii function
        VectorTools::interpolate_boundary_values(dof_handler, boundary_functions, boundary_values);

        // I've to take to account of this boundary conditions in my system
        // Drop the first and the last equations and replace them zeros except for the i,i elem = 1 (not sure of this 1)
        // The flag tell him: If I start with a symmetric matrix I want also the output matrix symmetric
        MatrixTools::apply_boundary_values(boundary_values,system_matrix,system_solution, system_rhs, true);
    }
}

// System solution.
void Poisson1D::solve() {
    // You can solve it in many different solver. Since the problem is small you can even use a direct solver
    // dealii offers few direct solver, It's easier to go with an iterative solver.

    // Problem is symmetric so lets use Gonjugate gradient
    // Stopping criteria
    unsigned int max_it = 1000;
    double tollerance = 1e-6 * system_rhs.l2_norm(); // relative stopping criteria
    SolverControl solver_control(max_it, tollerance);
    SolverCG<Vector<double>> solver(solver_control);

    // Solve the system using no preconditioners since the problem is small
    solver.solve(system_matrix,system_solution,system_rhs, PreconditionIdentity());

    std::cout << solver_control.last_step() << " CG iterations" << std::endl;
}

// Output.
void Poisson1D::output() const {
    // Write the solution to a file that can be opened from paraview

    DataOut<dim> data_out;

    // Insert 1 or more solution vectors, in our case only 1
    // Name used when we visualize the solution
    data_out.add_data_vector(dof_handler, system_solution, "solution");


    // Write it
    const std::string out_file_name = "myoutput-" + std::to_string(N+1) + ".vtk";
    data_out.build_patches();
    std::ofstream output_file(out_file_name);
    data_out.write_vtk(output_file);

}