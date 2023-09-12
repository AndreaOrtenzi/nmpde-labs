#include "ParametricPoisson.hpp"

void
ParametricPoisson::setup()
{
  #if ! GET_TIME
  pcout << "===============================================" << std::endl;
  #endif
  // Create the mesh.
  {
    #if ! GET_TIME
    pcout << "Initializing the mesh" << std::endl;
    #endif
    Triangulation<dim> mesh_serial;
    {
      #if IS_MESH_FROM_FILE

      GridIn<dim> grid_in;
      grid_in.attach_triangulation(mesh_serial);
      std::ifstream grid_in_file(MESH_PATH(N));

      grid_in.read_msh(grid_in_file);
      #else
      pcout << "Generating mesh" << std::endl;
      GridGenerator::subdivided_hyper_cube(mesh_serial, N + 1, DOMAIN_START, DOMAIN_END, true);
      #if ! IS_QUADRILATERAL_MESH
      GridGenerator::convert_hypercube_to_simplex_mesh(mesh_serial, mesh_serial);
      #endif
      #endif
    }
  // Then, we copy the triangulation into the parallel one.
    {
      // Split among mpi ranks
      GridTools::partition_triangulation(mpi_size, mesh_serial);
      // Describes what the triangulation contains without having the triangulation
      const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      // Use the description to create the distributed mesh
      mesh.create_triangulation(construction_data);
    }
    #if ! GET_TIME
    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
    #endif
  }

  #if ! GET_TIME
  pcout << "-----------------------------------------------" << std::endl;
  #endif

  // Initialize the finite element space.
  {
    #if ! GET_TIME
    pcout << "Initializing the finite element space" << std::endl;
    #endif
    // Construct the finite element object. Notice that we use the FE_SimplexP
    // class here, that is suitable for triangular (or tetrahedral) meshes.
    #if IS_QUADRILATERAL_MESH
    fe = std::make_unique<FE_Q<dim>>(r);
    #else
    fe = std::make_unique<FE_SimplexP<dim>>(r);
    #endif

    #if ! GET_TIME
    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;
    #endif

    // Construct the quadrature formula of the appopriate degree of exactness.
    #if IS_QUADRILATERAL_MESH
    quadrature = std::make_unique<QGauss<dim>>(r + 1);
    #else
    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
    #endif

    #if ! GET_TIME
    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
    #endif
  }
  #if ! GET_TIME
  pcout << "-----------------------------------------------" << std::endl;
  #endif
  // Initialize the DoF handler.
  {
    #if ! GET_TIME
    pcout << "Initializing the DoF handler" << std::endl;
    #endif

    // Initialize the DoF handler with the mesh we constructed.
    dof_handler.reinit(mesh);

    // "Distribute" the degrees of freedom. For a given finite element space,
    // initializes info on the control variables (how many they are, where
    // they are collocated, their "global indices", ...).
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();

    #if ! GET_TIME
    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
    #endif
  }
  #if ! GET_TIME
  pcout << "-----------------------------------------------" << std::endl;
  #endif

  // Initialize the linear system.
  {
    #if ! GET_TIME
    pcout << "Initializing the linear system" << std::endl;
    #endif

    // We first initialize a "sparsity pattern", i.e. a data structure that
    // indicates which entries of the matrix are zero and which are different
    // from zero. To do so, we construct first a DynamicSparsityPattern (a
    // sparsity pattern stored in a memory- and access-inefficient way, but
    // fast to write) and then convert it to a SparsityPattern (which is more
    // efficient, but cannot be modified).
    #if ! GET_TIME
    pcout << "  Initializing the sparsity pattern" << std::endl;
    #endif
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    // Then, we use the sparsity pattern to initialize the system matrix
    #if ! GET_TIME
    pcout << "  Initializing the system matrix" << std::endl;
    #endif
    
    system_matrix.reinit(sparsity);

    // Finally, we initialize the right-hand side and solution vectors.
    #if ! GET_TIME
    pcout << "  Initializing the system right-hand side" << std::endl;
    #endif
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    #if ! GET_TIME
    pcout << "  Initializing the solution vector" << std::endl;
    #endif
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }

  // Create all the boundary functions
  {
    
    create_boundary_fun();

  }
}

void
ParametricPoisson::assemble()
{
  #if ! GET_TIME
  pcout << "===============================================" << std::endl;

  pcout << "  Assembling the linear system" << std::endl;
  #endif
  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  // FEValues instance. This object allows to compute basis functions, their
  // derivatives, the reference-to-current element mapping and its
  // derivatives on all quadrature points of all elements.
  FEValues<dim> fe_values(
    *fe,
    *quadrature,
    // Here we specify what quantities we need FEValues to compute on
    // quadrature points. For our test, we need:
    // - the values of shape functions (update_values);
    // - the derivative of shape functions (update_gradients);
    // - the position of quadrature points (update_quadrature_points);
    // - the product J_c(x_q)*w_q (update_JxW_values).
    update_values | update_gradients | update_quadrature_points |
      update_JxW_values);

  // Local matrix and right-hand side vector. We will overwrite them for
  // each element within the loop.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  // We will use this vector to store the global indices of the DoFs of the
  // current element within the loop.
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // Reinitialize the FEValues object on current element. This
      // precomputes all the quantities we requested when constructing
      // FEValues (see the update_* flags above) for all quadrature nodes of
      // the current cell.
      fe_values.reinit(cell);

      // We reset the cell matrix and vector (discarding any leftovers from
      // previous element).
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Here we assemble the local contribution for current cell and
          // current quadrature point, filling the local matrix and vector.
          const double forcing_val = forcing_term.value(fe_values.quadrature_point(q));
          const double diffusion_val = diffusion_coefficient.value(
                                         fe_values.quadrature_point(q));
          // Here we iterate over *local* DoF indices.
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Diffusion term.
                  cell_matrix(i, j) +=  diffusion_val// mu(x)
                                       * fe_values.shape_grad(i, q)     // (I)
                                       * fe_values.shape_grad(j, q)     // (II)
                                       * fe_values.JxW(q);              // (III)
                }

              cell_rhs(i) += forcing_val *
                             fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);

      // Then, we add the local matrix and vector into the corresponding
      // positions of the global matrix and vector.
      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Boundary conditions.
  {
    // We construct a map that stores, for each DoF corresponding to a
    // Dirichlet condition, the corresponding value. E.g., if the Dirichlet
    // condition is u_i = b, the map will contain the pair (i, b).
    std::map<types::global_dof_index, double> boundary_values;

    // Then, we build a map that, for each boundary tag, stores the
    // corresponding boundary function.

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    Functions::ZeroFunction<dim> bc_function;

    for (unsigned int i = 0; i < DIM*2;++i){
      boundary_functions[i] = &(bc_function);
    }

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    // Finally, we modify the linear system to apply the boundary
    // conditions. This replaces the equations for the boundary DoFs with
    // the corresponding u_i = 0 equations.
    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  }
}

void
ParametricPoisson::solve()
{
  #if ! GET_TIME
  pcout << "===============================================" << std::endl;
  #endif
  // Here we specify the maximum number of iterations of the iterative solver,
  // and its tolerance.
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

  // Since the system matrix is symmetric and positive definite, we solve the
  // system using the conjugate gradient method.
  /*
    0 -> no prec
    1 -> Jacobi
    2 -> GS
    3 -> SOR
    4 -> SSOR
  */
  #if PRECODITIONER == 2 || PRECONDITIONER == 3
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSOR preconditioner; 
  constexpr double r = R_SOR;
  preconditioner.initialize(system_matrix, TrilinosWrappers::PreconditionSOR::AdditionalData(r)); 
  #else
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  #if PRECODITIONER == 0
  TrilinosWrappers::PreconditionIdentity preconditioner;
  
  #elif PRECODITIONER == 1
  TrilinosWrappers::PreconditionJacobi preconditioner;
  preconditioner.initialize(system_matrix);
  
  #elif PRECODITIONER == 4
  TrilinosWrappers::PreconditionSSOR preconditioner;
  constexpr double r = R_SOR;
  preconditioner.initialize(system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(r)); 

  #endif
  #endif

  #if ! GET_TIME
  pcout << "  Solving the linear system" << std::endl;
  #endif
  // We don't use any preconditioner for now, so we pass the identity matrix
  // as preconditioner.
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " iterations"
            << std::endl;
}

void
ParametricPoisson::output() const
{
  #if ! GET_TIME
  pcout << "===============================================" << std::endl;
  #endif

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs,
                                               locally_relevant_dofs,
                                               MPI_COMM_WORLD);
  solution_ghost = solution;
  // The DataOut class manages writing the results to a file.
  DataOut<dim> data_out;

  // It can write multiple variables (defined on the same mesh) to a single
  // file. Each of them can be added by calling add_data_vector, passing the
  // associated DoFHandler and a name.
  data_out.add_data_vector(dof_handler, solution_ghost, "solution");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  // partitioning has only one element for each cell of the domain, not fro each dof so no dof_handler
  data_out.add_data_vector(partitioning, "partitioning");
  data_out.build_patches();

  // Then, use one of the many write_* methods to write the file in an
  // appropriate format.
  const std::string output_file_name =
    "output-" + std::to_string(N + 1);
  DataOutBase::DataOutFilter data_filter(
    DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices = */ false,
                                    /*xdmf_hdf5_output = */ true));
  data_out.write_filtered_data(data_filter);
  // Write in binary format hdf5 but it cann't be opened alone so write also a "legend" in xdmf
  data_out.write_hdf5_parallel(data_filter,
                               output_file_name + ".h5",
                               MPI_COMM_WORLD);

  // It contains pointers to hdf5 file 
  std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(
    data_filter, output_file_name + ".h5", 0.0, MPI_COMM_WORLD)});
  data_out.write_xdmf_file(xdmf_entries,
                           output_file_name + ".xdmf",
                           MPI_COMM_WORLD);

  #if ! GET_TIME
  pcout << "Output written to " << output_file_name << std::endl;

  pcout << "===============================================" << std::endl;
  #endif
}


#if CHECK_CONVERGENCE
double
ParametricPoisson::compute_error(const VectorTools::NormType &norm_type) const
{
  #if IS_QUADRILATERAL_MESH
  const QGauss<dim> quadrature_error(r + 2);
  #else
  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);
  const QGaussSimplex<dim> quadrature_error(r + 2);
  #endif
  // The error is an integral, and we approximate that integral using a
  // quadrature formula. To make sure we are accurate enough, we use a
  // quadrature formula with one node more than what we used in assembly.
  

  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell(mesh.n_active_cells());
  #if IS_QUADRILATERAL_MESH
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    ExactSolution(),
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);
  #else
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    ExactSolution(),
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);
  #endif
  // Then, we add out all the cells.
  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}
#endif