#include "ParametricHeat.hpp"

void
ParametricHeat::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;
    #if IS_MESH_FROM_FILE

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);
    std::ifstream grid_in_file(MESH_PATH(N));

    grid_in.read_msh(grid_in_file);
    #else
    GridGenerator::subdivided_hyper_cube(mesh_serial, N + 1, DOMAIN_START, DOMAIN_END, true);
    #if ! IS_QUADRILATERAL_MESH
    GridGenerator::convert_hypercube_to_simplex_mesh(mesh_serial, mesh_serial);
    #endif
    #endif

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    #if IS_QUADRILATERAL_MESH
    fe = std::make_unique<FE_Q<dim>>(r);
    #else
    fe = std::make_unique<FE_SimplexP<dim>>(r);
    #endif

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    #if IS_QUADRILATERAL_MESH
    quadrature = std::make_unique<QGauss<dim>>(r + 1);
    #else
    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
    #endif

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    // Quadrature for Newman conditions
    #if IS_QUADRILATERAL_MESH
    quadrature_boundary = std::make_unique<QGauss<dim - 1>>(r + 1);
    #else
    quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(r + 1);
    #endif

    pcout << "  Quadrature points per boundary cell = "
              << quadrature_boundary->size() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    
    // could use the same names and do a single formulation but I don't have time
    #if IS_LINEAR
    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    #else
    jacobian_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    solution_old = solution;
    #endif
  }
  // Create all the boundary functions
  {
    
    create_boundary_fun();

  }
}

#if ! IS_LINEAR
void
ParametricHeat::assemble_system()
{
  
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_values_boundary(*fe,
                                       *quadrature_boundary,
                                       update_values |
                                         update_quadrature_points |
                                         update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_residual(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  residual_vector = 0.0;

  // Value and gradient of the solution on current cell.
  std::vector<double>         solution_loc(n_q);
  std::vector<Tensor<1, dim>> solution_gradient_loc(n_q);

  // Value of the solution at previous timestep (un) on current cell.
  std::vector<double> solution_old_loc(n_q);

  forcing_term.set_time(time);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix   = 0.0;
      cell_residual = 0.0;

      // u_n+1
      fe_values.get_function_values(solution, solution_loc);
      // grad (u_n+1)
      fe_values.get_function_gradients(solution, solution_gradient_loc);
      // u_n
      fe_values.get_function_values(solution_old, solution_old_loc);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Evaluate coefficients on this quadrature node.
          const double mu_0_loc = mu_0.value(fe_values.quadrature_point(q));
          const double mu_1_loc = mu_1.value(fe_values.quadrature_point(q));
          const double f_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Mass matrix.
                  cell_matrix(i, j) += fe_values.shape_value(i, q) *
                                       fe_values.shape_value(j, q) / deltat *
                                       fe_values.JxW(q);

                  // Non-linear stiffness matrix, first term.
                  cell_matrix(i, j) += (mu_0_loc
                      + 2.0 * mu_1_loc * fe_values.shape_value(j, q) * solution_loc[q]) *
                    scalar_product(solution_gradient_loc[q],
                                   fe_values.shape_grad(i, q)) *
                    fe_values.JxW(q);

                  // Non-linear stiffness matrix, second term.
                  cell_matrix(i, j) += (mu_0_loc 
                    + mu_1_loc * solution_loc[q] * solution_loc[q] ) *
                    scalar_product(fe_values.shape_grad(j, q),
                                   fe_values.shape_grad(i, q)) *
                    fe_values.JxW(q);

                }

              // Assemble the residual vector (with changed sign due to -R).

              // Time derivative term.
              cell_residual(i) -= (solution_loc[q] - solution_old_loc[q]) /
                                  deltat * fe_values.shape_value(i, q) *
                                  fe_values.JxW(q);

              // Diffusion term.
              cell_residual(i) -=
                (mu_0_loc + mu_1_loc * solution_loc[q] * solution_loc[q]) *
                scalar_product(solution_gradient_loc[q],
                               fe_values.shape_grad(i, q)) *
                fe_values.JxW(q);

              // Forcing term.
              cell_residual(i) +=
                f_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);

      jacobian_matrix.add(dof_indices, cell_matrix);
      residual_vector.add(dof_indices, cell_residual);
    }

  jacobian_matrix.compress(VectorOperation::add);
  residual_vector.compress(VectorOperation::add);

  // We apply Dirichlet boundary conditions.
  // The linear system solution is delta, which is the difference between
  // u_{n+1}^{(k+1)} and u_{n+1}^{(k)}. Both must satisfy the same Dirichlet
  // boundary conditions: therefore, on the boundary, delta = u_{n+1}^{(k+1)} -
  // u_{n+1}^{(k+1)} = 0. We impose homogeneous Dirichlet BCs.
  {
    std::map<types::global_dof_index, double> boundary_values;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    for (unsigned int i = 0; i < dirichlet_ids.size();++i){
      boundary_functions[dirichlet_ids[i]] = &(dirichlet_functions[i]);
    }
     
    VectorTools::interpolate_boundary_values(dof_handler,
                                              boundary_functions,
                                              boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, jacobian_matrix, delta_owned, residual_vector, false);
  }
}

void
ParametricHeat::solve_linear_system()
{
  SolverControl solver_control(N_MAX_ITERS, TOLERANCE * residual_vector.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
    jacobian_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(SSOR_PARAM));

  solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
}

void
ParametricHeat::solve_newton()
{
  const unsigned int n_max_iters        = N_MAX_ITERS;
  const double       residual_tolerance = TOLERANCE;

  unsigned int n_iter        = 0;
  double       residual_norm = residual_tolerance + 1;

  // ************** Difference here
  // We apply the boundary conditions to the initial guess (which is stored in
  // solution_owned and solution).
  {
    IndexSet dirichlet_dofs = DoFTools::extract_boundary_dofs(dof_handler);
    dirichlet_dofs          = dirichlet_dofs & dof_handler.locally_owned_dofs();

    function_g.set_time(time);

    TrilinosWrappers::MPI::Vector vector_dirichlet(solution_owned);
    VectorTools::interpolate(dof_handler, function_g, vector_dirichlet);

    for (const auto &idx : dirichlet_dofs)
      solution_owned[idx] = vector_dirichlet[idx];

    solution_owned.compress(VectorOperation::insert);
    solution = solution_owned;

//     IndexSet boundary_dofs = DoFTools::extract_boundary_dofs(dof_handler);
//     boundary_dofs          = boundary_dofs & dof_handler.locally_owned_dofs();
    
//     bool found = false;
//     for (unsigned int i = 0; i < dirichlet_ids.size();++i){
//       // Check if this dof is locally owned:
//       found = false;
//       for (const auto &idx : boundary_dofs) {
//         if (idx == dirichlet_ids[i]){
//           found = true;
//           break;
//         }
           
//       }
//       if (found) {
//         TrilinosWrappers::MPI::Vector vector_dirichlet(solution_owned);
//         dirichlet_functions[i].set_time(time);
//         VectorTools::interpolate(dof_handler, dirichlet_functions[i], vector_dirichlet);
// // NOT SURE HERE 
//         solution_owned[i] = vector_dirichlet[i];
//       }
    // }


    // solution_owned.compress(VectorOperation::insert);
    // solution = solution_owned;
  }

  while (n_iter < n_max_iters && residual_norm > residual_tolerance)
    {
      assemble_system();
      residual_norm = residual_vector.l2_norm();

      pcout << "  Newton iteration " << n_iter << "/" << n_max_iters
            << " - ||r|| = " << std::scientific << std::setprecision(6)
            << residual_norm << std::flush;

      // We actually solve the system only if the residual is larger than the
      // tolerance.
      if (residual_norm > residual_tolerance)
        {
          solve_linear_system();

          solution_owned += delta_owned;
          solution = solution_owned;
        }
      else
        {
          pcout << " < tolerance" << std::endl;
        }

      ++n_iter;
    }
}
#endif // ! IS_LINEAR

void
ParametricHeat::output(const unsigned int &time_step, const double &time) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  std::string output_file_name = std::to_string(time_step);

  // Pad with zeros.
  output_file_name = "output-" + std::string(4 - output_file_name.size(), '0') +
                     output_file_name;

  DataOutBase::DataOutFilter data_filter(
    DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices = */ false,
                                    /*xdmf_hdf5_output = */ true));
  data_out.write_filtered_data(data_filter);
  data_out.write_hdf5_parallel(data_filter,
                               output_file_name + ".h5",
                               MPI_COMM_WORLD);

  std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(
    data_filter, output_file_name + ".h5", time, MPI_COMM_WORLD)});
  data_out.write_xdmf_file(xdmf_entries,
                           output_file_name + ".xdmf",
                           MPI_COMM_WORLD);
}

void
ParametricHeat::solve()
{
  
  pcout << "===============================================" << std::endl;
  #if IS_LINEAR
  assemble_matrices();
  #endif

  time = INI_TIME;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    #if CHECK_CONVERGENCE
    exact_solution.set_time(time);
    VectorTools::interpolate(dof_handler, exact_solution, solution_owned);
    #else
    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    #endif
    
    solution = solution_owned;

    // Output the initial solution.
    output(0, 0.0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat) // due to finite arithmetic
    {
      time += deltat;
      ++time_step;

      #if IS_LINEAR
      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      assemble_rhs(time);
      solve_time_step();
      output(time_step, time);
      #else
      // Store the old solution, so that it is available for assembly.
      solution_old = solution;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << std::fixed << time << std::endl;

      // At every time step, we invoke Newton's method to solve the non-linear
      // problem.
      solve_newton();

      output(time_step, time);

      pcout << std::endl;
      #endif
    }
}

#if CHECK_CONVERGENCE
double
ParametricHeat::compute_error(const VectorTools::NormType &norm_type)
{
  #if IS_QUADRILATERAL_MESH
  FE_Q<dim> fe_linear(1);
  const QGauss<dim> quadrature_error = QGauss<dim>(r + 2);
  #else
  FE_SimplexP<dim> fe_linear(1);
  const QGaussSimplex<dim> quadrature_error = QGaussSimplex<dim>(r + 2);
  #endif
  MappingFE        mapping(fe_linear);

  

  // ************ Difference here
  // Time is a member of ParametricHeat class, no more defined in solve method
  exact_solution.set_time(time);

  Vector<double> error_per_cell;
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}
#endif

#if IS_LINEAR
void
ParametricHeat::assemble_matrices()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix      = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_mass_matrix      = 0.0;
      cell_stiffness_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Evaluate coefficients on this quadrature node.
          const double mu_loc = mu_0.value(fe_values.quadrature_point(q));
          const double sig_loc = reaction_coefficient.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                            fe_values.shape_value(j, q) /
                                            deltat * fe_values.JxW(q);

                  // diffusion
                  cell_stiffness_matrix(i, j) +=
                    mu_loc * fe_values.shape_grad(i, q) *
                    fe_values.shape_grad(j, q) * fe_values.JxW(q);

                  // reaction
                  cell_stiffness_matrix(i, j) += sig_loc * // sigma
                    fe_values.shape_value(i, q) *      // phi_i
                    fe_values.shape_value(j, q) *      // phi_j
                    fe_values.JxW(q);                  // dx
                }
            }
        }
      

      cell->get_dof_indices(dof_indices);

      mass_matrix.add(dof_indices, cell_mass_matrix);
      stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
    }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // We build the matrix on the left-hand side of the algebraic problem (the one
  // that we'll invert at each timestep).
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix.add(theta, stiffness_matrix);

  // We build the matrix on the right-hand side (the one that multiplies the old
  // solution un).
  rhs_matrix.copy_from(mass_matrix);
  rhs_matrix.add(-(1.0 - theta), stiffness_matrix);
}

void
ParametricHeat::assemble_rhs(const double &time)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);
                      
  // Since we need to compute integrals on the boundary for Neumann conditions,
  // we also need a FEValues object to compute quantities on boundary edges
  // (faces).
  FEFaceValues<dim> fe_values_boundary(*fe,
                                       *quadrature_boundary,
                                       update_values |
                                         update_quadrature_points |
                                         update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // We need to compute the forcing term at the current time (tn+1) and
          // at the old time (tn). deal.II Functions can be computed at a
          // specific time by calling their set_time method.

          // Compute f(tn+1)
          forcing_term.set_time(time);
          const double f_new_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          // Compute f(tn)
          forcing_term.set_time(time - deltat);
          const double f_old_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            cell_rhs(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) *
                            fe_values.shape_value(i, q) * fe_values.JxW(q);
          }
        }
      // Newman here very inefficient because constant over time but easier
      if (cell->at_boundary() && newman_ids.size()>0)
        {
          // ...we loop over its edges (referred to as faces in the deal.II
          // jargon).
          for (unsigned int face_number = 0; face_number < cell->n_faces();
               ++face_number)
            {
              // If current face lies on the boundary, and its boundary ID (or
              // tag) is that of one of the Neumann boundaries, we assemble the
              // boundary integral.
              int id_pos = -1;
              for (unsigned i = 0; i < newman_ids.size() && id_pos == -1; i++){
                  if (newman_ids[i] == cell->face(face_number)->boundary_id()) {
                    id_pos = i;
                  }
              }

              if (id_pos != -1 && cell->face(face_number)->at_boundary())
                {
                  
                  fe_values_boundary.reinit(cell, face_number);

                  for (unsigned int q = 0; q < quadrature_boundary->size(); ++q) {
                    // Compute h(tn+1)
                    newman_functions[id_pos].set_time(time);
                    const double newman_new_loc =
                      newman_functions[id_pos].value(fe_values_boundary.quadrature_point(q));

                    // Compute h(tn)
                    newman_functions[id_pos].set_time(time - deltat);
                    const double newman_old_loc =
                      newman_functions[id_pos].value(fe_values_boundary.quadrature_point(q));
                    
                    for (unsigned int i = 0; i < dofs_per_cell; ++i){
                      cell_rhs(i) += (theta * newman_new_loc + (1.0 - theta) * newman_old_loc) *
                             fe_values_boundary.shape_value(i, q) * fe_values_boundary.JxW(q);
                    }
                  }
                  
                }
            }
        }

      cell->get_dof_indices(dof_indices);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_rhs.compress(VectorOperation::add);

  // Add the term that comes from the old solution.
  rhs_matrix.vmult_add(system_rhs, solution_owned);

  // We apply boundary conditions to the algebraic system.
  {
    std::map<types::global_dof_index, double> boundary_values;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    
// CHECK HERE ***************
    #if ! CHECK_CONVERGENCE
    for (unsigned int i = 0; i < dirichlet_ids.size();++i){
      boundary_functions[dirichlet_ids[i]] = &(dirichlet_functions[i]);
    }
    #else
    for (const auto idx : DoFTools::extract_boundary_dofs(dof_handler))
      boundary_functions[idx] = &exact_solution;
    #endif

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, lhs_matrix, solution_owned, system_rhs, false);
  }
}

void
ParametricHeat::solve_time_step()
{
  SolverControl solver_control(N_MAX_ITERS, TOLERANCE * system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
    lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(SSOR_PARAM));

  solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;

  solution = solution_owned;
}
#endif // IS_LINEAR