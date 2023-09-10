#include "ParametricStokes.hpp"

void
ParametricStokes::setup()
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
    const FE_Q<dim> fe_scalar_velocity(degree_velocity);
    const FE_Q<dim> fe_scalar_pressure(degree_pressure);
    #else
    const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
    const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
    #endif
    fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity,
                                         VEL_DIM,
                                         fe_scalar_pressure,
                                         PRES_DIM);

    pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree
          << std::endl;
    pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree
          << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    #if IS_QUADRILATERAL_MESH
    quadrature = std::make_unique<QGauss<dim>>(fe->degree + 1);
    #else
    quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);
    #endif

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    #if IS_QUADRILATERAL_MESH
    quadrature_face = std::make_unique<QGauss<dim - 1>>(fe->degree + 1);
    #else
    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);
    #endif

    pcout << "  Quadrature points per face = " << quadrature_face->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We want to reorder DoFs so that all velocity DoFs come first, and then
    // all pressure DoFs.
    // This allow the block structure of the block matrix to be preverved, but
    // it would work in any case
    std::vector<unsigned int> block_component(VEL_DIM + PRES_DIM, 0);
    for (unsigned int i=VEL_DIM;i<VEL_DIM+PRES_DIM;++i){
      block_component[i] = 1; // block_component = {0,0,0,1} -> vel block 0, pres block 1
    }
    
    DoFRenumbering::component_wise(dof_handler, block_component);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Besides the locally owned and locally relevant indices for the whole
    // system (velocity and pressure), we will also need those for the
    // individual velocity and pressure blocks.
    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    block_owned_dofs.resize(2);
    block_relevant_dofs.resize(2);
    block_owned_dofs[0]    = locally_owned_dofs.get_view(0, n_u);
    block_owned_dofs[1]    = locally_owned_dofs.get_view(n_u, n_u + n_p);
    block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    pcout << "  Number of DoFs: " << std::endl;
    pcout << "    velocity = " << n_u << std::endl;
    pcout << "    pressure = " << n_p << std::endl;
    pcout << "    total    = " << n_u + n_p << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    // Velocity DoFs interact with other velocity DoFs (the weak formulation has
    // terms involving u times v) [A_ij], and pressure DoFs interact with velocity DoFs
    // (there are terms involving p times v or u times q) [B_ij]. However, pressure
    // DoFs do not interact with other pressure DoFs (there are no terms
    // involving p times q). We build a table to store this information, so that
    // the sparsity pattern can be built accordingly.
    // #row = #col = # components in my problem 
    Table<2, DoFTools::Coupling> coupling(VEL_DIM + PRES_DIM, VEL_DIM + PRES_DIM);
    for (unsigned int c = 0; c < VEL_DIM + PRES_DIM; ++c)
      {
        for (unsigned int d = 0; d < VEL_DIM + PRES_DIM; ++d)
          {
            // Pressure is the last component in my system
            if (c >= VEL_DIM && d >= VEL_DIM) // pressure-pressure term
              coupling[c][d] = DoFTools::none;
            else // other combinations
              coupling[c][d] = DoFTools::always;
          }
      }

    TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                    MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
    sparsity.compress();

    // We also build a sparsity pattern for the pressure mass matrix.
    for (unsigned int c = 0; c < VEL_DIM + PRES_DIM; ++c)
      {
        for (unsigned int d = 0; d < VEL_DIM + PRES_DIM; ++d)
          {
            if (c >= VEL_DIM && d >= VEL_DIM) // pressure-pressure term
              coupling[c][d] = DoFTools::always;
            else // other combinations
              coupling[c][d] = DoFTools::none;
          }
      }
    TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
      block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    coupling,
                                    sparsity_pressure_mass);
    sparsity_pressure_mass.compress();

    pcout << "  Initializing the matrices" << std::endl;
    system_matrix.reinit(sparsity);
    pressure_mass.reinit(sparsity_pressure_mass);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  }
  // Create all the boundary functions
  {
    
    create_boundary_fun();

  }
}

void
ParametricStokes::assemble()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();
  const unsigned int n_q_face      = quadrature_face->size();

  FEValues<dim>     fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  
  // Add flag here to access fe_face_values.normal_vector(q)
  FEFaceValues<dim> fe_face_values(*fe,
                                   *quadrature_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs    = 0.0;
  pressure_mass = 0.0; // Used in the preconditioner

  // There are more efficient solutions but FEValuesExtractors is easier
  FEValuesExtractors::Vector velocity(0); // Vector that start at component 0
// CHANGED HERE VECTOR
  #if PRES_DIM == 1
  FEValuesExtractors::Scalar pressure(VEL_DIM); // Scalar that start at component 3, pressure
  #else
  FEValuesExtractors::Vector pressure(VEL_DIM); // Vector that start at component 3, pressure
  pcout << "FIX ME FROM HERE! FIX cell_matrix(i,j)=..." << std::endl;
  throw 2;
  #endif
  

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix               = 0.0;
      cell_rhs                  = 0.0;
      cell_pressure_mass_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          Vector<double> forcing_term_loc(dim);
          forcing_term.vector_value(fe_values.quadrature_point(q),
                                    forcing_term_loc);
          Tensor<1, dim> forcing_term_tensor;
          for (unsigned int d = 0; d < dim; ++d)
            forcing_term_tensor[d] = forcing_term_loc[d];

          const double nu_loc = nu.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Viscosity term.
                  cell_matrix(i, j) +=
                    nu_loc *
                    scalar_product(fe_values[velocity].gradient(i, q),
                                   fe_values[velocity].gradient(j, q)) *
                    fe_values.JxW(q);

                  // Pressure term in the momentum equation.
                  cell_matrix(i, j) -= fe_values[velocity].divergence(i, q) *
                                       fe_values[pressure].value(j, q) *
                                       fe_values.JxW(q);

                  // Pressure term in the continuity equation.
                  cell_matrix(i, j) -= fe_values[velocity].divergence(j, q) *
                                       fe_values[pressure].value(i, q) *
                                       fe_values.JxW(q);

                  // Pressure mass matrix.
                  cell_pressure_mass_matrix(i, j) +=
                    fe_values[pressure].value(i, q) *
                    fe_values[pressure].value(j, q) / nu_loc * fe_values.JxW(q); // Used in the preconditioner
                }

              // Forcing term.
              cell_rhs(i) += scalar_product(forcing_term_tensor,
                                            fe_values[velocity].value(i, q)) *
                             fe_values.JxW(q);
            }
        }

      // Boundary integral for Neumann BCs.
      if (cell->at_boundary() && newman_ids.size() > 0)
        {
          // Iterate over the basis of the cell
          for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
              int id_pos = -1;
              for (unsigned i = 0; i < newman_ids.size() && id_pos == -1; i++){
                  if (newman_ids[i] == cell->face(f)->boundary_id()) {
                    id_pos = i;
                  }
              }

              if (cell->face(f)->at_boundary() &&
                  id_pos != -1) // CHANGE HERE THE LABEL
                {
                  // 
                  fe_face_values.reinit(cell, f);

                  for (unsigned int q = 0; q < n_q_face; ++q)
                    {
                      const double p_out = newman_functions[id_pos].value(
                          fe_face_values.quadrature_point(q));

                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          cell_rhs(i) +=
                            -p_out *
                            scalar_product(fe_face_values.normal_vector(q),
                                           fe_face_values[velocity].value(i,
                                                                          q)) *
                            fe_face_values.JxW(q);
                        }
                    }
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
      pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
    }
  pcout << "HERE **************" <<std::endl;
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  pressure_mass.compress(VectorOperation::add);

  // Dirichlet boundary conditions.
  {
    std::map<types::global_dof_index, double>           boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    // We interpolate first the inlet velocity condition alone, then the wall
    // condition alone, so that the latter "win" over the former where the two
    // boundaries touch.
    // boundary_functions[0] = &inlet_velocity;
    for (unsigned int i = 0; i < dirichlet_ids.size();++i){
      boundary_functions.clear();
      boundary_functions[dirichlet_ids[i]] = &(dirichlet_functions[i]);
    
      std::vector< bool > comp_mask(VEL_DIM+PRES_DIM,true);
      for (unsigned int i = VEL_DIM; i < VEL_DIM+PRES_DIM; ++i){
        comp_mask[i] = false;
      }

      VectorTools::interpolate_boundary_values(dof_handler,
                                              boundary_functions,
                                              boundary_values,
                                              ComponentMask(
                                                comp_mask)); // Apply b.cond. only to velocity
    }
    // boundary_functions.clear();
    // Functions::ZeroFunction<dim> zero_function(dim + 1);
    // boundary_functions[1] = &zero_function;
    // VectorTools::interpolate_boundary_values(dof_handler,
    //                                          boundary_functions,
    //                                          boundary_values,
    //                                          ComponentMask(
    //                                            {true, true, true, false}));

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, false);
  }
}

void
ParametricStokes::solve()
{
  pcout << "===============================================" << std::endl;

  SolverControl solver_control(N_MAX_ITERS, TOLERANCE * system_rhs.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

  // Defined in the header
  // PreconditionBlockDiagonal preconditioner;
  // preconditioner.initialize(system_matrix.block(0, 0),
  //                           pressure_mass.block(1, 1));

  PreconditionBlockTriangular preconditioner;
  preconditioner.initialize(system_matrix.block(0, 0),
                            pressure_mass.block(1, 1),
                            system_matrix.block(1, 0));

  pcout << "Solving the linear system" << std::endl;
  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " GMRES iterations"
        << std::endl;

  solution = solution_owned;
}

void
ParametricStokes::output()
{
  pcout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  #if PRES_DIM == 1
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);
  #else
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_part_of_vector);
  #endif

  std::vector<std::string> names(VEL_DIM+PRES_DIM,"velocity");
  for (unsigned int i = VEL_DIM; i < VEL_DIM+PRES_DIM; ++i){
    names[i] = "pressure";
  }

  data_out.add_data_vector(dof_handler,
                           solution,
                           names,
                           data_component_interpretation);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string output_file_name = "output-" + std::to_string(N);

  DataOutBase::DataOutFilter data_filter(
    DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices = */ false,
                                    /*xdmf_hdf5_output = */ true));
  data_out.write_filtered_data(data_filter);
  data_out.write_hdf5_parallel(data_filter,
                               output_file_name + ".h5",
                               MPI_COMM_WORLD);

  std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(
    data_filter, output_file_name + ".h5", 0, MPI_COMM_WORLD)});
  data_out.write_xdmf_file(xdmf_entries,
                           output_file_name + ".xdmf",
                           MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << std::endl;
  pcout << "===============================================" << std::endl;
}