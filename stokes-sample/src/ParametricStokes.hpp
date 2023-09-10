#ifndef STOKES_HPP
#define STOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// ******************************************************************************************
// Problem parameters
#define DIM 3
#define VEL_DIM DIM
#define PRES_DIM 1
#define H_VALUES 0.2//0.1, 0.05, 0.025, 0.0125}
#define N_VALUES 4//9, 19, 39, 79}
#define P_GRADE 1
#define V_GRADE 2

// nu (time independent)
static constexpr auto lin_diffusion_function = [](const dealii::Point<DIM> &p) -> double { 

  return 1.0;
};

// f
static constexpr auto forcing_function = [](const dealii::Point<DIM> &p,const Function<DIM> *fun, Vector<double> &values) -> void {

    // values[0] = -1.0 * p[1] * (2.0 - p[1]) * (1.0 - p[2]) * (2.0 - p[2]);

    for (unsigned int i = 0; i < DIM; ++i)
      values[i] = 0.0;
    
    // return (29 * M_PI * M_PI * std::sin(5 * M_PI * fun->get_time()) +
    //           5 * M_PI * std::cos(5 * M_PI * fun->get_time())) *
    //          std::sin(2 * M_PI * p[0]) * std::sin(3 * M_PI * p[1]) *
    //          std::sin(4 * M_PI * p[2]);
};

// exact solution:
#define CHECK_CONVERGENCE false
#if CHECK_CONVERGENCE
static constexpr auto exact_sol_function = [](const dealii::Point<DIM> &p,const Function<DIM> *fun) -> double {
    
    return std::sin(5*M_PI*fun->get_time())*std::sin(2*M_PI*p[0])*std::sin(3*M_PI*p[1])*std::sin(4*M_PI*p[2]);
};
// gradient exact solution:
static constexpr auto gradient_exact_sol_function = [](const dealii::Point<DIM> &p, dealii::Tensor<1, DIM> &result,const Function<DIM> *fun) {

    // duex / dx
    result[0] = 2 * M_PI * std::sin(5 * M_PI * fun->get_time()) *
                std::cos(2 * M_PI * p[0]) * std::sin(3 * M_PI * p[1]) *
                std::sin(4 * M_PI * p[2]);

    // duex / dy
    result[1] = 3 * M_PI * std::sin(5 * M_PI * fun->get_time()) *
                std::sin(2 * M_PI * p[0]) * std::cos(3 * M_PI * p[1]) *
                std::sin(4 * M_PI * p[2]);

    // duex / dz
    result[2] = 4 * M_PI * std::sin(5 * M_PI * fun->get_time()) *
                std::sin(2 * M_PI * p[0]) * std::sin(3 * M_PI * p[1]) *
                std::cos(4 * M_PI * p[2]);
};
#endif


// Mesh:
#define IS_MESH_FROM_FILE true
#define IS_QUADRILATERAL_MESH false

#if IS_MESH_FROM_FILE
#ifdef H_VALUES
// #define MESH_PATH(N) "../../examples/gmsh/mesh-square-h" + std::to_string(1.0/(N + 1.0)) + ".msh"
#define MESH_PATH(N) "../../lab-09/mesh/mesh-step-" + std::to_string(N + 1) + ".msh"
#else
#define MESH_PATH(N) "../../examples/gmsh/mesh-square-h" + std::to_string(N + 1) + ".msh"
#endif
#else
#define DOMAIN_START 0.0
#define DOMAIN_END 1.0
#endif

// Newman boundary functions: NOT USED IN NON LINEAR PROBLEMS
#define NM_B_EXISTS true
#define NEWMAN_BOUNDARIES_IDS {2}
static std::vector<std::function<double (const dealii::Point<DIM>&, const Function<DIM> *fun)>> newman_f = {
    [](const Point<DIM> &p, const Function<DIM> *fun) -> double { 
        
      return 10; // Pa
        
    }
};

// Dirichlet boundary functions:
#define DC_B_EXISTS true
// #define DIRICHLET_BOUNDARIES_IDS DoFTools::extract_boundary_dofs(dof_handler)
#define DIRICHLET_BOUNDARIES_IDS {0,1}
static std::vector<std::function<void (const dealii::Point<DIM>&, const Function<DIM> *fun, Vector<double> &values)>> dirichlet_f = {
    
    [](const Point<DIM> &p, const Function<DIM> *fun, Vector<double> &values) -> void {
        values[0] = -1.0 * p[1] * (2.0 - p[1]) * (1.0 - p[2]) * (2.0 - p[2]);
        for (unsigned int i = 1; i < VEL_DIM+PRES_DIM; ++i)
          values[i] = 0.0;
    },

    [](const Point<DIM> &p, const Function<DIM> *fun, Vector<double> &values) -> void {
        for (unsigned int i = 0; i < VEL_DIM+PRES_DIM; ++i)
          values[i] = 0.0;
    }
};


#define TOLERANCE 1e-6
#define N_MAX_ITERS 2000
#define SSOR_PARAM 1.0
// ************************************************************************************************************************************************************************************
// ************************************************************************************************************************************************************************************
// ************************************************************************************************************************************************************************************


// Class implementing a solver for the Stokes problem.
class ParametricStokes
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = DIM;

  // Function for the mu_0 coefficient.
  class FunctionNu : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      return lin_diffusion_function(p);
    }
  };

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    // Evaluation.
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      forcing_function(p,this,values);
    }

    virtual double
    value(const Point<dim> & p,
          const unsigned int component = 0) const override
    {
      Vector<double> values(DIM);
      forcing_function(p,this,values);
      return values[component];
    }
      
  };

  // class DirichletFunction : public Function<dim>
  // {
  // public:
  //   virtual void
  //   vector_value(const Point<dim> & /*p*/,
  //                Vector<double> &values) const override
  //   {
  //     for (unsigned int i = 0; i < dim; ++i)
  //       values[i] = 0.0;

  //     // values[dim - 1] = -g;
  //   }

  //   virtual double
  //   value(const Point<dim> & /*p*/,
  //         const unsigned int component = 0) const override
  //   {
  //     // if (component == dim - 1)
  //     //   return -g;
  //     // else
  //       return 0.0;
  //   }

  // protected:
  //   const double g = 0.0;
  // };

  // Function for inlet velocity. This actually returns an object with four
  // components (one for each velocity component, and one for the pressure), but
  // then only the first three are really used (see the component mask when
  // applying boundary conditions at the end of assembly). If we only return
  // three components, however, we may get an error message due to this function
  // being incompatible with the finite element space.
  class InletVelocity : public Function<dim>
  {
  public:
    InletVelocity()
      : Function<dim>(dim + 1)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      values[0] = -alpha * p[1] * (2.0 - p[1]) * (1.0 - p[2]) * (2.0 - p[2]);

      for (unsigned int i = 1; i < dim + 1; ++i)
        values[i] = 0.0;
    }

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      if (component == 0)
        return -alpha * p[1] * (2.0 - p[1]) * (1.0 - p[2]) * (2.0 - p[2]);
      else
        return 0.0;
    }

  protected:
    const double alpha = 1.0;
  };

  class ParametricNewman : public Function<dim>
  {
  public:
    // Constructor.
    ParametricNewman(std::function<double(const Point<dim>&, const Function<DIM> *fun)> lambdaFunction)
        : lambdaFunction(lambdaFunction) {}

    virtual double
    value(const Point<dim> & p,
          const unsigned int component = 0) const override
    {
      return lambdaFunction(p,this);
    }
    private:
      std::function<double (const Point<dim>&, const Function<DIM> *fun)> lambdaFunction;

  };

  class ParametricDirichlet : public Function<dim>
  {
  public:
    // Constructor.
    ParametricDirichlet(std::function<void(const Point<dim>&, const Function<DIM> *fun, Vector<double> &values)> lambdaFunction)
        : lambdaFunction(lambdaFunction) {}

    // Evaluation.
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      lambdaFunction(p,this,values);
      // values[0] = -alpha * p[1] * (2.0 - p[1]) * (1.0 - p[2]) * (2.0 - p[2]);

      // for (unsigned int i = 1; i < dim + 1; ++i)
      //   values[i] = 0.0;
    }

    virtual double
    value(const Point<dim> & p,
          const unsigned int component = 0) const override
    {
      Vector<double> values(VEL_DIM+PRES_DIM);
      lambdaFunction(p,this,values);
      return values[component];
    }
    private:
      std::function<void (const Point<dim>&, const Function<DIM> *fun, Vector<double> &values)> lambdaFunction;

  };

  // Since we're working with block matrices, we need to make our own
  // preconditioner class. A preconditioner class can be any class that exposes
  // a vmult method that applies the inverse of the preconditioner.

  // Identity preconditioner.
  class PreconditionIdentity
  {
  public:
    // Application of the preconditioner: we just copy the input vector (src)
    // into the output vector (dst).
    void
    vmult(TrilinosWrappers::MPI::BlockVector &      dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      dst = src;
    }

  protected:
  };

  // Block-diagonal preconditioner.
  class PreconditionBlockDiagonal
  {
  public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_) // already divided by nu
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;

      preconditioner_velocity.initialize(velocity_stiffness_);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    // Application of the preconditioner.
    // Any preconditioner class must have this class
    // dst is the block vector P_1^{-1}z
    void
    vmult(TrilinosWrappers::MPI::BlockVector &      dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      SolverControl                           solver_control_velocity(1000,
                                            1e-2 * src.block(0).l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
        solver_control_velocity);
      // velocity_stiffness = A
      // dst.block(0) = A^{-1}z_u
      // src.block(0) = z_u
      // preconditioner_velocity = prec used to solve it
      solver_cg_velocity.solve(*velocity_stiffness,
                               dst.block(0),
                               src.block(0),
                               preconditioner_velocity);

      SolverControl solver_control_pressure(1000,
                      1e-2 * src.block(1).l2_norm()); // large tollerance
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
        solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1), // Put the results in block 1
                               src.block(1),
                               preconditioner_pressure);
    }

  protected:
    // Velocity stiffness matrix.
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    TrilinosWrappers::PreconditionILU preconditioner_velocity;

    // Pressure mass matrix.
    // Black box preconditioner used for A
    const TrilinosWrappers::SparseMatrix *pressure_mass; 

    // Preconditioner used for the pressure block.
    // Black box preconditioner used for M_p
    TrilinosWrappers::PreconditionILU preconditioner_pressure;
  };

  // Block-triangular preconditioner.
  class PreconditionBlockTriangular
  {
  public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_,
               const TrilinosWrappers::SparseMatrix &B_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;
      B                  = &B_;

      preconditioner_velocity.initialize(velocity_stiffness_);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector &      dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      SolverControl                           solver_control_velocity(1000,
                                            1e-2 * src.block(0).l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
        solver_control_velocity);
      solver_cg_velocity.solve(*velocity_stiffness,
                               dst.block(0),
                               src.block(0),
                               preconditioner_velocity);

      tmp.reinit(src.block(1));
      B->vmult(tmp, dst.block(0));
      tmp.sadd(-1.0, src.block(1));

      SolverControl                           solver_control_pressure(1000,
                                            1e-2 * src.block(1).l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
        solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1),
                               tmp,
                               preconditioner_pressure);
    }

  protected:
    // Velocity stiffness matrix.
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    TrilinosWrappers::PreconditionILU preconditioner_velocity;

    // Pressure mass matrix.
    const TrilinosWrappers::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionILU preconditioner_pressure;

    // B matrix.
    const TrilinosWrappers::SparseMatrix *B;

    // Temporary vector.
    mutable TrilinosWrappers::MPI::Vector tmp;
  };

  // Constructor.
  ParametricStokes(const unsigned int &N_,
         const unsigned int &degree_velocity_,
         const unsigned int &degree_pressure_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , N(N_)
    , degree_velocity(degree_velocity_)
    , degree_pressure(degree_pressure_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Setup system.
  void
  setup();

  // Assemble system. We also assemble the pressure mass matrix (needed for the
  // preconditioner).
  void
  assemble();

  // Solve system.
  void
  solve();

  // Output results.
  void
  output();

protected:
  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // Kinematic viscosity [m2/s].
  FunctionNu nu;

  // Outlet pressure [Pa]. Newman = -p_out n
  // const double p_out = 10;

  // Forcing term.
  ForcingTerm forcing_term;

  // Inlet velocity.
  // InletVelocity inlet_velocity;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh refinement.
  const unsigned int N;

  // Polynomial degree used for velocity.
  const unsigned int degree_velocity;

  // Polynomial degree used for pressure.
  const unsigned int degree_pressure;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for face integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs owned by current process in the velocity and pressure blocks.
  std::vector<IndexSet> block_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // DoFs relevant to current process in the velocity and pressure blocks.
  std::vector<IndexSet> block_relevant_dofs;

  // System matrix.
  TrilinosWrappers::BlockSparseMatrix system_matrix;

  // Pressure mass matrix, needed for preconditioning. We use a block matrix for
  // convenience, but in practice we only look at the pressure-pressure block.
  TrilinosWrappers::BlockSparseMatrix pressure_mass;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::BlockVector system_rhs;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::BlockVector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::BlockVector solution;

  // Boundary conditions.
  // Dirichlet
  std::vector<unsigned int> dirichlet_ids = DIRICHLET_BOUNDARIES_IDS;
  std::vector<ParametricDirichlet> dirichlet_functions;

  // Newman
  std::vector<unsigned int> newman_ids = NEWMAN_BOUNDARIES_IDS;
  std::vector<ParametricNewman> newman_functions;

private:
  void create_boundary_fun(){

    for (auto f : dirichlet_f){
      ParametricDirichlet fun(f);
      dirichlet_functions.push_back(fun);
    }
    if (dirichlet_ids.size() != dirichlet_functions.size()) {
      std::cerr << "Size of dirichlet_ids and dirichlet_functions are not equal";
      throw 1;
    }
    for (auto f : newman_f){
      ParametricNewman fun(f);
      newman_functions.push_back(fun);
    }
    if (newman_ids.size() != newman_functions.size()) {
      std::cerr << "Size of newman_ids and newman_functions are not equal";
      throw 1;
    }
  };
};

#endif