#ifndef HEAT_NON_LINEAR_HPP
#define HEAT_NON_LINEAR_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
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
#define H_VALUES {0.1}//0.1, 0.05, 0.025, 0.0125}
#define N_VALUES {19}//9, 19, 39, 79}
#define P_GRADES {1}

// time parameters
#define INI_TIME 0.0
#define FIN_TIME 1.0
#define THETA 1 // Theta method for time discretization, 1->Implicit E.
#define dt_VALUES {0.05}// 0.25, 0.125, 0.0625, 0.03125, 0.015625}

// mu0 (time independent)
static constexpr auto lin_diffusion_function = [](const dealii::Point<DIM> &p) -> double { 

  return 0.1;
};
// mu1 (time independent)
#define IS_LINEAR false
#if ! IS_LINEAR
static constexpr auto quad_diffusion_function = [](const dealii::Point<DIM> &p) -> double { 
    
  return 1.0;
};
#endif

// sigma (time independent) NOT USED IN NON LINEAR PROBLEMS
static constexpr auto reaction_function = [](const dealii::Point<DIM> &p) -> double {
    
  return 0.0;
};

// f
static constexpr auto forcing_function = [](const dealii::Point<DIM> &p,const Function<DIM> *fun) -> double {
    if (fun->get_time() < 0.25)
        return 2.0;
    return 0.0;
    
    // return (29 * M_PI * M_PI * std::sin(5 * M_PI * fun->get_time()) +
    //           5 * M_PI * std::cos(5 * M_PI * fun->get_time())) *
    //          std::sin(2 * M_PI * p[0]) * std::sin(3 * M_PI * p[1]) *
    //          std::sin(4 * M_PI * p[2]);
};
// u0 (time independent)
static constexpr auto ini_cond_function = [](const dealii::Point<DIM> &p) -> double {
    // if (get_time() < 0.25)
    //     return 2.0;
    
    return 0.0;
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
#define MESH_PATH(N) "../../lab-08/mesh/mesh-cube-" + std::to_string(N + 1) + ".msh"
#else
#define MESH_PATH(N) "../../examples/gmsh/mesh-square-h" + std::to_string(N + 1) + ".msh"
#endif
#else
#define DOMAIN_START 0.0
#define DOMAIN_END 1.0
#endif

// Newman boundary functions: NOT USED IN NON LINEAR PROBLEMS
#define NM_B_EXISTS true
#define NEWMAN_BOUNDARIES_IDS {}
static std::vector<std::function<double (const dealii::Point<DIM>&, const Function<DIM> *fun)>> newman_f = {};
//     [](const Point<DIM> &p, const Function<DIM> *fun) -> double { 
        
//       return M_E*(std::exp(p[1])-1);
        
//     },
//     [](const Point<DIM> &p, const Function<DIM> *fun) -> double { 
        
//       return M_E*(std::exp(p[0])-1);
        
//     }
// };

// Dirichlet boundary functions:
#define DC_B_EXISTS true
// #define DIRICHLET_BOUNDARIES_IDS DoFTools::extract_boundary_dofs(dof_handler)
#define DIRICHLET_BOUNDARIES_IDS {0,1,2,3,4,5}
static std::vector<std::function<double (const dealii::Point<DIM>&, const Function<DIM> *fun)>> dirichlet_f = {
    
    [](const Point<DIM> &p, const Function<DIM> *fun) -> double {
        return 0.0;
    },

    [](const Point<DIM> &p, const Function<DIM> *fun) -> double {
        return 0.0;
    },

    [](const Point<DIM> &p, const Function<DIM> *fun) -> double {
        return 0.0;
    },

    [](const Point<DIM> &p, const Function<DIM> *fun) -> double {
        return 0.0;
    },
    
    [](const Point<DIM> &p, const Function<DIM> *fun) -> double {
        return 0.0;
    },

    [](const Point<DIM> &p, const Function<DIM> *fun) -> double {
        return 0.0;
    }
};


#define TOLERANCE 1e-6
#define N_MAX_ITERS 1000
#define SSOR_PARAM 1.0
// ******************************************************************************************

// Class representing the non-linear diffusion problem.
class ParametricHeat
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = DIM;

  // Function for the mu_0 coefficient.
  class FunctionMu0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      return lin_diffusion_function(p);
    }
  };

  // Function for the mu_1 coefficient.
  #if ! IS_LINEAR
  class FunctionMu1 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      return quad_diffusion_function(p);
    }
  };
  #endif

  class FunctionSigma : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      return reaction_function(p);
    }
  };

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {      
      return forcing_function(p,this);
    }
  };

  // Function for initial conditions.
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      return ini_cond_function(p);
    }
  };

  #if CHECK_CONVERGENCE
  class ExactSolution : public Function<dim>
  {
  public:
    // Constructor.
    ExactSolution() {}

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return exact_sol_function(p,this);
    }

    // Gradient evaluation.
    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;
      gradient_exact_sol_function(p,result,this);
      return result;
    }
  };
  #endif

  class ParametricFunction : public Function<dim>
  {
  public:
    // Constructor.
    ParametricFunction(std::function<double(const Point<dim>&, const Function<DIM> *fun)> lambdaFunction)
        : lambdaFunction(lambdaFunction) {}

    // Evaluation.
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      return lambdaFunction(p,this);
    }
    private:
      std::function<double(const Point<dim>&, const Function<DIM> *fun)> lambdaFunction;

  };

  class FunctionG : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };
  FunctionG function_g;

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  ParametricHeat(const unsigned int &N_,
                const unsigned int &r_,
                const double &      T_,
                const double &      deltat_,
                const double &      theta_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , N(N_)
    , r(r_)
    , deltat(deltat_)
    , theta(theta_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

  #if CHECK_CONVERGENCE
  // Compute the error.
  double compute_error(const VectorTools::NormType &norm_type);
  #endif


protected:
   #if IS_LINEAR
   // Assemble the mass and stiffness matrices.
  void assemble_matrices();

  // Assemble the right-hand side of the problem.
  void assemble_rhs(const double &time);

  // Solve the problem for one time step.
  void solve_time_step();

  #endif

  #if ! IS_LINEAR
  // Assemble the tangent problem.
  void
  assemble_system();

  // Solve the linear system associated to the tangent problem.
  void
  solve_linear_system();

  // Solve the problem for one time step using Newton's method.
  void
  solve_newton();
  #endif

  // Output.
  void
  output(const unsigned int &time_step, const double &time) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // mu_0 coefficient.
  FunctionMu0 mu_0;

  // mu_1 coefficient.
  #if ! IS_LINEAR
  FunctionMu1 mu_1;
  #endif

  // sigma coefficient.
  FunctionSigma reaction_coefficient;

  // Forcing term.
  ForcingTerm forcing_term;

  // Initial conditions.
  FunctionU0 u_0;

  #if CHECK_CONVERGENCE
  // Exact solution
  ExactSolution exact_solution;
  #endif

  // Current time.
  double time;

  // Final time.
  const double T;

  // Boundary conditions.
  // Dirichlet
  std::vector<unsigned int> dirichlet_ids = DIRICHLET_BOUNDARIES_IDS;
  std::vector<ParametricFunction> dirichlet_functions;

  // Newman
  std::vector<unsigned int> newman_ids = NEWMAN_BOUNDARIES_IDS;
  std::vector<ParametricFunction> newman_functions;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh refinement.
  const unsigned int N;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;
  std::unique_ptr<Quadrature<dim-1>> quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  #if IS_LINEAR
  // Mass matrix M / deltat.
  TrilinosWrappers::SparseMatrix mass_matrix;

  // Stiffness matrix A.
  TrilinosWrappers::SparseMatrix stiffness_matrix;

  // Matrix on the left-hand side (M / deltat + theta A).
  TrilinosWrappers::SparseMatrix lhs_matrix;

  // Matrix on the right-hand side (M / deltat - (1 - theta) A).
  TrilinosWrappers::SparseMatrix rhs_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::Vector system_rhs;
  #else
  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;

  // Increment of the solution between Newton iterations.
  TrilinosWrappers::MPI::Vector delta_owned;
  #endif

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // System solution at previous time step.
  // Last time we didn't need to store it
  TrilinosWrappers::MPI::Vector solution_old; // u_n

private:
  void create_boundary_fun(){

    for (auto f : dirichlet_f){
      ParametricFunction fun(f);
      dirichlet_functions.push_back(fun);
    }
    if (dirichlet_ids.size() != dirichlet_functions.size()) {
      std::cerr << "Size of dirichlet_ids and dirichlet_functions are not equal";
      throw 1;
    }
    for (auto f : newman_f){
      ParametricFunction fun(f);
      newman_functions.push_back(fun);
    }
    if (newman_ids.size() != newman_functions.size()) {
      std::cerr << "Size of newman_ids and newman_functions are not equal";
      throw 1;
    }
  };
};

#endif