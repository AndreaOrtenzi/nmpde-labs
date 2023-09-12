#ifndef POISSON_1D_HPP
#define POISSON_1D_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>


#include <fstream>
#include <iostream>
#include <algorithm>

#include "sphere.hpp"

using namespace dealii;

// ******************************************************************************************
// Problem parameters
// #define N_VALUES {4, 9, 19, 39}
#define DIM 3
#define H_VALUES {0.05}// 0.1, 0.05, 0.025, 0.0125}
#define N_VALUES {4,9} // 4, 9, 19, 39}
#define EPSILON 4
#define P_GRADES {1,2}
#define WRITE_ON_FILE false

// NUM_ITER_TO_TIME
#define iTT 10
#define GET_TIME true


// mus
static constexpr auto diffusion_function = [](const dealii::Point<DIM> &p) -> double { 
    
    const std::vector<Sphere<DIM>> spheres = {
        { { 0.10000000000067776, 0.10000000000000485, 0.10000000000002289 }, 0.1 },
        { { 0.842484606531941, 0.1500000000000169, 0.15000000000000135 }, 0.15 },
        { { 0.7999832973018323, 0.7999962223715145, 0.41313791257885574 }, 0.2 },
        { { 0.2500000000002333, 0.3219940474925781, 0.7494309059799914 }, 0.25 }
    };

    for (unsigned int i=0; i<spheres.size();++i){
      if (spheres[i].is_in(p))
        return std::pow(10,EPSILON);
    }
    
    return 1.0;
};

// sigma (Not used)
static constexpr auto reaction_function = [](const dealii::Point<DIM> &p) -> double {
    
        return 0.0;
};

// f
static constexpr auto forcing_function = [](const dealii::Point<DIM> &p) -> double {
    
        return 1.0;// (std::exp(p[0])-1)*(std::exp(p[1])-1);
};

// exact solution:
#define CHECK_CONVERGENCE false
#if CHECK_CONVERGENCE
static constexpr auto exact_sol_function = [](const dealii::Point<DIM> &p) -> double {
    
    return (std::exp(p[0])-1)*(std::exp(p[1])-1);
};
// gradient exact solution:
static constexpr auto gradient_exact_sol_function = [](const dealii::Point<DIM> &p, dealii::Tensor<1, DIM> &result) {

    result[0] =
        (std::exp(p[1])-1)*(std::exp(p[0]));
    result[1] =
        (std::exp(p[0])-1)*(std::exp(p[1]));
};
#endif


// Mesh:
#define IS_MESH_FROM_FILE true
#define IS_QUADRILATERAL_MESH false

#if IS_MESH_FROM_FILE
#define MESH_PATH(N) "../mesh/mesh-cube-" + std::to_string(N + 1) + ".msh"
#else
#define DOMAIN_START 0.0
#define DOMAIN_END 1.0
#endif

// Newman boundary functions:
#define NM_B_EXISTS true
#define NEWMAN_BOUNDARIES_IDS {}
static std::vector<std::function<double (const dealii::Point<DIM>&)>> newman_f = {};

// Dirichlet boundary functions:
#define DC_B_EXISTS true
#define DIRICHLET_BOUNDARIES_IDS {} // defined by hand
static std::vector<std::function<double (const dealii::Point<DIM>&)>> dirichlet_f = {};
//     [](const Point<DIM> &p) -> double {
//         return 0.0;
//     },
//     [](const Point<DIM> &p) -> double {
//         return 0.0;
//     },
//     [](const Point<DIM> &p) -> double {
//         return 0.0;
//     },
//     [](const Point<DIM> &p) -> double {
//         return 0.0;
//     },
//     [](const Point<DIM> &p) -> double {
//         return 0.0;
//     },
//     [](const Point<DIM> &p) -> double {
//         return 0.0;
//     }
// };



// Solve
/*
0 -> no prec
1 -> Jacobi
2 -> GS
3 -> SOR
4 -> SSOR
*/
#define PRECODITIONER 4

#if PRECODITIONER == 2
#define R_SOR 1 // Don't touch here
#else
// Change here
#define R_SOR 1
#endif

// ******************************************************************************************


/**
 * Class managing the differential problem.
 */
class ParametricPoisson
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = DIM;


  class ParametricFunction : public Function<dim>
  {
  public:
    // Constructor.
    ParametricFunction(std::function<double(const Point<dim>&)> lambdaFunction)
        : lambdaFunction(lambdaFunction) {}

    // Evaluation.
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      return lambdaFunction(p);
    }
    private:
      std::function<double(const Point<dim>&)> lambdaFunction;

  };

  // Diffusion coefficient.
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    DiffusionCoefficient()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      return diffusion_function(p);
    }

  };

  class ForcingFunction : public Function<dim>
  {
  public:
    // Constructor.
    ForcingFunction()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
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
      return exact_sol_function(p);
    }

    // Gradient evaluation.
    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;
      gradient_exact_sol_function(p,result);
      return result;
    }
  };
  #endif

  // Constructor.
  ParametricPoisson(const unsigned int &N_, const unsigned int &r_)
    : N(N_)
    , r(r_) 
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0) {}

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

  // Compute the error.
  #if CHECK_CONVERGENCE
  double compute_error(const VectorTools::NormType &norm_type) const;
  #endif
  
protected:
  // N+1 is the number of elements.
  const unsigned int N;

  // Polynomial degree.
  const unsigned int r;

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Diffusion coefficient.
  DiffusionCoefficient diffusion_coefficient;


  // Forcing term.
  ForcingFunction forcing_term;

  // Dirichlet boundaries
  std::vector<unsigned int> dirichlet_ids = DIRICHLET_BOUNDARIES_IDS;
  std::vector<ParametricFunction> dirichlet_functions;

  // Newman boundaries
  std::vector<unsigned int> newman_ids = NEWMAN_BOUNDARIES_IDS;
  std::vector<ParametricFunction> newman_functions;

  // Triangulation.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution.
  TrilinosWrappers::MPI::Vector solution;

  // Parallel output stream.
  ConditionalOStream pcout;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

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