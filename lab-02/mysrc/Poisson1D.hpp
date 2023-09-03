#ifndef POISSON_1D_HPP
#define POISSON_1D_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Poisson1D
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 1;

  // ************ Functions ************
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    DiffusionCoefficient()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const
    {
      return 1.0;
    }
  };

  // Forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    // Constructor.
    ForcingTerm()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const
    {
      // Points 3 and 4.
      return 4.0 * M_PI * M_PI * std::sin(2.0 * M_PI * p[0]);

      // Point 5.
      // if (p[0] < 0.5)
      //   return 0.0;
      // else
      //   return -std::sqrt(p[0] - 0.5);
    }
  };

  // Constructor.
  Poisson1D(const unsigned int &N_, const unsigned int &r_)
    : N(N_)
    , r(r_)
  {}

  // Initialization.
  void setup();

  // System assembly.
  void assemble();

  // System solution.
  void solve();

  // Output.
  void output() const;


  // ********************* Convergence analysis *********************
  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    // Constructor.
    ExactSolution()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const
    {
      // Points 3 and 4.
      return std::sin(2.0 * M_PI * p[0]);

      // Point 5.
      // if (p[0] < 0.5)
      //   return A * p[0];
      // else
      //   return A * p[0] + 4.0 / 15.0 * std::pow(p[0] - 0.5, 2.5);
    }

    // Since we'll compute the H1 norm we need to compute the derivative (gradient) of this function
    // Gradient evaluation.
    // deal.II requires this method to return a Tensor (not a double), i.e. a
    // dim-dimensional vector. In our case, dim = 1, so that the Tensor will in
    // practice contain a single number. Nonetheless, we need to return an
    // object of type Tensor.
    virtual Tensor<1, dim>
    gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const
    {
      Tensor<1, dim> result; // Tensor<1, ..> is a Vector

      // Points 3 and 4.
      result[0] = 2.0 * M_PI * std::cos(2.0 * M_PI * p[0]);

      // Point 5.
      // if (p[0] < 0.5)
      //   result[0] = A;
      // else
      //   result[0] = A + 2.0 / 3.0 * std::pow(p[0] - 0.5, 1.5);

      return result;
    }

    static constexpr double A = -4.0 / 15.0 * std::pow(0.5, 2.5);
  };

  // Compute the error.
  double compute_error(const VectorTools::NormType &norm_type) const;

  

protected:
  // N+1 is the number of elements.
  const unsigned int N;

  // Polynomial degree.
  const unsigned int r;

  
  DiffusionCoefficient diffusion_coefficient;
  ForcingTerm forcing_term;
  ExactSolution exact_solution;


  // Triangulation.
  Triangulation<dim> mesh;

  std::unique_ptr<FiniteElement<dim>> fe;
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // Sparsity pattern.
  SparsityPattern sparsity_pattern;

  // System matrix.
  SparseMatrix<double> system_matrix;

  // System right-hand side.
  Vector<double> system_rhs;

  // System solution.
  Vector<double> solution;
};

#endif