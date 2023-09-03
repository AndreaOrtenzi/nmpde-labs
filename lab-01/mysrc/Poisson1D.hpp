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
    static constexpr unsigned int dim = 1;

    // *************** Functions ***************
    // Diffusion coefficient.
    // In deal.ii, functions are implemented by deriving the dealii::Function
    // class, which provides an interface for the computation of function values
    // and their derivatives.
    class DiffusionCoefficient : public Function<dim>
    {
    public:
        DiffusionCoefficient(){}

        // Evaluation.
        virtual double value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const
        {
            return 1.0;
        }
    };

    // Forcing term.
    class ForcingTerm : public Function<dim>
    {
    public:
        ForcingTerm() {}

        // Evaluation.
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const
        {
            if (p[0] <= 1.0 / 8 || p[0] > 1.0 / 4.0)
                return 0.0;
            else
                return -1.0;
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

protected:
    // N+1 is the number of elements.
    const unsigned int N;

    // Polynomial degree.
    const unsigned int r;


    // *************** Setup ***************

    // Mesh is stored with a dealii class called Triangulation.
    Triangulation<dim> mesh;

    // Finite element space.
    // We use a unique_ptr here so that we can choose the type and degree of the
    // finite elements at runtime (the degree is a constructor parameter). 
    // The class FiniteElement<dim> is an abstract class from which all types of finite elements implemented by deal.ii inherit.
    // Different types of FE means different degree but also FE on cubes, triangle ...
    std::unique_ptr<FiniteElement<dim>> fe;

    // Quadrature formula.
    // Quadrature is an abstract class.
    // We use a unique_ptr here so that we can choose the type and order of the
    // quadrature formula at runtime (the order is a constructor parameter).
    std::unique_ptr<Quadrature<dim>> quadrature;

    // DoF handler. concrete class
    DoFHandler<dim> dof_handler;

    // System matrix.
    SparseMatrix<double> system_matrix;

    // System right-hand side.
    Vector<double> system_rhs;

    // System solution.
    Vector<double> system_solution;

    // Sparsity pattern.
    // It's computationally more advantageous to build a matrix in 2 steps
    // First decide which elements are not zero (sparsity pattern) then build the matrix
    // Reason comes more evident in parallel systems and when you have to build a matrix multiple times (Time dep problem)
    SparsityPattern sparsity_pattern;

    // *************** Assemble ***************
    // Diffusion coefficient.
    DiffusionCoefficient diffusion_coefficient;

    // Forcing term.
    ForcingTerm forcing_term;

};

#endif