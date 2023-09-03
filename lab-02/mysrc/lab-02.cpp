#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "Poisson1D.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  // Use a table to show the results on terminal and study the order of convergency
  ConvergenceTable table;

  const std::vector<unsigned int> N_values = {9, 19, 39, 79, 159, 319};
  const unsigned int              degree   = 2;

  // Write the convergence values into convergence.csv
  std::ofstream convergence_file("convergence.csv");
  convergence_file << "h,eL2,eH1" << std::endl;

  for (const unsigned int &N : N_values)
    {
      Poisson1D problem(N, degree);

      problem.setup();
      problem.assemble();
      problem.solve();
      problem.output();

      const double h        = 1.0 / (N + 1.0); // Important to remember .0 otherwise it will do integer division
      const double error_L2 = problem.compute_error(VectorTools::L2_norm);
      const double error_H1 = problem.compute_error(VectorTools::H1_norm);

      table.add_value("h", h);
      table.add_value("eL2", error_L2);
      table.add_value("eH1", error_H1);

      convergence_file << h << "," << error_L2 << "," << error_H1 << std::endl;
    }
  
  // Estimate the convergence rate directly with the dealii library
  // ConvergenceTable::reduction_rate_log2 only if you double the N at each iteration: 10 20 40 ...
  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);

  // Enable scientific notation for the print out of numbers
  table.set_scientific("eL2", true);
  table.set_scientific("eH1", true);

  table.write_text(std::cout);

  return 0;
}