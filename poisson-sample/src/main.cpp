#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "ParametricPoisson.hpp"
// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  
  #ifdef H_VALUES
  const std::vector<double> h_values = H_VALUES;
  #endif
  std::vector<unsigned int> N_values = N_VALUES;
  const std::vector<unsigned int> degree_values   = P_GRADES;

  #if CHECK_CONVERGENCE
  std::vector<double> errors_L2;
  std::vector<double> errors_H1;
  ConvergenceTable table;
  #endif

  for (const unsigned int &degree : degree_values) {
    #if CHECK_CONVERGENCE
    std::ofstream convergence_file("convergence" + std::to_string(degree) + ".csv");
    std::ofstream convergence_file_no_lib("myconvergence" + std::to_string(degree) + ".csv");
    convergence_file << "h,eL2,eH1" << std::endl;
    convergence_file_no_lib << "h,eL2,eH1" << std::endl;
    errors_L2.clear();
    errors_H1.clear();
    #endif

    for (const unsigned int &N : N_values)
      {
        ParametricPoisson problem(N, degree);

        problem.setup();
        problem.assemble();
        problem.solve();
        problem.output();

        #if CHECK_CONVERGENCE
        const double h        = 1.0 / (N + 1.0);
        const double error_L2 = problem.compute_error(VectorTools::L2_norm);
        const double error_H1 = problem.compute_error(VectorTools::H1_norm);

        table.add_value("h", h);
        table.add_value("L2", error_L2);
        table.add_value("H1", error_H1);

        convergence_file << h << "," << error_L2 << "," << error_H1 << std::endl;

        errors_L2.push_back(error_L2);
        errors_H1.push_back(error_H1);

        convergence_file_no_lib << h << "," << error_L2 << "," << error_H1 << std::endl;
        #endif
      }

    #if CHECK_CONVERGENCE
    table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
    table.set_scientific("L2", true);
    table.set_scientific("H1", true);
    std::cout << "Try with degree = " << degree << std::endl;
    table.write_text(std::cout);
    
    std::cout << "=========== my estimation ===================" << std::endl;
    for (unsigned i = 0; i < N_values.size();++i){
      const double h        = 1.0 / (N_values[i] + 1.0);

      std::cout << std::scientific << "h = " << std::setw(4)
                    << std::setprecision(2) << h;

      std::cout << std::scientific << " | eL2 = " << errors_L2[i];

      // Estimate the convergence order.
      if (i > 0)
        {
          const double p =
            std::log(errors_L2[i] / errors_L2[i - 1]) /
            std::log(h * (N_values[i-1] + 1.0));

          std::cout << " (" << std::fixed << std::setprecision(2)
                    << std::setw(4) << p << ")";
        }
      else
        std::cout << " (  - )";

      std::cout << std::scientific << " | eH1 = " << errors_H1[i];

      // Estimate the convergence order.
      if (i > 0)
        {
          const double p =
            std::log(errors_H1[i] / errors_H1[i - 1]) /
            std::log(h * (N_values[i-1] + 1.0));

          std::cout << " (" << std::fixed << std::setprecision(2)
                    << std::setw(4) << p << ")";
        }
      else
        std::cout << " (  - )";

      std::cout << "\n";
    }
    #endif // CHECK_CONVERGENCE
  }

  return 0;
}