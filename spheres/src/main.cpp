#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "ParametricPoisson.hpp"
// Main function.

#if GET_TIME
#include <chrono>
#include <string>
#endif

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  std::vector<unsigned int> N_values = N_VALUES;
  const std::vector<unsigned int> degree_values   = P_GRADES;
  const std::vector<int> epsilon_values   = P_GRADES;

  // #if CHECK_CONVERGENCE
  // std::vector<double> errors_L2;
  // std::vector<double> errors_H1;
  // ConvergenceTable table;
  // #endif
  #if GET_TIME
  using clock = std::chrono::steady_clock;
  using unitOfTime = std::chrono::duration<double, std::milli>;
  const std::string unitTimeStr = " ms";

  std::chrono::time_point<clock> begin,setup,assemble,final,solver;
  double total = 0.0, setup_tot = 0.0, assemble_tot = 0.0, solver_tot = 0.0,writing_tot;

  std::ofstream timing_file_DN("executiontimeDN.csv",std::ios_base::app);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0){
    // std::ofstream timing_file_ND("executiontimeDN.csv");
    timing_file_DN << "mpiProc,epsilon,deg,N,time (" << unitTimeStr << " ),setup,assemble,solve" << std::endl;
  }
  // timing_file_ND << "N,deg,time" << std::endl;
  #endif

  for (const unsigned int &epsilon : epsilon_values) {
    for (const unsigned int &degree : degree_values) {
      // #if CHECK_CONVERGENCE
      // std::ofstream convergence_file("convergence" + std::to_string(degree) + ".csv");
      // std::ofstream convergence_file_no_lib("myconvergence" + std::to_string(degree) + ".csv");
      // convergence_file << "h,eL2,eH1" << std::endl;
      // convergence_file_no_lib << "h,eL2,eH1" << std::endl;
      // errors_L2.clear();
      // errors_H1.clear();
      // #endif

      for (const unsigned int &N : N_values)
        {
          #if GET_TIME
          total = 0.0;
          setup_tot = 0.0;
          assemble_tot = 0.0;
          writing_tot = 0.0;
          solver_tot = 0.0;

          for (unsigned int i = 0; i < iTT; ++i) {
          #endif
            ParametricPoisson problem(N, degree,epsilon);

          #if GET_TIME
            begin = clock::now();
          #endif

            problem.setup();
          #if GET_TIME
            setup = clock::now();
          #endif
            problem.assemble();
          #if GET_TIME
            assemble = clock::now();
          #endif
            problem.solve();
          #if GET_TIME
            solver = clock::now();
          #endif

            // #if CHECK_CONVERGENCE
            // const double h        = 1.0 / (N + 1.0);
            // const double error_L2 = problem.compute_error(VectorTools::L2_norm);
            // const double error_H1 = problem.compute_error(VectorTools::H1_norm);

            // table.add_value("h", h);
            // table.add_value("L2", error_L2);
            // table.add_value("H1", error_H1);

            // convergence_file << h << "," << error_L2 << "," << error_H1 << std::endl;

            // errors_L2.push_back(error_L2);
            // errors_H1.push_back(error_H1);

            // convergence_file_no_lib << h << "," << error_L2 << "," << error_H1 << std::endl;
            // #endif
            problem.output();
            
          #if GET_TIME
            final = clock::now();
            total += std::chrono::duration_cast<unitOfTime>(final - begin).count();
            setup_tot += std::chrono::duration_cast<unitOfTime>(setup - begin).count();
            assemble_tot += std::chrono::duration_cast<unitOfTime>(assemble - setup).count();
            solver_tot += std::chrono::duration_cast<unitOfTime>(solver - assemble).count();
            writing_tot += std::chrono::duration_cast<unitOfTime>(final - solver).count();
          
          }
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0){
            total = total/iTT;
            setup_tot = setup_tot / (iTT);
            // assemble_tot = assemble_tot / (iTT * total) * 100;
            assemble_tot = assemble_tot / (iTT) ;
            solver_tot = solver_tot / (iTT );
            writing_tot = writing_tot / (iTT);
            std::cout << "Mesh size: " << N+1 << " p degree: " << degree << " took on average (" << iTT << ") on rank 0: " << total << unitTimeStr << ", setup: " << setup_tot << ", assemble: " << assemble_tot << ", solver: " << solver_tot << ", writing: " << writing_tot << std::endl;
            timing_file_DN << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << "," << epsilon << "," << degree << "," << N+1 << "," << total << "," << setup_tot << "," << assemble_tot << "," << solver_tot << std::endl;
          }
          #endif
        }

      // #if CHECK_CONVERGENCE
      // table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
      // table.set_scientific("L2", true);
      // table.set_scientific("H1", true);
      // std::cout << "Try with degree = " << degree << std::endl;
      // table.write_text(std::cout);
      
      // std::cout << "=========== my estimation ===================" << std::endl;
      // for (unsigned i = 0; i < N_values.size();++i){
      //   const double h        = 1.0 / (N_values[i] + 1.0);

      //   std::cout << std::scientific << "h = " << std::setw(4)
      //                 << std::setprecision(2) << h;

      //   std::cout << std::scientific << " | eL2 = " << errors_L2[i];

      //   // Estimate the convergence order.
      //   if (i > 0)
      //     {
      //       const double p =
      //         std::log(errors_L2[i] / errors_L2[i - 1]) /
      //         std::log(h * (N_values[i-1] + 1.0));

      //       std::cout << " (" << std::fixed << std::setprecision(2)
      //                 << std::setw(4) << p << ")";
      //     }
      //   else
      //     std::cout << " (  - )";

      //   std::cout << std::scientific << " | eH1 = " << errors_H1[i];

      //   // Estimate the convergence order.
      //   if (i > 0)
      //     {
      //       const double p =
      //         std::log(errors_H1[i] / errors_H1[i - 1]) /
      //         std::log(h * (N_values[i-1] + 1.0));

      //       std::cout << " (" << std::fixed << std::setprecision(2)
      //                 << std::setw(4) << p << ")";
      //     }
      //   else
      //     std::cout << " (  - )";

      //   std::cout << "\n";
      // }
      // #endif // CHECK_CONVERGENCE
    }
  }

  return 0;
}