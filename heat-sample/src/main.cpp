#include <fstream>
#include "ParametricHeat.hpp"


// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int               mpi_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  #ifdef H_VALUES
  const std::vector<double> h_values = H_VALUES;
  #endif
  const unsigned int N      = N_VALUES;
  const std::vector<unsigned int> degree_values   = P_GRADES;

  constexpr double T     = FIN_TIME;
  constexpr double theta = THETA;

  const std::vector<double> deltat_vector = dt_VALUES;

  for (const unsigned int &degree : degree_values) {
    #if CHECK_CONVERGENCE
    std::vector<double> errors_L2;
    std::vector<double> errors_H1;
    #endif

    for (const auto &deltat : deltat_vector)
      {
        ParametricHeat problem(N, degree, T, deltat, theta);

        problem.setup();
        problem.solve();

        #if CHECK_CONVERGENCE
        errors_L2.push_back(problem.compute_error(VectorTools::L2_norm));
        errors_H1.push_back(problem.compute_error(VectorTools::H1_norm));
        #endif
      }

    // Print the errors and estimate the convergence order.
    #if CHECK_CONVERGENCE
    if (mpi_rank == 0)
      {
        std::cout << "==============================================="
                  << std::endl;

        std::ofstream convergence_file("convergence" + std::to_string(degree) + ".csv");
        convergence_file << "dt,eL2,eH1" << std::endl;

        for (unsigned int i = 0; i < deltat_vector.size(); ++i)
          {
            convergence_file << deltat_vector[i] << "," << errors_L2[i] << ","
                            << errors_H1[i] << std::endl;

            std::cout << std::scientific << "dt = " << std::setw(4)
                      << std::setprecision(2) << deltat_vector[i];

            std::cout << std::scientific << " | eL2 = " << errors_L2[i];

            // Estimate the convergence order.
            if (i > 0)
              {
                const double p =
                  std::log(errors_L2[i] / errors_L2[i - 1]) /
                  std::log(deltat_vector[i] / deltat_vector[i - 1]);

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
                  std::log(deltat_vector[i] / deltat_vector[i - 1]);

                std::cout << " (" << std::fixed << std::setprecision(2)
                          << std::setw(4) << p << ")";
              }
            else
              std::cout << " (  - )";

            std::cout << "\n";
          }
      }
    #endif // CHECK_CONVERGENCE
  }

  return 0;
}