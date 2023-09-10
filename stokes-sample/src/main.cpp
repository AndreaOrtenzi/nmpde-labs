#include "ParametricStokes.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int N               = N_VALUES; // Since h = 0.1
  const unsigned int degree_velocity = V_GRADE;
  const unsigned int degree_pressure = P_GRADE;

  ParametricStokes problem(N, degree_velocity, degree_pressure);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}