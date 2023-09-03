
# How to open the results in paraview


- Open paraview from terminal with `paraview`

- Open the file in the build folder called output-*.vtk

- Click to apply

- Since it's a 1D problem the normal view is not very clear, so use a filter to plot the results on a graph

- Select the object in the pipeline (top left box)

- click ctrl+space to search for the right filter

- search for Plot over line and press enter

- select apply again for this new object and sbam it's done


# How to start the container


- `docker run --name hpc-courses -v /media/andrea/Files/Git-dirs/nmpde-labs:/home/jellyfish/shared-folder -it fudezhou/hpc_courses`

- `docker start hpc-courses`

- `docker container exec -it hpc-courses /bin/bash`

# How to build the lab-01:


- `cd` in the current directory

- `mkdir build`

- `cd build/`  

- `module load gcc-glibc dealii`

- `cmake ..`

- `make`

