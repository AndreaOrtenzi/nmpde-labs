
#include <vector>
#include <deal.II/base/quadrature_lib.h>


template <unsigned int DIM>
class Sphere
{
public:
    Sphere(std::vector<double> center_, double radius_): radius(radius_),center(center_){};

    bool is_in(const dealii::Point<DIM> &p) const {
        
        // Calculate the squared distance between the point and the sphere's center
        double squaredDistance = 0.0;
        for (int i = 0; i < DIM; ++i)
        {
            double diff = p(i) - center[i];
            squaredDistance += diff * diff;
        }

        // Check if the squared distance is less than or equal to the squared radius
        return squaredDistance <= (radius * radius);
        
    };


    const double radius;
    const std::vector<double> center;



};