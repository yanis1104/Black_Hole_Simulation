#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <SFML/Graphics.hpp>

#define PI (3.14159265358979323846)

__global__ void _create_black_hole(unsigned char* d_image, int3* background, int2 dimensions, int3* res, float2 bh_pos);
void compute_black_hole(int2& dimensions, dim3& block, dim3& grid, int3* d_pixel_results, int3* d_background_absolute, float2& BH_pos, sf::Image& image);