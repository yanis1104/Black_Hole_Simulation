#include "rotate_background.cuh"

__device__ void update_background_last_column(unsigned char* d_image, int3* background, int3* temp, int2 dimensions)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = (x * dimensions.y + y) * 4;

    if (y >= dimensions.y)
        return;
    background[(dimensions.x - 1) * dimensions.y + y] = temp[0 * dimensions.y + y];
    d_image[index + 0] = background[(x + 1) * dimensions.y + y].x;
    d_image[index + 1] = background[(x + 1) * dimensions.y + y].y;
    d_image[index + 2] = background[(x + 1) * dimensions.y + y].z;
}

__device__ void update_background(unsigned char* d_image, int3* background, int3* res, int2 dimensions)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = (x * dimensions.y + y) * 4;

    if (x >= dimensions.x || y >= dimensions.y)
        return;

    if (x + 1 < dimensions.x) {
        res[x * dimensions.y + y] = background[(x + 1) * dimensions.y + y];
        d_image[index + 0] = background[(x + 1) * dimensions.y + y].x;
        d_image[index + 1] = background[(x + 1) * dimensions.y + y].y;
        d_image[index + 2] = background[(x + 1) * dimensions.y + y].z;
    }
}

__global__ void move_background(unsigned char* d_image, int3* background_absolute, int3* res, int2 dimensions)
{
    int3* temp = background_absolute;

    update_background(d_image, background_absolute, res, dimensions);
    update_background_last_column(d_image, background_absolute, temp, dimensions);
    
}

void move_background_to_left(int3* d_background_absolute, sf::Image& image, sf::Clock& background_clock, int2 dimensions)
{
    unsigned char* d_image;
    dim3 block(16, 16);
    dim3 grid((dimensions.x + block.x - 1) / block.x, (dimensions.y + block.y - 1) / block.y);

    if (background_clock.getElapsedTime().asMilliseconds() >= 5) {
        int3* res;
        cudaMalloc((void**)&res, dimensions.x * dimensions.y * sizeof(int3));
        cudaMalloc(&d_image, image.getSize().x * image.getSize().y * 4);
        cudaMemcpy(d_image, image.getPixelsPtr(), image.getSize().x * image.getSize().y * 4, cudaMemcpyHostToDevice);
        move_background << <grid, block >> > (d_image, d_background_absolute, res, dimensions);
        cudaDeviceSynchronize();
        cudaMemcpy(d_background_absolute, res, dimensions.x * dimensions.y * sizeof(int3), cudaMemcpyDeviceToDevice);
        cudaFree(res);
        cudaMemcpy((void*)image.getPixelsPtr(), d_image, image.getSize().x * image.getSize().y * 4, cudaMemcpyDeviceToHost);
        cudaFree(d_image);
        background_clock.restart();
    }
}