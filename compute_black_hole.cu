#include "compute_black_hole.cuh"

__global__ void _create_black_hole(unsigned char* d_image, int3* background, int2 dimensions, int3* res, float2 bh_pos)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;


    if (i >= dimensions.x || j >= dimensions.y)
        return;

    float length = 4.0; //5.0 distance obervateur - trou noir
    float radius = 0.1; //0.075 rayon de la sphère
    float dt = 0.001;
    float maxDistance = 1.00001; //1.00001 distance max à parcourir par le photon pour determiner si sa trajectoire est stable

    float radDistL = 0.0;
    float iPhoton = 0.0;
    float jPhoton = 0.0;

    float alpha = 0.0;
    float beta = 0.0;
    float x = 1.0;
    float y = 0.0;
    float t = 0.0;

    float vx = 0.0;
    float vy = 0.0;
    float ax = 0.0;
    float ay = 0.0;

    float dphi = 0.0;
    float pAlpha = 0.0;

    alpha = sqrt(pow(i - bh_pos.x, 2) + pow(bh_pos.y - j, 2)) * length / (dimensions.x);
    beta = atan2(bh_pos.y - j, i - bh_pos.x);
    x = 1.0; //1.0
    y = 0.0; //0.0
    t = 0.0;
    vx = -cos(alpha);
    vy = sin(alpha);
    while (sqrt(x * x + y * y) < maxDistance && sqrt(x * x + y * y) > radius) {
        t += dt;
        x += dt * vx;
        y += dt * vy;
        radDistL = sqrt(x * x + y * y);
        dphi = (x * vy - y * vx) / radDistL / radDistL;
        ax = (-3 / 2) * radius * (dphi * dphi) * x / radDistL;
        ay = (-3 / 2) * radius * (dphi * dphi) * y / radDistL;
        vx += dt * ax;
        vy += dt * ay;
    }
    if (radDistL >= maxDistance) {
        pAlpha = atan2(y, 1 - x);
        iPhoton = bh_pos.x + pAlpha * cos(beta) * dimensions.x / PI / 2;
        jPhoton = bh_pos.y - pAlpha * sin(beta) * dimensions.y / PI;

        iPhoton = round(iPhoton);
        jPhoton = round(jPhoton);

        while (iPhoton < 0)
            iPhoton = iPhoton + dimensions.x;
        while (iPhoton >= dimensions.x)
            iPhoton = iPhoton - dimensions.x;
        while (jPhoton < 0)
            jPhoton = jPhoton + dimensions.y;
        while (jPhoton >= dimensions.y)
            jPhoton = jPhoton - dimensions.y;
        res[i * dimensions.y + j] = background[(int)iPhoton * dimensions.y + (int)jPhoton];
        d_image[(j * dimensions.x + i) * 4 + 0] = background[(int)iPhoton * dimensions.y + (int)jPhoton].x;
        d_image[(j * dimensions.x + i) * 4 + 1] = background[(int)iPhoton * dimensions.y + (int)jPhoton].y;
        d_image[(j * dimensions.x + i) * 4 + 2] = background[(int)iPhoton * dimensions.y + (int)jPhoton].z;
    }
    else {
        res[i * dimensions.y + j].x = 0;
        res[i * dimensions.y + j].y = 0;
        res[i * dimensions.y + j].z = 0;
        d_image[(j * dimensions.x + i) * 4 + 0] = 0;
        d_image[(j * dimensions.x + i) * 4 + 1] = 0;
        d_image[(j * dimensions.x + i) * 4 + 2] = 0;
    }
}

void compute_black_hole(int2 &dimensions, dim3 &block, dim3 &grid, int3* d_pixel_results, int3* d_background_absolute, float2 &BH_pos, sf::Image &image)
{
    unsigned char* d_image;
    cudaMalloc(&d_image, image.getSize().x * image.getSize().y * 4);
    cudaMemcpy(d_image, image.getPixelsPtr(), image.getSize().x * image.getSize().y * 4, cudaMemcpyHostToDevice);

    _create_black_hole << <grid, block >> > (d_image, d_background_absolute, dimensions, d_pixel_results, BH_pos);
    cudaDeviceSynchronize();
    int3* pixel_results = new int3[dimensions.x * dimensions.y];
    cudaMemcpy(pixel_results, d_pixel_results, dimensions.x * dimensions.y * sizeof(int3), cudaMemcpyDeviceToHost);

    cudaMemcpy((void*)image.getPixelsPtr(), d_image, image.getSize().x * image.getSize().y * 4, cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    delete[] pixel_results;
}