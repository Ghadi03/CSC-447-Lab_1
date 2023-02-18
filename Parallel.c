#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 1000
#define XMIN -2.0
#define XMAX 1.0
#define YMIN -1.5
#define YMAX 1.5

int mandelbrot(double x, double y) {
    double cx = x, cy = y;
    double zx = 0.0, zy = 0.0;
    int i;
    for (i = 0; i < MAX_ITER; i++) {
        double zx_new = zx * zx - zy * zy + cx;
        double zy_new = 2 * zx * zy + cy;
        zx = zx_new;
        zy = zy_new;
        if (sqrt(zx * zx + zy * zy) > 2.0) {
            return i;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = 640, height = 480;
    int x, y;
    double dx = (XMAX - XMIN) / width;
    double dy = (YMAX - YMIN) / height;
    int *counts = (int*)malloc(sizeof(int) * size);
    int *displs = (int*)malloc(sizeof(int) * size);
    int *buffer = (int*)malloc(sizeof(int) * width * height);

    for (int i = 0; i < size; i++) {
        counts[i] = (height / size) * width;
        displs[i] = (height / size) * i * width;
    }
    counts[size-1] += (height % size) * width;

    int my_count = counts[rank];
    int my_displ = displs[rank];
    for (y = 0; y < height / size; y++) {
        for (x = 0; x < width; x++) {
            double cx = XMIN + x * dx;
            double cy = YMIN + (my_displ / width + y) * dy;
            buffer[my_displ + y * width + x] = mandelbrot(cx, cy);
        }
    }

    MPI_Gatherv(buffer + my_displ, my_count, MPI_INT, buffer, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *fp = fopen("mandelbrot.ppm", "wb");
        fprintf(fp, "P6\n%d %d\n255\n", width, height);
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                int n = buffer[y * width + x];
                unsigned char r = n % 256;
                unsigned char g = n % 256;
                unsigned char b = n % 256;
                fputc(r, fp);
                fputc(g, fp);
                fputc(b, fp);
            }
        }
        fclose
