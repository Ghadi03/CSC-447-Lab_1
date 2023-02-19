#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 1000
#define HEIGHT 1000
#define MAX_ITERATIONS 1000

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = HEIGHT / size;
    int remaining_rows = HEIGHT % size;

    int* local_mandelbrot = (int*)malloc(sizeof(int) * WIDTH * rows_per_process);

    // calculate starting row for this process
    int start_row = rank * rows_per_process;
    if (rank < remaining_rows) {
        rows_per_process++;
        start_row += rank;
    } else {
        start_row += remaining_rows;
    }

    for (int i = 0; i < rows_per_process; i++) {
        int y = start_row + i;
        for (int x = 0; x < WIDTH; x++) {
            double zx = 0.0;
            double zy = 0.0;
            double cx = ((double)x - WIDTH/2) * 4.0/WIDTH;
            double cy = ((double)y - HEIGHT/2) * 4.0/WIDTH;
            int iterations = 0;
            while (zx*zx + zy*zy < 4 && iterations < MAX_ITERATIONS) {
                double new_zx = zx*zx - zy*zy + cx;
                double new_zy = 2*zx*zy + cy;
                zx = new_zx;
                zy = new_zy;
                iterations++;
            }
            local_mandelbrot[i*WIDTH + x] = iterations;
        }
    }

    int* global_mandelbrot = NULL;
    if (rank == 0) {
        global_mandelbrot = (int*)malloc(sizeof(int) * WIDTH * HEIGHT);
    }

    MPI_Gather(local_mandelbrot, WIDTH * rows_per_process, MPI_INT, global_mandelbrot, WIDTH * rows_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // output the mandelbrot set as a PGM image
        printf("P2\n%d %d\n%d\n", WIDTH, HEIGHT, MAX_ITERATIONS);
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                printf("%d ", global_mandelbrot[i*WIDTH + j]);
            }
            printf("\n");
        }
        free(global_mandelbrot);
    }

    free(local_mandelbrot);

    MPI_Finalize();
    return 0;
}
