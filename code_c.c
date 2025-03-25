#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lodepng.h" // Pour la gestion des images PNG
 
#define SIGMA_S 2.0
#define SIGMA_R 50.0
#define KERNEL_SIZE 5
 
// Fonction pour charger une image PNG
typedef struct {
    unsigned char *image;
    unsigned width, height;
} Image;
 
void load_image(const char *filename, Image *img) {
    unsigned error = lodepng_decode32_file(&img->image, &img->width, &img->height, filename);
    if (error) {
        printf("Erreur lors du chargement de l'image: %s\n", lodepng_error_text(error));
        exit(1);
    }
}
 
void save_image(const char *filename, Image *img) {
    unsigned error = lodepng_encode32_file(filename, img->image, img->width, img->height);
    if (error) {
        printf("Erreur lors de l'enregistrement de l'image: %s\n", lodepng_error_text(error));
        exit(1);
    }
}
 
// Implémentation du filtre bilatéral en C
void bilateral_filter(Image *img) {
    int width = img->width;
    int height = img->height;
    unsigned char *output = (unsigned char *)malloc(width * height * 4);
 
    double spatial_weight[KERNEL_SIZE][KERNEL_SIZE];
    int half_size = KERNEL_SIZE / 2;
    double sigma_s2 = 2.0 * SIGMA_S * SIGMA_S;
    double sigma_r2 = 2.0 * SIGMA_R * SIGMA_R;
 
    for (int i = -half_size; i <= half_size; i++) {
        for (int j = -half_size; j <= half_size; j++) {
            spatial_weight[i + half_size][j + half_size] = exp(-(i * i + j * j) / sigma_s2);
        }
    }
 
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double sum_r = 0, sum_g = 0, sum_b = 0, norm_factor = 0;
            int idx = 4 * (y * width + x);
            unsigned char r = img->image[idx], g = img->image[idx+1], b = img->image[idx+2];
 
            for (int i = -half_size; i <= half_size; i++) {
                for (int j = -half_size; j <= half_size; j++) {
                    int yy = y + i;
                    int xx = x + j;
                    if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                        int neighbor_idx = 4 * (yy * width + xx);
                        unsigned char nr = img->image[neighbor_idx], ng = img->image[neighbor_idx+1], nb = img->image[neighbor_idx+2];
                        double intensity_diff = (r - nr) * (r - nr) + (g - ng) * (g - ng) + (b - nb) * (b - nb);
                        double range_weight = exp(-intensity_diff / sigma_r2);
                        double weight = spatial_weight[i + half_size][j + half_size] * range_weight;
                        sum_r += weight * nr;
                        sum_g += weight * ng;
                        sum_b += weight * nb;
                        norm_factor += weight;
                    }
                }
            }
 
            output[idx] = (unsigned char)(sum_r / norm_factor);
            output[idx+1] = (unsigned char)(sum_g / norm_factor);
            output[idx+2] = (unsigned char)(sum_b / norm_factor);
            output[idx+3] = img->image[idx+3];
        }
    }
    free(img->image);
    img->image = output;
}
 
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s input.png output.png\n", argv[0]);
        return 1;
    }
 
    Image img;
    load_image(argv[1], &img);
    bilateral_filter(&img);
    save_image(argv[2], &img);
 
    free(img.image);
    return 0;
}