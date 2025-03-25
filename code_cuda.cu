// ======================================================================================
// Gestion des librairies et des variables
// ======================================================================================
// Import des librairies
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>  // Pour mesurer le temps d'exécution
#include "lodepng.h"
#include <cuda_runtime.h>
#include <time.h>  // Pour mesurer le temps d'exécution

// Definition des variables
#define SIGMA_S 2.0   // Paramètre pour la composante spatiale
#define SIGMA_R 50.0  // Paramètre pour la composante de similitude d'intensité
#define KERNEL_SIZE 5 // Taille du noyau pour le filtrage

// Structure pour stocker les données d'une image
typedef struct {
    unsigned char *image; // Pointeur vers les pixels de l'image
    unsigned width, height; // Dimensions de l'image
} Image;


// ======================================================================================
// Chargement et sauvegarde de l'image
// ======================================================================================
// Fonction pour charger une image PNG à partir d'un fichier
void load_image(const char *filename, Image *img) {
    unsigned error = lodepng_decode32_file(&img->image, &img->width, &img->height, filename);
    if (error) {
        printf("Erreur lors du chargement de l'image: %s\n", lodepng_error_text(error));
        exit(1);
    }
}

// Fonction pour enregistrer une image PNG dans un fichier
void save_image(const char *filename, Image *img) {
    unsigned error = lodepng_encode32_file(filename, img->image, img->width, img->height);
    if (error) {
        printf("Erreur lors de l'enregistrement de l'image: %s\n", lodepng_error_text(error));
        exit(1);
    }
}


// ======================================================================================
// Filtre bilatéral en CUDA
// ======================================================================================
// Fonction CUDA pour appliquer le filtre bilatéral
__global__ void bilateral_filter_cuda(unsigned char *d_input, unsigned char *d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
 
    if (x >= width || y >= height) return; // Si le thread est en dehors des limites de l'image, on quitte

    int idx = 4 * (y * width + x); // Index du pixel dans l'image (r, g, b, a)
    float sum_r = 0, sum_g = 0, sum_b = 0, norm_factor = 0;
    float sigma_s2 = 2.0 * SIGMA_S * SIGMA_S;
    float sigma_r2 = 2.0 * SIGMA_R * SIGMA_R;
    int half_size = KERNEL_SIZE / 2;
 
    unsigned char r = d_input[idx], g = d_input[idx+1], b = d_input[idx+2];
 
    // Calcul du filtre bilatéral pour chaque pixel en utilisant ses voisins
    for (int i = -half_size; i <= half_size; i++) {
        for (int j = -half_size; j <= half_size; j++) {
            int yy = y + i;
            int xx = x + j;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) { // Vérifie si le voisin est dans l'image
                int neighbor_idx = 4 * (yy * width + xx);
                unsigned char nr = d_input[neighbor_idx], ng = d_input[neighbor_idx+1], nb = d_input[neighbor_idx+2];
                float intensity_diff = (r - nr) * (r - nr) + (g - ng) * (g - ng) + (b - nb) * (b - nb);
                float range_weight = expf(-intensity_diff / sigma_r2);  // Poids basé sur la différence d'intensité
                float spatial_weight = expf(-(i * i + j * j) / sigma_s2);  // Poids spatial
                float weight = spatial_weight * range_weight;
                sum_r += weight * nr;
                sum_g += weight * ng;
                sum_b += weight * nb;
                norm_factor += weight;
            }
        }
    }
 
    d_output[idx] = (unsigned char)(sum_r / norm_factor);
    d_output[idx+1] = (unsigned char)(sum_g / norm_factor);
    d_output[idx+2] = (unsigned char)(sum_b / norm_factor);
    d_output[idx+3] = d_input[idx+3]; // Conserve la composante alpha
}

// Fonction pour appliquer le filtre bilatéral avec CUDA
void bilateral_filter(Image *img) {
    int width = img->width;
    int height = img->height;
    size_t img_size = width * height * 4 * sizeof(unsigned char);
    unsigned char *d_input, *d_output;
    
    // Allocation de la mémoire sur le GPU
    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, img_size);
    cudaMemcpy(d_input, img->image, img_size, cudaMemcpyHostToDevice);
 
    // Dimensions des blocs et de la grille CUDA
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Lancement du noyau CUDA pour appliquer le filtre bilatéral
    bilateral_filter_cuda<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    
    // Copie des résultats du GPU vers le CPU
    cudaMemcpy(img->image, d_output, img_size, cudaMemcpyDeviceToHost);
    
    // Libération de la mémoire GPU
    cudaFree(d_input);
    cudaFree(d_output);
}


// ======================================================================================
// Fonction main pour exécuter le programme
// ======================================================================================
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s input.png output.png\n", argv[0]);
        return 1;
    }

    Image img;

    // Chargement de l'image
    load_image(argv[1], &img);

    // Démarrage du chronomètre
    clock_t start_time = clock();

    // Application du filtre bilatéral
    bilateral_filter(&img);

    // Arrêt du chronomètre
    clock_t end_time = clock();

    // Calcul du temps d'exécution en secondes
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Sauvegarde de l'image résultante
    save_image(argv[2], &img);

    // Affichage du temps d'exécution
    printf("Temps d'exécution : %.4f secondes\n", total_time);

    // Libération de la mémoire de l'image
    free(img.image);

    return 0;
}
