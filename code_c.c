// ======================================================================================
// Gestion des librairies et des variables
// ======================================================================================
// Import des librairies
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lodepng.h" // Pour la gestion des images PNG
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
// Filtre bilatéral en C
// ======================================================================================
// Implémentation du filtre bilatéral en C
void bilateral_filter(Image *img) {
    // Déclaration des variables
    int width = img->width;
    int height = img->height;
    unsigned char *output = (unsigned char *)malloc(width * height * 4); // Tableau pour stocker l'image filtrée
    double spatial_weight[KERNEL_SIZE][KERNEL_SIZE];  // Poids spatial pour le noyau de filtrage
    int half_size = KERNEL_SIZE / 2;
    double sigma_s2 = 2.0 * SIGMA_S * SIGMA_S;  // Paramètre pour la distance spatiale
    double sigma_r2 = 2.0 * SIGMA_R * SIGMA_R;  // Paramètre pour la différence d'intensité
 
    // Calcul des poids spatiaux pour chaque pixel voisin dans le noyau
    for (int i = -half_size; i <= half_size; i++) {
        for (int j = -half_size; j <= half_size; j++) {
            spatial_weight[i + half_size][j + half_size] = exp(-(i * i + j * j) / sigma_s2);
        }
    }

    // Filtrage bilatéral sur chaque pixel de l'image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double sum_r = 0, sum_g = 0, sum_b = 0, norm_factor = 0;
            int idx = 4 * (y * width + x); // Index du pixel dans l'image (r, g, b, a)
            unsigned char r = img->image[idx], g = img->image[idx+1], b = img->image[idx+2];

            // Calcul du filtre bilatéral pour chaque pixel en utilisant ses voisins
            for (int i = -half_size; i <= half_size; i++) {
                for (int j = -half_size; j <= half_size; j++) {
                    int yy = y + i;
                    int xx = x + j;
                    if (xx >= 0 && xx < width && yy >= 0 && yy < height) {  // Vérifie si le voisin est dans l'image
                        int neighbor_idx = 4 * (yy * width + xx);
                        unsigned char nr = img->image[neighbor_idx], ng = img->image[neighbor_idx+1], nb = img->image[neighbor_idx+2];

                        // Calcul de la différence d'intensité entre le pixel courant et son voisin
                        double intensity_diff = (r - nr) * (r - nr) + (g - ng) * (g - ng) + (b - nb) * (b - nb);
                        double range_weight = exp(-intensity_diff / sigma_r2);  // Poids basé sur la différence d'intensité

                        // Poids spatial multiplié par le poids de la similarité d'intensité
                        double weight = spatial_weight[i + half_size][j + half_size] * range_weight;

                        sum_r += weight * nr;
                        sum_g += weight * ng;
                        sum_b += weight * nb;
                        norm_factor += weight;
                    }
                }
            }

            // Calcul de la couleur moyenne pondérée
            output[idx] = (unsigned char)(sum_r / norm_factor);
            output[idx+1] = (unsigned char)(sum_g / norm_factor);
            output[idx+2] = (unsigned char)(sum_b / norm_factor);
            output[idx+3] = img->image[idx+3]; // Conserve la composante alpha
        }
    }
    
    free(img->image);  // Libère la mémoire de l'ancienne image
    img->image = output;  // Met à jour l'image avec l'image filtrée
}


// ======================================================================================
// Fonction main pour éxécuter le programme
// ======================================================================================
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s input.png output.png\n", argv[0]);
        return 1;
    }

    Image img;
    
    // Mesure du temps d'exécution
    clock_t start_time, end_time;
    double total_time;

    // Chargement de l'image
    load_image(argv[1], &img);

    // Démarrage du chronomètre
    start_time = clock();

    // Application du filtre bilatéral
    bilateral_filter(&img);

    // Arrêt du chronomètre
    end_time = clock();

    // Calcul du temps d'exécution en secondes
    total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Sauvegarde de l'image résultante
    save_image(argv[2], &img);

    // Affichage du temps d'exécution
    printf("Temps d'exécution : %.4f secondes\n", total_time);

    free(img.image); // Libère la mémoire de l'image
    return 0;
}
