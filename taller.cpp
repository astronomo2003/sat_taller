#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

int main(){

    cv::Mat img = cv::imread("imagen_sat.jpeg", cv::IMREAD_GRAYSCALE);

    if(img.empty()){
        std::cout << "Error cargando imagen" << std::endl;
        return -1;
    }

    cv::imshow("Original", img);

    int filas = img.rows;
    int columnas = img.cols;

    std::cout << "Filas: " << filas << std::endl;
    std::cout << "Columnas: " << columnas << std::endl;
    std::cout << "Canales: " << img.channels() << std::endl;
    std::cout << "Profundidad: " << img.depth() << std::endl;

    // =========================
    // MEDIA
    // =========================

    double suma = 0.0;

    for(int i = 0; i < filas; i++){
        for(int j = 0; j < columnas; j++){
            suma += img.at<uchar>(i,j);
        }
    }

    double media = suma / (filas * columnas);

    // =========================
    // DESVIACION
    // =========================

    double suma_varianza = 0.0;

    for(int i = 0; i < filas; i++){
        for(int j = 0; j < columnas; j++){
            double diferencia = img.at<uchar>(i,j) - media;
            suma_varianza += diferencia * diferencia;
        }
    }

    double desviacion = std::sqrt(suma_varianza / (filas * columnas));

    std::cout << "Media: " << media << std::endl;
    std::cout << "Desviacion: " << desviacion << std::endl;

    // =========================
    // SEGMENTACION 2 SIGMA
    // =========================

    cv::Mat mascara = cv::Mat::zeros(img.size(), CV_8U);

    for(int i = 0; i < filas; i++){
        for(int j = 0; j < columnas; j++){
            uchar pixel = img.at<uchar>(i,j);

            if(pixel > media - 2*desviacion &&
               pixel < media + 2*desviacion){
                mascara.at<uchar>(i,j) = 255;
            }
        }
    }

    cv::imshow("Segmentacion 2 Sigma", mascara);

    // =========================
    // FILTRO GAUSSIANO 3x3
    // =========================

    cv::Mat suavizada = cv::Mat::zeros(img.size(), CV_8U);

    int kernel_gauss[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    for(int i = 1; i < filas-1; i++){
        for(int j = 1; j < columnas-1; j++){

            int suma_kernel = 0;

            for(int ki = -1; ki <= 1; ki++){
                for(int kj = -1; kj <= 1; kj++){
                    suma_kernel += img.at<uchar>(i+ki, j+kj) *
                                   kernel_gauss[ki+1][kj+1];
                }
            }

            suavizada.at<uchar>(i,j) = suma_kernel / 16;
        }
    }

    cv::imshow("Suavizada", suavizada);

    // =========================
    // LAPLACIANO 3x3
    // =========================

    cv::Mat laplaciano = cv::Mat::zeros(img.size(), CV_8U);

    int kernel_lap[3][3] = {
        {0, -1, 0},
        {-1, 4, -1},
        {0, -1, 0}
    };

    for(int i = 1; i < filas-1; i++){
        for(int j = 1; j < columnas-1; j++){

            int suma_kernel = 0;

            for(int ki = -1; ki <= 1; ki++){
                for(int kj = -1; kj <= 1; kj++){
                    suma_kernel += img.at<uchar>(i+ki, j+kj) *
                                   kernel_lap[ki+1][kj+1];
                }
            }

            if(suma_kernel < 0) suma_kernel = 0;
            if(suma_kernel > 255) suma_kernel = 255;

            laplaciano.at<uchar>(i,j) = suma_kernel;
        }
    }

    cv::imshow("Laplaciano", laplaciano);

    // =========================
    // AFILADO MANUAL
    // =========================

    cv::Mat afilada = cv::Mat::zeros(img.size(), CV_8U);

    for(int i = 0; i < filas; i++){
        for(int j = 0; j < columnas; j++){

            int valor = img.at<uchar>(i,j) +
                        laplaciano.at<uchar>(i,j);

            if(valor > 255) valor = 255;

            afilada.at<uchar>(i,j) = valor;
        }
    }

    cv::imshow("Afilada", afilada);

    cv::waitKey(0);
    return 0;
}
