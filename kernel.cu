
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

__global__ void bgr_to_gray_kernel(unsigned char* input,
	unsigned char* output,
	int szer,
	int wys,
	int colorWidthStep,
	int grayWidthStep);

__global__ void kontury_kernel(unsigned char* input,
	unsigned char* output,
	int szer,
	int wys,
	int colorWidthStep);

__global__ void konwolucja_kernel(unsigned char* input,
	unsigned char* output,
	int szer,
	int wys,
	int colorWidthStep, float* macierz);

__global__ void negatyw_kernel(unsigned char* input,
	unsigned char* output,
	int szer,
	int wys,
	int colorWidthStep);

void konwolucja(const Mat& input, Mat& output, float* macierz);
void kontury(const Mat& input, Mat& output);
void negatyw(const Mat& input, Mat& output);


void convert_to_gray(const Mat& input, Mat& output);
int main(int argc, char** argv)
{

	Mat image = imread("F:/test1.jpg");

	if (image.empty())
	{
		cout << "Nie znaleziono zdjecia" << endl;
		system("pause");
		return -1;
	}
	cv::resize(image, image, cv::Size(), 0.5, 0.5);

	cv::Mat temp = image.clone();



	float GaussianBlur[9] = { 1, 2, 1,2,4,2,1,2,1 };

	for (int i = 0; i < 9; i++)
	{
		float x = GaussianBlur[i];
		GaussianBlur[i] = float((x / 16.0));
	}


	float BoxBlur[9] = { 1,1,1,1,1,1,1,1,1 };
	for (int i = 0; i < 9; i++)
	{
		float x = BoxBlur[i];
		BoxBlur[i] = float((x / 9));
	}
	float  Sharpen[9] = { 0, -1, 0,-1,5,-1,0,-1,0 };
	float EdgeDetection[9] = { -1, -1, -1,-1,8,-1, -1, -1,-1 };



	while (true) {
		cv::Mat temp = image.clone();
		cv::Mat output_mono(image.rows, image.cols, CV_8UC1);
		cv::Mat output_negative(image.rows, image.cols, CV_8UC3);
		//cv::Mat output(temp);
		int decyzja;
		cout << "Wybierz rodzaj filtru jaki chcesz zastosowaæ: " << endl << "1. Wykrywanie krawedzi filtrem Sobela" << endl <<
			"2. Filtr monochromatyczny" << endl << "3. Wykrywanie krawedzi filtrem konwolucyjnym" << endl <<
			"4. Filtr GaussianBlur" << endl << "5. Filtr BoxBlur" << endl << "6. Wyostrzanie obrazu" << endl << "7. Negatyw" << endl;
		cin >> decyzja;

		switch (decyzja)
		{
		case 1:
			kontury(temp, temp);
		case 2:

			convert_to_gray(temp, output_mono);
		case 3:
			konwolucja(temp, temp, EdgeDetection);
		case 4:
			konwolucja(temp, temp, GaussianBlur);
			konwolucja(temp, temp, GaussianBlur);
			konwolucja(temp, temp, GaussianBlur);
		case 5:
			konwolucja(temp, temp, BoxBlur);
			konwolucja(temp, temp, BoxBlur);
			konwolucja(temp, temp, BoxBlur);
		case 6:
			konwolucja(temp, temp, Sharpen);
		case 7:
			negatyw(temp, output_negative);
		default:
			cout << "Bledny numer!" << endl;
			break;

		}

		String windowName = "Output";
		namedWindow(windowName);
		if (decyzja == 2) {
			imshow(windowName, output_mono);
		}
		else if (decyzja == 7)
		{
			imshow(windowName, output_negative);
		}
		else imshow(windowName, temp);
		waitKey(0);
		destroyWindow(windowName);
	}



	return 0;
}
void convert_to_gray(const cv::Mat& input, cv::Mat& output)
{
	//Obliczanie rozmiaru, step zawiera liczbe bajtow dana na 1 rzad
	const int SizeKol = input.step * input.rows;
	const int SizeSzary = output.step * output.rows;

	unsigned char* d_input, * d_output;

	//Alokacja
	cudaMalloc<unsigned char>(&d_input, SizeKol);
	cudaMalloc<unsigned char>(&d_output, SizeSzary);



	cudaMemcpy(d_input, input.ptr(), SizeKol, cudaMemcpyHostToDevice);

	const dim3 block(16, 16);

	//Liczymy wymagany rozmiar gridu
	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);


	bgr_to_gray_kernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, input.step, output.step);

	cudaMemcpy(output.ptr(), d_output, SizeSzary, cudaMemcpyDeviceToHost);


	//Free the device memory
	cudaFree(d_input);
	cudaFree(d_output);
}

void kontury(const Mat& input, Mat& output)
{
	// Implementacja filtru Sobela


	//liczymy rozmiary obrazow
	const int sizeWejscia = input.step * input.rows;
	const int sizeWyjscia = output.step * output.rows;

	//alokacja
	unsigned char* d_input, * d_output;
	cudaMalloc<unsigned char>(&d_input, sizeWejscia);
	cudaMalloc<unsigned char>(&d_output, sizeWyjscia);

	cudaMemcpy(d_input, input.ptr(), sizeWejscia, cudaMemcpyHostToDevice);
	const dim3 block(16, 16);

	//Liczymy grid który pomiesci obraz
	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	kontury_kernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, input.step);

	cudaMemcpy(output.ptr(), d_output, sizeWyjscia, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

}
void negatyw(const Mat& input, Mat& output) {
	const int sizeWejscia = input.step * input.rows;
	const int sizeWyjscia = output.step * output.rows;

	unsigned char* d_input, * d_output;
	cudaMalloc<unsigned char>(&d_input, sizeWejscia);
	cudaMalloc<unsigned char>(&d_output, sizeWyjscia);

	cudaMemcpy(d_input, input.ptr(), sizeWejscia, cudaMemcpyHostToDevice);
	const dim3 block(16, 16);

	//Liczymy grid który pomiesci obraz
	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	negatyw_kernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, input.step);

	cudaMemcpy(output.ptr(), d_output, sizeWyjscia, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

}
void konwolucja(const Mat& input, Mat& output, float* macierz)
{
	//liczymy rozmiary obrazow
	const int sizeWejscia = input.step * input.rows;
	const int sizeWyjscia = output.step * output.rows;

	//alokacja
	unsigned char* d_input, * d_output;
	float* d_macierz;
	cudaMalloc<unsigned char>(&d_input, sizeWejscia);
	cudaMalloc<unsigned char>(&d_output, sizeWyjscia);

	int rozmiar_macierzy = 9 * sizeof(float);
	cudaMalloc((void**)&d_macierz, rozmiar_macierzy);

	//kopiowanie

	cudaMemcpy(d_input, input.ptr(), sizeWejscia, cudaMemcpyHostToDevice);
	cudaMemcpy(d_macierz, macierz, rozmiar_macierzy, cudaMemcpyHostToDevice);

	const dim3 block(16, 16);

	//Liczymy grid który pomiesci obraz
	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	konwolucja_kernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, input.step, d_macierz);

	cudaMemcpy(output.ptr(), d_output, sizeWyjscia, cudaMemcpyDeviceToHost);
	cudaFree(d_input); cudaFree(d_output); cudaFree(d_macierz);
}

__global__ void bgr_to_gray_kernel(unsigned char* input,
	unsigned char* output,
	int szer,
	int wys,
	int colorWidthStep,
	int grayWidthStep)
{

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;


	if ((xIndex < szer) && (yIndex < wys))
	{

		const int piksel_id = yIndex * colorWidthStep + (3 * xIndex);


		const int gray_tid = yIndex * grayWidthStep + xIndex;

		const unsigned char blue = input[piksel_id];
		const unsigned char green = input[piksel_id + 1];
		const unsigned char red = input[piksel_id + 2];

		const float gray = (red * 0.3f + green * 0.59f + blue * 0.11f);


		output[gray_tid] = static_cast<unsigned char>(gray);

	}
}


__global__ void kontury_kernel(unsigned char* input,
	unsigned char* output,
	int szer,
	int wys,
	int colorWidthStep)
{
	int Gx[3][3] = { {-1,0,1 },{-2,0,2},{-1,0,1} };
	int Gy[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1} };

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if ((xIndex < szer - 1 && (xIndex > 0)) && (yIndex < wys - 1 && (yIndex > 0))) {

		const int piksel_id = yIndex * colorWidthStep + (3 * xIndex); //indeks jednego konkretnego piksela obrazu, kolejne 3 to kolory


		float pix_x = (Gx[0][0] * input[(yIndex - 1) * colorWidthStep + (3 * (xIndex - 1))]) +
			(Gx[0][1] * input[(yIndex)*colorWidthStep + (3 * (xIndex - 1))]) +
			(Gx[0][2] * input[(yIndex + 1) * colorWidthStep + (3 * (xIndex - 1))]) +
			(Gx[1][0] * input[(yIndex - 1) * colorWidthStep + (3 * (xIndex))]) +
			(Gx[1][1] * input[(yIndex)*colorWidthStep + (3 * (xIndex))]) +
			(Gx[1][2] * input[(yIndex + 1) * colorWidthStep + (3 * (xIndex))]) +
			(Gx[2][0] * input[(yIndex - 1) * colorWidthStep + (3 * (xIndex + 1))]) +
			(Gx[2][1] * input[(yIndex)*colorWidthStep + (3 * (xIndex + 1))]) +
			(Gx[2][2] * input[(yIndex + 1) * colorWidthStep + (3 * (xIndex + 1))]);

		float pix_y = (Gy[0][0] * input[(yIndex - 1) * colorWidthStep + (3 * (xIndex - 1))]) +
			(Gy[0][1] * input[(yIndex)*colorWidthStep + (3 * (xIndex - 1))]) +
			(Gy[0][2] * input[(yIndex + 1) * colorWidthStep + (3 * (xIndex - 1))]) +
			(Gy[1][0] * input[(yIndex - 1) * colorWidthStep + (3 * (xIndex))]) +
			(Gy[1][1] * input[(yIndex)*colorWidthStep + (3 * (xIndex))]) +
			(Gy[1][2] * input[(yIndex + 1) * colorWidthStep + (3 * (xIndex))]) +
			(Gy[2][0] * input[(yIndex - 1) * colorWidthStep + (3 * (xIndex + 1))]) +
			(Gy[2][1] * input[(yIndex)*colorWidthStep + (3 * (xIndex + 1))]) +
			(Gy[2][2] * input[(yIndex + 1) * colorWidthStep + (3 * (xIndex + 1))]);
		float val1 = sqrtf((pix_x * pix_x) + (pix_y * pix_y));


		unsigned char val = static_cast<unsigned char>(val1);

		output[piksel_id] = val;
		output[piksel_id + 1] = val;
		output[piksel_id + 2] = val;
	}


}

__global__ void konwolucja_kernel(unsigned char* input,
	unsigned char* output,
	int szer,
	int wys,
	int colorWidthStep, float* macierz)
{

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	__syncthreads();

	if ((xIndex < szer - 3 && (xIndex > 2)) && (yIndex < wys - 3 && (yIndex > 2))) {


		const int piksel_id = yIndex * colorWidthStep + (3 * xIndex); //indeks jednego konkretnego piksela obrazu, kolejne 3 to kolory
		float val_r = 0;
		float val_g = 0;
		float val_b = 0;
		int r = 0;
		int c = 0;
		for (int i = 0; i < 3; i++)
		{
			r = -1 + i;
			for (int j = 0; j < 3; j++)
			{
				c = -1 + j;
				val_r += macierz[3 * i + j] * input[(yIndex + c) * colorWidthStep + (3 * (xIndex + r))];
				val_g += macierz[3 * i + j] * input[(yIndex + c) * colorWidthStep + (3 * (xIndex + r)) + 1];
				val_b += macierz[3 * i + j] * input[(yIndex + c) * colorWidthStep + (3 * (xIndex + r)) + 2];
			}
		}
		unsigned char val_rc = static_cast<unsigned char>(val_r);
		unsigned char val_gc = static_cast<unsigned char>(val_g);
		unsigned char val_bc = static_cast<unsigned char>(val_b);

		output[piksel_id] = val_rc;
		output[piksel_id + 1] = val_gc;
		output[piksel_id + 2] = val_bc;
	}

}

__global__ void negatyw_kernel(unsigned char* input,
	unsigned char* output,
	int szer,
	int wys,
	int colorWidthStep)
{
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	__syncthreads();

	if ((xIndex < szer && (xIndex > 0)) && (yIndex < wys && (yIndex > 0))) {

		const int piksel_id = yIndex * colorWidthStep + (3 * xIndex); //indeks jednego konkretnego piksela obrazu, kolejne 3 to kolory
		float val_r = 0;
		float val_g = 0;
		float val_b = 0;

		val_r = 255 - input[piksel_id];
		val_g = 255 - input[piksel_id + 1];
		val_b = 255 - input[piksel_id + 2];

		unsigned char val_rc = static_cast<unsigned char>(val_r);
		unsigned char val_gc = static_cast<unsigned char>(val_g);
		unsigned char val_bc = static_cast<unsigned char>(val_b);

		output[piksel_id] = val_rc;
		output[piksel_id + 1] = val_gc;
		output[piksel_id + 2] = val_bc;

	}
}