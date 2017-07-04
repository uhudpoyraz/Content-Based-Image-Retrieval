#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include <time.h>

using namespace cv;
using namespace std;

typedef struct kMeanWithHue {
	int cluster;
	int h;
} Matrix;

typedef struct Train {
	char name[60];
	int segment;
	float histogram[360];
} Train;

// this function set a pixel to a cluster of kmean
int setCluster(int centerCount, int* centerMatrix, int H) {
	int minIndex = 0, i = 0;
	int tempValue;
	tempValue = abs(H - centerMatrix[0]);
	for (i = 0; i < centerCount; i++) {
		if (abs(H - centerMatrix[i]) < tempValue) {
			minIndex = i;
			tempValue = abs(H - centerMatrix[i]);
		}
	}
	return minIndex;
}

// calculate histogram of image and write to file 
void traning(Mat hsvImage,
	Matrix** kMeanMatrix,
	vector<Mat> hsvChannels,
	int centerCount,
	char* filename,
	double kMeanClusterSizeThreshold) {
	int totalPixel = hsvImage.rows * hsvImage.cols;
	int threshholdSize = int(totalPixel * kMeanClusterSizeThreshold);
	int counting = 0;
	int histogramSize = 360;
	int* histogram = (int*)calloc(histogramSize, sizeof(int));

	printf("Toplam Pixel %d\n", totalPixel);
	printf("threshold %d\n", threshholdSize);

	for (int k = 0; k < centerCount; k++) {
		for (int i = 0; i < hsvImage.rows; ++i) {
			for (int j = 0; j < hsvImage.cols; ++j) {
				if (k == kMeanMatrix[i][j].cluster) {
					counting++;
					histogram[hsvChannels[0].at<uchar>(i, j)]++;
				}
			}
		}
		if (counting > threshholdSize) {
			FILE* fp = fopen("training.txt", "a");
			fprintf(fp, "%s %d\n", filename, k);
			printf("%d. Segment: %d\n", k, counting);
			for (int i = 0; i < histogramSize; i++) {
				fprintf(fp, "%f ", (float)histogram[i] / (float)counting);
				histogram[i] = 0;
			}
			fprintf(fp, "\n");
			fclose(fp);
		}
		counting = 0;
	}
}


//test image prepare for comparing trained images
int writeToMatrix(Mat hsvImage,
	Matrix** kMeanMatrix,
	vector<Mat> hsvChannels,
	int centerCount,
	Train* testImage,
	double kMeanClusterSizeThreshold) {
	int totalPixel = hsvImage.rows * hsvImage.cols;
	int threshholdSize = int(totalPixel * 0.08);
	int counting = 0;
	int countSegmentToMatrix = 0;
	int histogramSize = 360;
	int* histogram = (int*)calloc(histogramSize, sizeof(int));

	printf("Total Pixel %d\n", totalPixel);
	printf("threshold %d\n", threshholdSize);

	for (int k = 0; k < centerCount; k++) {
		for (int i = 0; i < hsvImage.rows; ++i) {
			for (int j = 0; j < hsvImage.cols; ++j) {
				if (k == kMeanMatrix[i][j].cluster) {
					counting++;
					histogram[hsvChannels[0].at<uchar>(i, j)]++;
				}
			}
		}

		if (counting > threshholdSize) {
			printf("%d. Segment: %d\n", k, counting);
			for (int i = 0; i < histogramSize; i++) {
				testImage[countSegmentToMatrix].histogram[i] =
					(float)histogram[i] / (float)counting;
				histogram[i] = 0;
			}
			countSegmentToMatrix++;
		}
		counting = 0;
	}

	return countSegmentToMatrix;
}

// training image calculated histogram and write to file
int readFromFile(Train* trainList) {
	int segment, i;

	char ad[60];
	int histogramSize = 360;
	FILE* fp;
	int index = 0;

	double* imageHistogram = (double*)malloc(sizeof(double)* 360);

	if ((fp = fopen("training.txt", "r")) == NULL)
		printf("Dosya açýlamadý\n");
	else {
		fscanf(fp, "%s%d", trainList[index].name, &trainList[index].segment);
		i = 0;
		while (i < 360) {
			fscanf(fp, "%f", &trainList[index].histogram[i]);
			i++;
		}
		index++;
		while (!feof(fp)) {
			fscanf(fp, "%s%d", trainList[index].name, &trainList[index].segment);
			i = 0;
			while (i < 360) {
				fscanf(fp, "%f", &trainList[index].histogram[i]);

				i++;
			}
			index++;
		}
		fclose(fp);
	}

	return index;
}

// creating kmeans
void createKmeans(Mat hsvImage,
	Matrix** kMeanMatrix,
	vector<Mat> hsvChannels,
	int* centerMatrix,
	int* oldcenterMatrix,
	int centerCount) {
	int isChanged, j, i;
	for (int i = 0; i < centerCount; i++) {
		centerMatrix[i] = hsvChannels[0].at<uchar>(rand() % hsvImage.rows,
			rand() % hsvImage.cols);
		j = 0;
		while (j < i) {
			if (centerMatrix[i] == centerMatrix[j]) {
				centerMatrix[i] = hsvChannels[0].at<uchar>(rand() % hsvImage.rows,
					rand() % hsvImage.cols);

				j = 0;
			}
			j++;
		}
	}

	isChanged = centerCount;
	while (isChanged > 0) {
		int sum = 0;
		int sumCount = 0;
		for (int i = 0; i < hsvImage.rows; i++) {
			for (int j = 0; j < hsvImage.cols; j++) {
				kMeanMatrix[i][j].h = int(hsvChannels[0].at<uchar>(i, j));
				kMeanMatrix[i][j].cluster =
					setCluster(centerCount, centerMatrix, kMeanMatrix[i][j].h);
			}
		}

		for (int i = 0; i < centerCount; i++) {
			oldcenterMatrix[i] = centerMatrix[i];
		}

		for (int k = 0; k < centerCount; k++) {
			for (int i = 0; i < hsvImage.rows; i++) {
				for (int j = 0; j < hsvImage.cols; j++) {
					if (kMeanMatrix[i][j].cluster == k) {
						sum = sum + int(hsvChannels[0].at<uchar>(i, j));
						sumCount++;
					}
				}
			}
			if (sumCount != 0)
				centerMatrix[k] = int(sum / (sumCount));
			sum = 0;
			sumCount = 0;
		}
		isChanged = 0;
		i = 0;
		while (centerMatrix[i] != oldcenterMatrix[i]) {
			isChanged++;
			i++;
		}
	}
}
int main(int argc, char** argv) {
	srand((unsigned int)time(NULL));
	char filename[100];
	int centerCount;
	int cond, isChanged;
	int testImageRowCount = 0;
	int traingImagesRowCount = 0;
	double distanceThreshold, kMeanClusterSizeThreshold;

	int menuId;
	bool menuLoop = true;
	int *centerMatrix, *oldcenterMatrix;
	vector<Mat> hsvChannels;
	Mat rgbImage, hsvImage, H1, H2;
	Train *testImage, *trainList;
	Matrix** kMeanMatrix;
	while (menuLoop) {
		cout << " 1) Train set\n";
		cout << " 2) Test image\n";
		cin >> menuId;

		cout << " Enter the file name:\n";
		cin >> filename;

		cout << "Please enter K value: \n";
		cin >> centerCount;
		cout << "Please enter the distance threshold between 0-1 \n";
		cin >> distanceThreshold;

		cout << "Please enter the cluster pixel size threshold between 0-1 \n";
		cin >> kMeanClusterSizeThreshold;

		centerMatrix = (int*)malloc(sizeof(int)* centerCount);
		oldcenterMatrix = (int*)malloc(sizeof(int)* centerCount);

		rgbImage = imread(filename, 1);
		cvtColor(rgbImage, hsvImage, CV_RGB2HSV);
		if (rgbImage.empty()) {
			printf("Empty");
			system("pause");
		}
		imshow("input", rgbImage);
		split(hsvImage, hsvChannels);

		testImage = (Train*)malloc(sizeof(Train)* centerCount);
		trainList = (Train*)malloc(sizeof(Train)* 500);
		kMeanMatrix = (kMeanWithHue**)malloc(sizeof(kMeanWithHue*)* hsvImage.rows);
		for (int i = 0; i < hsvImage.rows; i++) {
			kMeanMatrix[i] =
				(kMeanWithHue*)malloc(sizeof(kMeanWithHue)* hsvImage.cols);
		}
		createKmeans(hsvImage, kMeanMatrix, hsvChannels, centerMatrix,
			oldcenterMatrix, centerCount);

		if (menuId == 1) {
			traning(hsvImage, kMeanMatrix, hsvChannels, centerCount, filename,
				kMeanClusterSizeThreshold);
		}
		else if (menuId == 2) {
			traingImagesRowCount = readFromFile(trainList);
			testImageRowCount =
				writeToMatrix(hsvImage, kMeanMatrix, hsvChannels, centerCount,
				testImage, kMeanClusterSizeThreshold);
			printf("best match image closest to zero\n");
			for (int i = 0; i < testImageRowCount; i++) {
				for (int j = 0; j < traingImagesRowCount; j++) {
					H1 = Mat(1, 360, CV_32FC1, testImage[i].histogram);
					H2 = Mat(1, 360, CV_32FC1, trainList[j].histogram);
					float distance = compareHist(H2, H1, CV_COMP_BHATTACHARYYA);
					if (distanceThreshold > distance) {
						printf("%s %d ", trainList[j].name, trainList[j].segment);
						printf("result: %f \n", distance);
					}
				}
			}

			free(kMeanMatrix);
			free(testImage);
			free(trainList);
			free(centerMatrix);
			free(oldcenterMatrix);
		}
		else {
			menuLoop = false;
		}

		waitKey();
		system("cls");
	}

	return 0;
}
