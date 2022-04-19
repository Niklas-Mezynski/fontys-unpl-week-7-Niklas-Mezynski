#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include <time.h>

#define ARR_SIZE (1048576)
#define EXPECTED_RESULT 960766820994252800
// #define EXPECTED_RESULT 895204346883

void initArrays(float *arr1, float *arr2);
float dotP_cpu_float_precision(float *arr1, float *arr2);
double dotP_cpu_double_precision(float *arr1, float *arr2);
float dotP_avx_float_precision(float *arr1, float *arr2);
double dotP_avx_double_precision(float *arr1, float *arr2);
double calcRelativeErrorDouble(double result);
float calcRelativeErrorFloat(float result);

int main()
{
    float *arr1 = (float *)malloc(ARR_SIZE * sizeof(float));
    float *arr2 = (float *)malloc(ARR_SIZE * sizeof(float));
    initArrays(arr1, arr2);

    // Calc the dot product on the CPU
    clock_t beginFloatCpu = clock();
    float floatResultCpu = dotP_cpu_float_precision(arr1, arr2);
    clock_t endFloatCpu = clock();
    clock_t time_spent_float_cpu = (double)endFloatCpu - beginFloatCpu / (double) CLOCKS_PER_SEC;

    clock_t beginDoubleCpu = clock();
    double doubleResultCpu = dotP_cpu_double_precision(arr1, arr2);
    clock_t endDoubleCpu = clock();
    double time_spent_Double_cpu = (double)(endDoubleCpu - beginDoubleCpu) / (double) CLOCKS_PER_SEC;

    clock_t beginFloatAVX = clock();
    float floatResultAVX = dotP_avx_float_precision(arr1, arr2);
    clock_t endFloatAVX = clock();
    double time_spent_float_AVX = (double)(endFloatAVX - beginFloatAVX) / (double) CLOCKS_PER_SEC;

    clock_t beginDoubleAVX = clock();
    double doubleResultAVX = dotP_avx_double_precision(arr1, arr2) / (double) CLOCKS_PER_SEC;
    clock_t endDoubleAVX = clock();
    double time_spent_Double_AVX = (double)(endDoubleAVX - beginDoubleAVX) / (double) CLOCKS_PER_SEC;

    printf("Relative error CPU Calculation with float precission [Time: %ld | Result: %f]: %f\n", time_spent_float_cpu, floatResultCpu, calcRelativeErrorFloat(floatResultCpu));
    printf("Relative error CPU Calculation with double precission [Time: %lf | Result: %lf]: %lf\n", time_spent_Double_cpu, doubleResultCpu, calcRelativeErrorDouble(doubleResultCpu));

    printf("Relative error AVX Calculation with float precission [Time: %lf | Result: %f]: %f\n", time_spent_float_AVX, floatResultAVX, calcRelativeErrorFloat(floatResultAVX));
    printf("Relative error AVX Calculation with double precission [Time: %lf | Result: %lf]: %lf\n", time_spent_Double_AVX, doubleResultAVX, calcRelativeErrorDouble(doubleResultAVX));

    return 0;
}

void initArrays(float *arr1, float *arr2)
{
    for (int i = 0; i < ARR_SIZE; i++)
    {
        arr1[i] = (float)i;
        arr2[i] = (float)(i + ARR_SIZE);
    }
}

float dotP_cpu_float_precision(float *arr1, float *arr2)
{
    float dotP = 0.0f;
    for (int i = 0; i < ARR_SIZE; i++)
    {
        dotP += arr1[i] * arr2[i];
    }
    return dotP;
}

double dotP_cpu_double_precision(float *arr1, float *arr2)
{
    double dotP = 0.0;
    for (int i = 0; i < ARR_SIZE; i++)
    {
        dotP += ((double)arr1[i]) * ((double)arr2[i]);
    }
    return dotP;
}

double calcRelativeErrorDouble(double result)
{
    return fabs(1 - (result / EXPECTED_RESULT));
}

float calcRelativeErrorFloat(float result)
{
    return fabsf(1 - (result / EXPECTED_RESULT ));
}

float dotP_avx_float_precision(float *arr1, float *arr2)
{
    float result = 0.0f;
    for (int i = 0; i < ARR_SIZE; i += 8)
    {
        __m256 avx1 = _mm256_set_ps(arr1[i], arr1[i + 1], arr1[i + 2], arr1[i + 3], arr1[i + 4], arr1[i + 5], arr1[i + 6], arr1[i + 7]);
        __m256 avx2 = _mm256_set_ps(arr2[i], arr2[i + 1], arr2[i + 2], arr2[i + 3], arr2[i + 4], arr2[i + 5], arr2[i + 6], arr2[i + 7]);

        __m256 r = _mm256_mul_ps(avx1, avx2);
        float *results = (float *)&r;
        for (int j = 0; j < 8; j++)
        {
            result += results[j];
        }
    }
    return result;
}

double dotP_avx_double_precision(float *arr1, float *arr2)
{
    double result = 0.0;
    for (int i = 0; i < ARR_SIZE; i += 4)
    {
        __m256d avx1 = _mm256_set_pd(arr1[i], arr1[i + 1], arr1[i + 2], arr1[i + 3]);
        __m256d avx2 = _mm256_set_pd(arr2[i], arr2[i + 1], arr2[i + 2], arr2[i + 3]);

        __m256d r = _mm256_mul_pd(avx1, avx2);
        double *results = (double *)&r;
        for (int j = 0; j < 4; j++)
        {
            result += results[j];
        }
    }
    return result;
}