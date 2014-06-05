#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>


const char* source[] = { 
 "__kernel void saxpy_opencl(int n, float a, __global float* \
x, __global float* y)", 
 "{", 
 " int i = get_global_id(0);", 
 " if( i < n ){", 
 " y[i] = a * x[i] + y[i];", 
 " }", 
 "}" 
};

int main(int argc, char* argv[]) { 
 int n = 10240; float a = 2.0; 
 float* h_x, *h_y; // Pointer to CPU memory 
 h_x = (float*) malloc(n * sizeof(float)); 
 h_y = (float*) malloc(n * sizeof(float)); 
 // Initialize h_x and h_y 
 for(int i=0; i<n; ++i){ 
 h_x[i]=i; h_y[i]=5.0*i-1.0; 
 } 
 // Get an OpenCL platform 
 cl_platform_id platform; 
 clGetPlatformIDs(1,&platform, NULL); 
 // Create context 
 cl_device_id device; 
 clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 
NULL); 
 cl_context context = clCreateContext(0, 1, &device, NULL, 
NULL, NULL); 
 // Create a command-queue on the GPU device 
 cl_command_queue queue = clCreateCommandQueue(context, 
device, 0, NULL); 
 
 
 // Create OpenCL program with source code 
 cl_program program = clCreateProgramWithSource(context, 7, source, 
NULL, NULL); 
 // Build the program 
 clBuildProgram(program, 0, NULL, NULL, NULL, NULL); 
 // Allocate memory on device on initialize with host data 
 cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY | 
CL_MEM_COPY_HOST_PTR, n*sizeof(float), h_x, NULL); 
 cl_mem d_y = clCreateBuffer(context, CL_MEM_READ_WRITE | 
CL_MEM_COPY_HOST_PTR, n*sizeof(float), h_y, NULL); 
 // Create kernel: handle to the compiled OpenCL function 
 cl_kernel saxpy_kernel = clCreateKernel(program, "saxpy_opencl", 
NULL); 
 // Set kernel arguments 
 clSetKernelArg(saxpy_kernel, 0, sizeof(int), &n); 
 clSetKernelArg(saxpy_kernel, 1, sizeof(float), &a); 
 clSetKernelArg(saxpy_kernel, 2, sizeof(cl_mem), &d_x); 
 clSetKernelArg(saxpy_kernel, 3, sizeof(cl_mem), &d_y); 
 // Enqueue kernel execution 
 size_t threadsPerWG[] = {128}; 
 size_t threadsTotal[] = {n}; 
 clEnqueueNDRangeKernel(queue, saxpy_kernel, 1, 0, threadsTotal, 
threadsPerWG, 0,0,0); 
 // Copy results from device to host 
 clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, n*sizeof(float), h_y, 
0, NULL, NULL); 
 // Cleanup 

 clFinish(queue);

 // display results to the screen
 int i;
 for(i=0;i<n; i++)
 {
	printf("%f \n",h_y[i]);
 }


 clReleaseKernel(saxpy_kernel); 
 clReleaseProgram(program); 
 clReleaseCommandQueue(queue); 
 clReleaseContext(context); 
 clReleaseMemObject(d_x); clReleaseMemObject(d_y); 
 free(h_x); free(h_y); return 0; 
}