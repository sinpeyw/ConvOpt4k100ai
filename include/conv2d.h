#ifndef __CONV2D_FP16_FWD_HEADER__
#define __CONV2D_FP16_FWD_HEADER__

#define __in__
#define __out__
#define __in_out__

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_K 16
#define BLOCK_SIZE_N 32
#define A_BLOCK_SIZE (BLOCK_SIZE_M*BLOCK_SIZE_K)
#define B_BLOCK_SIZE (BLOCK_SIZE_N*BLOCK_SIZE_K)

#define THREAD_SIZE_X 4
#define THREAD_SIZE_Y 4
#define THREAD_X_PER_BLOCK (BLOCK_SIZE_N / THREAD_SIZE_X)
#define THREAD_Y_PER_BLOCK (BLOCK_SIZE_M /THREAD_SIZE_Y) 
#define THREAD_NUM_PER_BLOCK (THREAD_X_PER_BLOCK*THREAD_X_PER_BLOCK)

typedef struct
{
    _Float16*   in;                             //输入数据地址
    _Float16*   weight;                         //权值数据地址
    _Float16*   out;                            //输出数据地址
    unsigned int      n;                              //batch szie              default value 1
    unsigned int      c;                              //channel number          default value 32
    unsigned int      h;                              //数据高                  default value 32
    unsigned int      w;                              //数据宽                  default value 32
    unsigned int      k;                              //卷积核数量              default value 32
    unsigned int      r;                              //卷积核高                default value 1
    unsigned int      s;                              //卷积核宽                default value 1
    unsigned int      u;                              //卷积在高方向上的步长     default value 1
    unsigned int      v;                              //卷积在宽方向上的步长     default value 1
    unsigned int      p;                              //卷积在高方向上的补边     default value 0
    unsigned int      q;                              //卷积在宽方向上的补边     default value 0
}problem_t;

typedef struct
{
    unsigned int         blockx;                    //blockx  number
    unsigned int         blocky;                    //blocky  number
    unsigned int         blockz;                    //blockz  number
    unsigned int         threadx;                   //threadx number per block
    unsigned int         thready;                   //thready number per block
    unsigned int         threadz;                   //threadz number per block
    unsigned int         dynmicLdsSize;             //动态分配的lds大小，如果不使用动态分配的lds，则该值为0；
    void*       kernelPtr;                 //kernel ptr
}kernelInfo_t;


int getParamsize(__in__ problem_t* problem, __out__ int* paramSize);
int getkernelInfo(__in__ problem_t* problem, __out__  kernelInfo_t* kernelInfo, __in_out__ void* param);

#endif