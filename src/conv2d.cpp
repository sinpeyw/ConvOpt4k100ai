#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "conv2d.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <numeric>

#include "hip/hip_runtime.h"
#include "hip/hip_ext.h"

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))


#define FETCH_FLOAT4(pointer) (*reinterpret_cast<float4*>(&pointer))
#define FETCH_FLOAT2(pointer) (*reinterpret_cast<float2*>(&pointer))
#define FETCH_FLOAT(pointer) (*reinterpret_cast<float*>(&pointer))


typedef float float4_ __attribute__((ext_vector_type(4)));
typedef float float2_ __attribute__((ext_vector_type(2)));

#define FETCH_FLOAT4_(pointer) (*reinterpret_cast<float4_*>(&pointer))

union RegisterUnion
{
  float4_ vector4;
  struct
  {
    float2_ vector_front;
    float2_ vector_rear;
  };
  struct{
    _Float16 a0,a1,a2,a3,a4,a5,a6,a7;
  };
};


/*自定义的kernel入参结构体*/
typedef struct mykernelParamType
{
    _Float16*   pin;                            //输入数据地址
    _Float16*   pweight;                        //权值数据地址
    _Float16*   pout;                           //输出数据地址
    unsigned int      n;                              //batch szie            
    unsigned int      c;                              //channel number        
    unsigned int      h;                              //数据高                
    unsigned int      w;                              //数据宽                
    unsigned int      k;                              //卷积核数量            
    unsigned int      r;                              //卷积核高              
    unsigned int      s;                              //卷积核宽              
    unsigned int      u;                              //卷积在高方向上的步长  
    unsigned int      v;                              //卷积在宽方向上的步长  
    unsigned int      p;                              //卷积在高方向上的补边  
    unsigned int      q;                              //卷积在宽方向上的补边  
    unsigned int      Oh;                             //卷积在高方向上输出大小    
    unsigned int      Ow;                             //卷积在宽方向上输出大小
    unsigned int      block_size_m;
    unsigned int      block_size_k;
    unsigned int      block_size_n;
    unsigned int      a_block_size;
    unsigned int      b_block_size;
    unsigned int      thread_x_per_block;
    unsigned int      thread_y_per_block;
}mykernelParamType;                          


//实现单个元素从矩阵B到输入矩阵的内存映射：返回数据版
__device__ _Float16 B2PIN(unsigned x,unsigned y, mykernelParamType param) {
    unsigned int num_slide_window_per_batch = param.Oh*param.Ow;
    unsigned int batch_i = x/num_slide_window_per_batch;         // 第几个batch
    unsigned int slide_i = x%num_slide_window_per_batch;          // batch 上的第几个滑动窗口
    unsigned int slide_y = slide_i / param.Ow;                // 滑动窗口的行号
    unsigned int slide_x = slide_i % param.Oh;                // 滑动窗口的列号
    unsigned int num_ele_per_kernel_per_channel =  param.r*param.s;
    unsigned int channel_i = y /num_ele_per_kernel_per_channel;              // 处于该滑动窗口的第几个通道
    unsigned int channel_n = y % num_ele_per_kernel_per_channel;              // 在该通道上的序号
    unsigned int channel_y = channel_n / param.s;                // 在该通道的该滑动窗口上的行号
    unsigned int channel_x = channel_n % param.s;                // 在该通道的该滑动窗口上的列号
    unsigned int c_y = slide_y*param.u - param.p + channel_y;      // 在通道上的行号
    unsigned int c_x = slide_x*param.v - param.q + channel_x;       // 在通道上的列号
    if(c_y>=0 && c_y < param.h && c_x>=0 && c_x < param.w){
        return param.pin[batch_i*param.c*param.h*param.w + channel_i*param.h*param.w + c_y*param.w + c_x];
    }
    else{
        return 0.0;
    }
}

// 实现单个元素从矩阵C到输出矩阵的内存映射
__device__ unsigned C2POUT(unsigned x,unsigned y,mykernelParamType param) {
    unsigned int size_per_outbatch = param.Oh*param.Ow*param.k;  //每个输出batch的大小
    unsigned int size_per_channel = param.Oh*param.Ow;
    unsigned int outbatch_i = x / size_per_channel ;              // 所在outbatch的序号
    unsigned int c_i = x % size_per_channel;               // 在输出channel上的序号
    return outbatch_i*size_per_outbatch + y *size_per_channel + c_i;
   
}

// 以下为不同的实现优化
// 第6组
extern "C" __global__ void t6_implicit_gemm_64x16x64_256(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{
    // shared memory
    __shared__ _Float16 lds[2*64*16 + 2*16*64];

    unsigned int K = param.r * param.s *param.c;

    const int tid = threadIdx.x;

    _Float16 ldg_a_reg[4];
    _Float16 ldg_b_reg[4];

    //transfer first tile from global mem to shared mem
    // load A from global memory to shared memory
    int row = (tid % 128) / 4; 
    int col = (tid % 4) * 4; 
    int G_x = col ;
    int G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
    if (tid < 16){
        FETCH_FLOAT2(ldg_a_reg[0]) = FETCH_FLOAT2(param.pweight[G_y*K + G_x]);
    }
    lds[(tid / 128) * 32 * 16 + col*32 + row] = ldg_a_reg[0];
    lds[(tid / 128) * 32 * 16 + (col+1)*32 + row] = ldg_a_reg[1];
    lds[(tid / 128) * 32 * 16 + (col+2)*32 + row] = ldg_a_reg[2];
    lds[(tid / 128) * 32 * 16 + (col+3)*32 + row] = ldg_a_reg[3];
    
    // load B from global memory to shared memory
    row = (tid % 128) / 8;
    col = (tid % 8) * 4;
    G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
    G_y = row;
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col] = B2PIN(G_x,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 1] = B2PIN(G_x + 1,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 2] = B2PIN(G_x + 2,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 3] = B2PIN(G_x + 3,G_y,param);
    
    __syncthreads();

    RegisterUnion fragA, fragB;
    float4_ fragC00, fragC01, fragC10, fragC11;
    fragC00 = {0, 0, 0, 0};
    fragC01 = {0, 0, 0, 0};
    fragC10 = {0, 0, 0, 0};
    fragC11 = {0, 0, 0, 0};


    int write_stage_idx = 1;
    int load_stage_idx = write_stage_idx ^ 1;
    int tile_idx = 0;

    // 大循环逻辑
    do{
        tile_idx += param.block_size_k;
        // load from share mem to register
        if(tid < 128){
            if(tid < 64){       // A0B0
                int lds_read_A0_offset = (load_stage_idx * param.block_size_k * param.block_size_m + tid * 8) * sizeof(_Float16);
                int lds_read_B0_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + tid * 8) * sizeof(_Float16);
                asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector4), "+v"(lds_read_A0_offset));
                asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B0_offset));

            }
            else if(tid >= 64 && tid < 128){        //A0B1
                int lds_read_A0_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid - 64) * 8) * sizeof(_Float16);
                int lds_read_B1_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + 16 * 32 +  (tid - 64) * 8) * sizeof(_Float16);
                asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector4), "+v"(lds_read_A0_offset));
                asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B1_offset));
            }
        }

        // load next tile from global mem to share mem:步骤1
        if(tile_idx< K){
            // load A 
            row = (tid % 128) / 4; 
            col = (tid % 4)*4; 
            G_x = col + tile_idx;
            G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
            if (tid < 16){
                 FETCH_FLOAT2(ldg_a_reg[0]) = FETCH_FLOAT2(param.pweight[G_y*K + G_x]);
            }
            // load B
            row = (tid % 128) / 8;
            col = (tid % 8) * 4;
            G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
            G_y = row + tile_idx;
            ldg_b_reg[0] = B2PIN(G_x,G_y,param);
            ldg_b_reg[1] = B2PIN(G_x + 1,G_y,param);
            ldg_b_reg[2] = B2PIN(G_x + 2,G_y,param);
            ldg_b_reg[3] = B2PIN(G_x + 3,G_y,param);
           

            if( tid < 128){
                asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
                asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
                asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));
            }
        
            // load A
            row = (tid % 128) / 4; 
            col = (tid % 4)*4; 
            G_x = col + tile_idx;
            G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
            lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + col*32 + row] = ldg_a_reg[0];
            lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+1)*32 + row] = ldg_a_reg[1];
            lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+2)*32 + row] = ldg_a_reg[2];
            lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+3)*32 + row] = ldg_a_reg[3];
            // load B
            row = (tid % 128) / 8;
            col = (tid % 8) * 4;
            G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
            G_y = row + tile_idx;
            FETCH_FLOAT2(lds[2 * param.block_size_k * param.block_size_m + write_stage_idx * param.block_size_k * param.block_size_n + (tid / 128) * 16 * 32 + row *32 + col]) = FETCH_FLOAT2(ldg_b_reg[0]);

            __syncthreads();
        }
        // switch  
        write_stage_idx ^= 1;
        load_stage_idx = write_stage_idx ^ 1;
    }while(tile_idx< K);
    
    // 完成最后一次计算
    if(tid < 128){
        asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));
    }
    __syncthreads();

    // 直接写回
    uint32_t output_row,output_col;
    if(tid < 64){
        output_row = tid % 16;
        output_col = tid / 16;
    }
    else if( tid >= 64 && tid < 128){
        output_row = (tid - 64) & 15;
        output_col = ((tid - 64) >> 4) + 32;
    }
    else if( tid >= 128 && tid < 192){
        output_row = ((tid - 128) & 15) + 32;
        output_col = ((tid - 128) >> 4);
    }
    else if( tid >= 192 && tid < 256){
        output_row = ((tid - 192) & 15) + 32;
        output_col = ((tid - 192) >> 4) + 32;
    }
    if(tid < 128){
        if( output_row < 4){
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col, param.block_size_m * blockIdx.y +  output_row,param)] =  (_Float16)fragC00.x;
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 4, param.block_size_m * blockIdx.y + output_row,param)] = (_Float16) fragC00.y;
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 8, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC00.z;
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 12, param.block_size_m * blockIdx.y + output_row,param)] = (_Float16) fragC00.w;


            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC10.x;
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 4, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC10.y;
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 8, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC10.z;
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 12, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC10.w;
        }
    }

}

// 第5组
extern "C" __global__ void t5_implicit_gemm_128x16x64_256(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{
    // shared memory
    __shared__ _Float16 lds[4*64*16 + 2*16*64];

    unsigned int K = param.r * param.s *param.c;

    const int tid = threadIdx.x;

    _Float16 ldg_a_reg[8];
    _Float16 ldg_b_reg[4];

    //transfer first tile from global mem to shared mem
    // load B from global memory to shared memory
    int row = (tid % 128) / 8;
    int col = (tid % 8) * 4;
    int G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
    int G_y = row;
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col] = B2PIN(G_x,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 1] = B2PIN(G_x + 1,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 2] = B2PIN(G_x + 2,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 3] = B2PIN(G_x + 3,G_y,param);
    
    // load A from global memory to shared memory
    //1
    row = (tid % 64) / 2; 
    col = (tid % 2) * 8; 
    G_x = col ;
    G_y = param.block_size_m * blockIdx.y + (tid / 64) * 32 + row;
    FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(param.pweight[G_y*K + G_x]);
    lds[(tid / 64) * 32 * 16 + col*32 + row] = ldg_a_reg[0];
    lds[(tid / 64) * 32 * 16 + (col+1)*32 + row] = ldg_a_reg[1];
    lds[(tid / 64) * 32 * 16 + (col+2)*32 + row] = ldg_a_reg[2];
    lds[(tid / 64) * 32 * 16 + (col+3)*32 + row] = ldg_a_reg[3];
    lds[(tid / 64) * 32 * 16 + (col+4)*32 + row] = ldg_a_reg[4];
    lds[(tid / 64) * 32 * 16 + (col+5)*32 + row] = ldg_a_reg[5];
    lds[(tid / 64) * 32 * 16 + (col+6)*32 + row] = ldg_a_reg[6];
    lds[(tid / 64) * 32 * 16 + (col+7)*32 + row] = ldg_a_reg[7];

    
    __syncthreads();

    RegisterUnion fragA[2], fragB;
    float4_ fragC00[2], fragC01[2], fragC10[2], fragC11[2];
    fragC00[0] = {0, 0, 0, 0};
    fragC01[0] = {0, 0, 0, 0};
    fragC10[0] = {0, 0, 0, 0};
    fragC11[0] = {0, 0, 0, 0};
    
    fragC00[1] = {0, 0, 0, 0};
    fragC01[1] = {0, 0, 0, 0};
    fragC10[1] = {0, 0, 0, 0};
    fragC11[1] = {0, 0, 0, 0};

    int write_stage_idx = 1;
    int load_stage_idx = write_stage_idx ^ 1;
    int tile_idx = param.block_size_k;

    // 大循环逻辑
    do{
        // load next tile from global mem to share mem:步骤1
        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        ldg_b_reg[0] = B2PIN(G_x,G_y,param);
        ldg_b_reg[1] = B2PIN(G_x + 1,G_y,param);
        ldg_b_reg[2] = B2PIN(G_x + 2,G_y,param);
        ldg_b_reg[3] = B2PIN(G_x + 3,G_y,param);
        
        // load from share mem to register
        int lds_read_A0_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        int lds_read_A1_offset = (64*16 + load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        int lds_read_B_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + ((tid % 128) / 64) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA[0].vector4), "+v"(lds_read_A0_offset));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA[1].vector4), "+v"(lds_read_A1_offset));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B_offset));
        
        
        // load A 
        row = (tid % 64) / 2; 
        col = (tid % 2)*8; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 64) * 32 + row;
        FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(param.pweight[G_y*K + G_x]);
    

        
        asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00[0]), "+v"(fragA[0].vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01[0]), "+v"(fragA[0].vector_rear), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10[0]), "+v"(fragA[0].vector_front), "+v"(fragB.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11[0]), "+v"(fragA[0].vector_rear), "+v"(fragB.vector_rear));
            
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00[1]), "+v"(fragA[1].vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01[1]), "+v"(fragA[1].vector_rear), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10[1]), "+v"(fragA[1].vector_front), "+v"(fragB.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11[1]), "+v"(fragA[1].vector_rear), "+v"(fragB.vector_rear));
    

        // load next tile from global mem to share mem:步骤2
        // load A
        row = (tid % 64) / 2; 
        col = (tid % 2)*8; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 64) * 32 + row;
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + col*32 + row] = ldg_a_reg[0];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+1)*32 + row] = ldg_a_reg[1];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+2)*32 + row] = ldg_a_reg[2];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+3)*32 + row] = ldg_a_reg[3];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+4)*32 + row] = ldg_a_reg[4];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+5)*32 + row] = ldg_a_reg[5];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+6)*32 + row] = ldg_a_reg[6];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+7)*32 + row] = ldg_a_reg[7];

        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        FETCH_FLOAT2(lds[2 * param.block_size_k * param.block_size_m + write_stage_idx * param.block_size_k * param.block_size_n + (tid / 128) * 16 * 32 + row *32 + col]) = FETCH_FLOAT2(ldg_b_reg[0]);
        

        // switch  
        write_stage_idx ^= 1;
        load_stage_idx = write_stage_idx ^ 1;
        tile_idx += param.block_size_k;
        __syncthreads();
    }while(tile_idx< K);
    
    // 完成最后一次计算
    int lds_read_A0_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
    int lds_read_A1_offset = (64*16 + load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
    int lds_read_B_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + ((tid % 128) / 64) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
    asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA[0].vector4), "+v"(lds_read_A0_offset));
    asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA[1].vector4), "+v"(lds_read_A1_offset));
    asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B_offset));

    // load from share mem to register
    asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00[0]), "+v"(fragA[0].vector_front), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01[0]), "+v"(fragA[0].vector_rear), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10[0]), "+v"(fragA[0].vector_front), "+v"(fragB.vector_rear));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11[0]), "+v"(fragA[0].vector_rear), "+v"(fragB.vector_rear));
        
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00[1]), "+v"(fragA[1].vector_front), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01[1]), "+v"(fragA[1].vector_rear), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10[1]), "+v"(fragA[1].vector_front), "+v"(fragB.vector_rear));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11[1]), "+v"(fragA[1].vector_rear), "+v"(fragB.vector_rear));   

    // store back to share mem:此处复用share mem
    uint32_t output_row = (tid / 128) * 32 + (tid % 64) %16; 
    uint32_t output_col = ((tid % 128) / 64) * 32 + ((tid % 64) / 16);
    //1
    __syncthreads();
    lds[ 64 * output_row + output_col] = fragC00[0].x;
    lds[  64 * output_row + output_col + 4] = fragC00[0].y;
    lds[  64 * output_row + output_col + 8] = fragC00[0].z;
    lds[  64 * output_row + output_col + 12] = fragC00[0].w;

    lds[  64 * (output_row + 16) + output_col] = fragC01[0].x;
    lds[  64 * (output_row + 16) + output_col + 4] = fragC01[0].y;
    lds[  64 * (output_row + 16) + output_col + 8] = fragC01[0].z;
    lds[  64 * (output_row + 16) + output_col + 12] = fragC01[0].w;

    lds[  64 * output_row + output_col + 16 ] = fragC10[0].x;
    lds[  64 * output_row + output_col + 16 + 4] = fragC10[0].y;
    lds[  64 * output_row + output_col + 16 + 8] = fragC10[0].z;
    lds[  64 * output_row + output_col + 16 + 12] = fragC10[0].w;

    lds[  64 * (output_row + 16) + output_col + 16] = fragC11[0].x;
    lds[  64 * (output_row + 16) + output_col + 16 + 4] = fragC11[0].y;
    lds[  64 * (output_row + 16) + output_col + 16 + 8] = fragC11[0].z;
    lds[  64 * (output_row + 16) + output_col + 16 + 12] = fragC11[0].w;
    __syncthreads();
    row = tid / 4; //每行由2个线程来处理，每个线程处理16个数据
    col = (tid % 4) * 16;
    FETCH_FLOAT4(param.pout[C2POUT(param.block_size_n * blockIdx.x + col, param.block_size_m * blockIdx.y  + row,param)]) = FETCH_FLOAT4(lds[64 * row + col]);
    FETCH_FLOAT4(param.pout[C2POUT(param.block_size_n * blockIdx.x + col + 8, param.block_size_m * blockIdx.y + row,param)]) = FETCH_FLOAT4(lds[64 * row + col + 8]);
    //2
    __syncthreads();
    lds[ 64 * output_row + output_col] = fragC00[1].x;
    lds[  64 * output_row + output_col + 4] = fragC00[1].y;
    lds[  64 * output_row + output_col + 8] = fragC00[1].z;
    lds[  64 * output_row + output_col + 12] = fragC00[1].w;

    lds[  64 * (output_row + 16) + output_col] = fragC01[1].x;
    lds[  64 * (output_row + 16) + output_col + 4] = fragC01[1].y;
    lds[  64 * (output_row + 16) + output_col + 8] = fragC01[1].z;
    lds[  64 * (output_row + 16) + output_col + 12] = fragC01[1].w;

    lds[  64 * output_row + output_col + 16 ] = fragC10[1].x;
    lds[  64 * output_row + output_col + 16 + 4] = fragC10[1].y;
    lds[  64 * output_row + output_col + 16 + 8] = fragC10[1].z;
    lds[  64 * output_row + output_col + 16 + 12] = fragC10[1].w;

    lds[  64 * (output_row + 16) + output_col + 16] = fragC11[1].x;
    lds[  64 * (output_row + 16) + output_col + 16 + 4] = fragC11[1].y;
    lds[  64 * (output_row + 16) + output_col + 16 + 8] = fragC11[1].z;
    lds[  64 * (output_row + 16) + output_col + 16 + 12] = fragC11[1].w;
    __syncthreads();
    row = tid / 4; //每行由2个线程来处理，每个线程处理16个数据
    col = (tid % 4) * 16;
    FETCH_FLOAT4(param.pout[C2POUT(param.block_size_n * blockIdx.x + col, param.block_size_m * blockIdx.y + 64 + row,param)]) = FETCH_FLOAT4(lds[64 * row + col]);
    FETCH_FLOAT4(param.pout[C2POUT(param.block_size_n * blockIdx.x + col + 8, param.block_size_m * blockIdx.y + 64 + row,param)]) = FETCH_FLOAT4(lds[64 * row + col + 8]);

}

// 第4组
extern "C" __global__ void t4_implicit_gemm_64x16x64_256(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{
    // shared memory
    __shared__ _Float16 lds[2*64*16 + 2*16*64];

    unsigned int K = param.r * param.s *param.c;

    const int tid = threadIdx.x;

    _Float16 ldg_a_reg[4];
    _Float16 ldg_b_reg[4];

    //transfer first tile from global mem to shared mem
    // load B from global memory to shared memory
    int row = (tid % 128) / 8;
    int col = (tid % 8) * 4;
    int G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
    int G_y = row;
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col] = B2PIN(G_x,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 1] = B2PIN(G_x + 1,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 2] = B2PIN(G_x + 2,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 3] = B2PIN(G_x + 3,G_y,param);
    
    // load A from global memory to shared memory
    row = (tid % 128) / 4; 
    col = (tid % 4) * 4; 
    G_x = col ;
    G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
    FETCH_FLOAT2(ldg_a_reg[0]) = FETCH_FLOAT2(param.pweight[G_y*K + G_x]);
    lds[(tid / 128) * 32 * 16 + col*32 + row] = ldg_a_reg[0];
    lds[(tid / 128) * 32 * 16 + (col+1)*32 + row] = ldg_a_reg[1];
    lds[(tid / 128) * 32 * 16 + (col+2)*32 + row] = ldg_a_reg[2];
    lds[(tid / 128) * 32 * 16 + (col+3)*32 + row] = ldg_a_reg[3];
    
    __syncthreads();

    RegisterUnion fragA, fragB;
    float4_ fragC00, fragC01, fragC10, fragC11;
    fragC00 = {0, 0, 0, 0};
    fragC01 = {0, 0, 0, 0};
    fragC10 = {0, 0, 0, 0};
    fragC11 = {0, 0, 0, 0};


    int write_stage_idx = 1;
    int load_stage_idx = write_stage_idx ^ 1;
    int tile_idx = param.block_size_k;

    // 大循环逻辑
    do{
        // 循环1
        // load next tile from global mem to share mem:步骤1
        // load A 
        row = (tid % 128) / 4; 
        col = (tid % 4)*4; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
        FETCH_FLOAT2(ldg_a_reg[0]) = FETCH_FLOAT2(param.pweight[G_y*K + G_x]);

        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        ldg_b_reg[0] = B2PIN(G_x,G_y,param);
        ldg_b_reg[1] = B2PIN(G_x + 1,G_y,param);
        ldg_b_reg[2] = B2PIN(G_x + 2,G_y,param);
        ldg_b_reg[3] = B2PIN(G_x + 3,G_y,param);

        // load from share mem to register
        int lds_read_A_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        int lds_read_B_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + ((tid % 128) / 64) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector4), "+v"(lds_read_A_offset));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B_offset));

        asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01), "+v"(fragA.vector_rear), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));


        // load next tile from global mem to share mem:步骤2
        // load A
        row = (tid % 128) / 4; 
        col = (tid % 4)*4; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + col*32 + row] = ldg_a_reg[0];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+1)*32 + row] = ldg_a_reg[1];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+2)*32 + row] = ldg_a_reg[2];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+3)*32 + row] = ldg_a_reg[3];
        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        FETCH_FLOAT2(lds[2 * param.block_size_k * param.block_size_m + write_stage_idx * param.block_size_k * param.block_size_n + (tid / 128) * 16 * 32 + row *32 + col]) = FETCH_FLOAT2(ldg_b_reg[0]);
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11), "+v"(fragA.vector_rear), "+v"(fragB.vector_rear));
        // switch  
        write_stage_idx ^= 1;
        load_stage_idx = write_stage_idx ^ 1;
        tile_idx += param.block_size_k;
        __syncthreads();

        // 循环2
        // load next tile from global mem to share mem:步骤1
        // load A 
        row = (tid % 128) / 4; 
        col = (tid % 4)*4; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
        FETCH_FLOAT2(ldg_a_reg[0]) = FETCH_FLOAT2(param.pweight[G_y*K + G_x]);

        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        ldg_b_reg[0] = B2PIN(G_x,G_y,param);
        ldg_b_reg[1] = B2PIN(G_x + 1,G_y,param);
        ldg_b_reg[2] = B2PIN(G_x + 2,G_y,param);
        ldg_b_reg[3] = B2PIN(G_x + 3,G_y,param);

        // load from share mem to register
        lds_read_A_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        lds_read_B_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + ((tid % 128) / 64) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector4), "+v"(lds_read_A_offset));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B_offset));

        asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01), "+v"(fragA.vector_rear), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));


        // load next tile from global mem to share mem:步骤2
        // load A
        row = (tid % 128) / 4; 
        col = (tid % 4)*4; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + col*32 + row] = ldg_a_reg[0];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+1)*32 + row] = ldg_a_reg[1];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+2)*32 + row] = ldg_a_reg[2];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+3)*32 + row] = ldg_a_reg[3];
        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        FETCH_FLOAT2(lds[2 * param.block_size_k * param.block_size_m + write_stage_idx * param.block_size_k * param.block_size_n + (tid / 128) * 16 * 32 + row *32 + col]) = FETCH_FLOAT2(ldg_b_reg[0]);
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11), "+v"(fragA.vector_rear), "+v"(fragB.vector_rear));
        // switch  
        write_stage_idx ^= 1;
        load_stage_idx = write_stage_idx ^ 1;
        tile_idx += param.block_size_k;
        __syncthreads();

        // 循环3
        // load next tile from global mem to share mem:步骤1
        // load A 
        row = (tid % 128) / 4; 
        col = (tid % 4)*4; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
        FETCH_FLOAT2(ldg_a_reg[0]) = FETCH_FLOAT2(param.pweight[G_y*K + G_x]);

        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        ldg_b_reg[0] = B2PIN(G_x,G_y,param);
        ldg_b_reg[1] = B2PIN(G_x + 1,G_y,param);
        ldg_b_reg[2] = B2PIN(G_x + 2,G_y,param);
        ldg_b_reg[3] = B2PIN(G_x + 3,G_y,param);

        // load from share mem to register
        lds_read_A_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        lds_read_B_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + ((tid % 128) / 64) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector4), "+v"(lds_read_A_offset));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B_offset));

        asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01), "+v"(fragA.vector_rear), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));


        // load next tile from global mem to share mem:步骤2
        // load A
        row = (tid % 128) / 4; 
        col = (tid % 4)*4; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + col*32 + row] = ldg_a_reg[0];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+1)*32 + row] = ldg_a_reg[1];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+2)*32 + row] = ldg_a_reg[2];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+3)*32 + row] = ldg_a_reg[3];
        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        FETCH_FLOAT2(lds[2 * param.block_size_k * param.block_size_m + write_stage_idx * param.block_size_k * param.block_size_n + (tid / 128) * 16 * 32 + row *32 + col]) = FETCH_FLOAT2(ldg_b_reg[0]);
        //asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11), "+v"(fragA.vector_rear), "+v"(fragB.vector_rear));
        // switch  
        write_stage_idx ^= 1;
        load_stage_idx = write_stage_idx ^ 1;
        tile_idx += param.block_size_k;
        __syncthreads();

        // 循环4
        // load next tile from global mem to share mem:步骤1
        // load A 
        row = (tid % 128) / 4; 
        col = (tid % 4)*4; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
        FETCH_FLOAT2(ldg_a_reg[0]) = FETCH_FLOAT2(param.pweight[G_y*K + G_x]);

        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        ldg_b_reg[0] = B2PIN(G_x,G_y,param);
        ldg_b_reg[1] = B2PIN(G_x + 1,G_y,param);
        ldg_b_reg[2] = B2PIN(G_x + 2,G_y,param);
        ldg_b_reg[3] = B2PIN(G_x + 3,G_y,param);

        // load from share mem to register
        lds_read_A_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        lds_read_B_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + ((tid % 128) / 64) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector4), "+v"(lds_read_A_offset));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B_offset));

        asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01), "+v"(fragA.vector_rear), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));


        // load next tile from global mem to share mem:步骤2
        // load A
        row = (tid % 128) / 4; 
        col = (tid % 4)*4; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + col*32 + row] = ldg_a_reg[0];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+1)*32 + row] = ldg_a_reg[1];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+2)*32 + row] = ldg_a_reg[2];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+3)*32 + row] = ldg_a_reg[3];
        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        FETCH_FLOAT2(lds[2 * param.block_size_k * param.block_size_m + write_stage_idx * param.block_size_k * param.block_size_n + (tid / 128) * 16 * 32 + row *32 + col]) = FETCH_FLOAT2(ldg_b_reg[0]);
        //asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11), "+v"(fragA.vector_rear), "+v"(fragB.vector_rear));
        // switch  
        write_stage_idx ^= 1;
        load_stage_idx = write_stage_idx ^ 1;
        tile_idx += param.block_size_k;
        __syncthreads();
    }while(tile_idx< (K - 4*param.block_size_k));

    //完成后四轮的前3轮
    do{
        // load next tile from global mem to share mem:步骤1
        // load A 
        row = (tid % 128) / 4; 
        col = (tid % 4)*4; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
        FETCH_FLOAT2(ldg_a_reg[0]) = FETCH_FLOAT2(param.pweight[G_y*K + G_x]);

        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        ldg_b_reg[0] = B2PIN(G_x,G_y,param);
        ldg_b_reg[1] = B2PIN(G_x + 1,G_y,param);
        ldg_b_reg[2] = B2PIN(G_x + 2,G_y,param);
        ldg_b_reg[3] = B2PIN(G_x + 3,G_y,param);

        // load from share mem to register
        int lds_read_A_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        int lds_read_B_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + ((tid % 128) / 64) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector4), "+v"(lds_read_A_offset));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B_offset));

        asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01), "+v"(fragA.vector_rear), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));

        // load next tile from global mem to share mem:步骤2
        // load A
        row = (tid % 128) / 4; 
        col = (tid % 4)*4; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + col*32 + row] = ldg_a_reg[0];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+1)*32 + row] = ldg_a_reg[1];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+2)*32 + row] = ldg_a_reg[2];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+3)*32 + row] = ldg_a_reg[3];
        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        FETCH_FLOAT2(lds[2 * param.block_size_k * param.block_size_m + write_stage_idx * param.block_size_k * param.block_size_n + (tid / 128) * 16 * 32 + row *32 + col]) = FETCH_FLOAT2(ldg_b_reg[0]);
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11), "+v"(fragA.vector_rear), "+v"(fragB.vector_rear));
        // switch  
        write_stage_idx ^= 1;
        load_stage_idx = write_stage_idx ^ 1;
        tile_idx += param.block_size_k;
        __syncthreads();
    }while(tile_idx< K);

    // 完成最后一次计算
    int lds_read_A_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
    int lds_read_B_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + ((tid % 128) / 64) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
    asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector4), "+v"(lds_read_A_offset));
    asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B_offset));
    asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01), "+v"(fragA.vector_rear), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11), "+v"(fragA.vector_rear), "+v"(fragB.vector_rear));
    __syncthreads();

    // 直接写回
    uint32_t output_row = (tid / 128) * 32 + ((tid % 64) % 16);
    uint32_t output_col = ((tid % 128) / 64) * 32 + ((tid % 64) / 16);

    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col, param.block_size_m * blockIdx.y +  output_row,param)] =  (_Float16)fragC00.x;
    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 4, param.block_size_m * blockIdx.y + output_row,param)] = (_Float16) fragC00.y;
    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 8, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC00.z;
    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 12, param.block_size_m * blockIdx.y + output_row,param)] = (_Float16) fragC00.w;

    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC01.x;
    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 4, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC01.y;
    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 8, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC01.z;
    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 12, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC01.w;


    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC10.x;
    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 4, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC10.y;
    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 8, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC10.z;
    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 12, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC10.w;

    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC11.x;
    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 4, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC11.y;
    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 8, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC11.z;
    param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 12, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC11.w;

}


// 第3组
extern "C" __global__ void t3_implicit_gemm_64x16x64_256(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{
    // shared memory
    __shared__ _Float16 lds[2*64*16 + 2*16*64];

    unsigned int K = param.r * param.s *param.c;

    const int tid = threadIdx.x;

    _Float16 ldg_a_reg[4];
    _Float16 ldg_b_reg[4];

    //transfer first tile from global mem to shared mem
    // load B from global memory to shared memory
    int row = (tid % 128) / 8;
    int col = (tid % 8) * 4;
    int G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
    int G_y = row;
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col] = B2PIN(G_x,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 1] = B2PIN(G_x + 1,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 2] = B2PIN(G_x + 2,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 3] = B2PIN(G_x + 3,G_y,param);
    
    // load A from global memory to shared memory
    row = (tid % 128) / 4; 
    col = (tid % 4) * 4; 
    G_x = col ;
    G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
    FETCH_FLOAT2(ldg_a_reg[0]) = FETCH_FLOAT2(param.pweight[G_y*K + G_x]);
    lds[(tid / 128) * 32 * 16 + col*32 + row] = ldg_a_reg[0];
    lds[(tid / 128) * 32 * 16 + (col+1)*32 + row] = ldg_a_reg[1];
    lds[(tid / 128) * 32 * 16 + (col+2)*32 + row] = ldg_a_reg[2];
    lds[(tid / 128) * 32 * 16 + (col+3)*32 + row] = ldg_a_reg[3];
    
    __syncthreads();

    RegisterUnion fragA, fragB;
    float4_ fragC00, fragC01, fragC10, fragC11;
    fragC00 = {0, 0, 0, 0};
    fragC01 = {0, 0, 0, 0};
    fragC10 = {0, 0, 0, 0};
    fragC11 = {0, 0, 0, 0};


    int write_stage_idx = 1;
    int load_stage_idx = write_stage_idx ^ 1;
    int tile_idx = param.block_size_k;

    // 大循环逻辑
    do{
        // load next tile from global mem to share mem:步骤1
        // load A 
        row = (tid % 128) / 4; 
        col = (tid % 4)*4; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
        FETCH_FLOAT2(ldg_a_reg[0]) = FETCH_FLOAT2(param.pweight[G_y*K + G_x]);

        int lds_read_A_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        int lds_read_B_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + ((tid % 128) / 64) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector4), "+v"(lds_read_A_offset));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B_offset));
        
        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        ldg_b_reg[0] = B2PIN(G_x,G_y,param);
        ldg_b_reg[1] = B2PIN(G_x + 1,G_y,param);
        ldg_b_reg[2] = B2PIN(G_x + 2,G_y,param);
        ldg_b_reg[3] = B2PIN(G_x + 3,G_y,param);
        
        // load from share mem to register
        asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01), "+v"(fragA.vector_rear), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));

    
        // load next tile from global mem to share mem:步骤2
        // load A
        row = (tid % 128) / 4; 
        col = (tid % 4)*4; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + col*32 + row] = ldg_a_reg[0];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+1)*32 + row] = ldg_a_reg[1];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+2)*32 + row] = ldg_a_reg[2];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+3)*32 + row] = ldg_a_reg[3];
        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        FETCH_FLOAT2(lds[2 * param.block_size_k * param.block_size_m + write_stage_idx * param.block_size_k * param.block_size_n + (tid / 128) * 16 * 32 + row *32 + col]) = FETCH_FLOAT2(ldg_b_reg[0]);
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11), "+v"(fragA.vector_rear), "+v"(fragB.vector_rear));
        // switch  
        write_stage_idx ^= 1;
        load_stage_idx = write_stage_idx ^ 1;
        tile_idx += param.block_size_k;
        __syncthreads();
    }while(tile_idx< K);
    
    // 完成最后一次计算
    int lds_read_A_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
    int lds_read_B_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + ((tid % 128) / 64) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
    asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector4), "+v"(lds_read_A_offset));
    asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B_offset));
    asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01), "+v"(fragA.vector_rear), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11), "+v"(fragA.vector_rear), "+v"(fragB.vector_rear));

    //先写回share mem
    // store back to share mem:此处复用share mem
    uint32_t output_row = (tid / 128) * 32 + (tid % 64) %16; 
    uint32_t output_col = ((tid % 128) / 64) * 32 + ((tid % 64) / 16);
    __syncthreads();
    lds[ 64 * output_row + output_col] = fragC00.x;
    lds[  64 * output_row + output_col + 4] = fragC00.y;
    lds[  64 * output_row + output_col + 8] = fragC00.z;
    lds[  64 * output_row + output_col + 12] = fragC00.w;

    lds[  64 * (output_row + 16) + output_col] = fragC01.x;
    lds[  64 * (output_row + 16) + output_col + 4] = fragC01.y;
    lds[  64 * (output_row + 16) + output_col + 8] = fragC01.z;
    lds[  64 * (output_row + 16) + output_col + 12] = fragC01.w;

    lds[  64 * output_row + output_col + 16 ] = fragC10.x;
    lds[  64 * output_row + output_col + 16 + 4] = fragC10.y;
    lds[  64 * output_row + output_col + 16 + 8] = fragC10.z;
    lds[  64 * output_row + output_col + 16 + 12] = fragC10.w;

    lds[  64 * (output_row + 16) + output_col + 16] = fragC11.x;
    lds[  64 * (output_row + 16) + output_col + 16 + 4] = fragC11.y;
    lds[  64 * (output_row + 16) + output_col + 16 + 8] = fragC11.z;
    lds[  64 * (output_row + 16) + output_col + 16 + 12] = fragC11.w;
    __syncthreads();
    row = tid / 4; //每行由2个线程来处理，每个线程处理16个数据
    col = (tid % 4) * 16;
    FETCH_FLOAT4(param.pout[C2POUT(param.block_size_n * blockIdx.x + col, param.block_size_m * blockIdx.y  + row,param)]) = FETCH_FLOAT4(lds[64 * row + col]);
    FETCH_FLOAT4(param.pout[C2POUT(param.block_size_n * blockIdx.x + col + 8, param.block_size_m * blockIdx.y + row,param)]) = FETCH_FLOAT4(lds[64 * row + col + 8]);

}


// 第2组
extern "C" __global__ void t2_implicit_gemm_128x16x64_256(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{
    // shared memory
    __shared__ _Float16 lds[4*64*16 + 2*16*64];

    unsigned int K = param.r * param.s *param.c;

    const int tid = threadIdx.x;

    _Float16 ldg_a_reg[8];
    _Float16 ldg_b_reg[4];

    //transfer first tile from global mem to shared mem
    // load B from global memory to shared memory
    int row = (tid % 128) / 8;
    int col = (tid % 8) * 4;
    int G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
    int G_y = row;
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col] = B2PIN(G_x,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 1] = B2PIN(G_x + 1,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 2] = B2PIN(G_x + 2,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 3] = B2PIN(G_x + 3,G_y,param);
    
    // load A from global memory to shared memory
    //1
    row = (tid % 64) / 2; 
    col = (tid % 2) * 8; 
    G_x = col ;
    G_y = param.block_size_m * blockIdx.y + (tid / 64) * 32 + row;
    FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(param.pweight[G_y*K + G_x]);
    lds[(tid / 64) * 32 * 16 + col*32 + row] = ldg_a_reg[0];
    lds[(tid / 64) * 32 * 16 + (col+1)*32 + row] = ldg_a_reg[1];
    lds[(tid / 64) * 32 * 16 + (col+2)*32 + row] = ldg_a_reg[2];
    lds[(tid / 64) * 32 * 16 + (col+3)*32 + row] = ldg_a_reg[3];
    lds[(tid / 64) * 32 * 16 + (col+4)*32 + row] = ldg_a_reg[4];
    lds[(tid / 64) * 32 * 16 + (col+5)*32 + row] = ldg_a_reg[5];
    lds[(tid / 64) * 32 * 16 + (col+6)*32 + row] = ldg_a_reg[6];
    lds[(tid / 64) * 32 * 16 + (col+7)*32 + row] = ldg_a_reg[7];

    
    __syncthreads();

    RegisterUnion fragA[2], fragB;
    float4_ fragC00[2], fragC01[2], fragC10[2], fragC11[2];
    fragC00[0] = {0, 0, 0, 0};
    fragC01[0] = {0, 0, 0, 0};
    fragC10[0] = {0, 0, 0, 0};
    fragC11[0] = {0, 0, 0, 0};
    
    fragC00[1] = {0, 0, 0, 0};
    fragC01[1] = {0, 0, 0, 0};
    fragC10[1] = {0, 0, 0, 0};
    fragC11[1] = {0, 0, 0, 0};

    int write_stage_idx = 1;
    int load_stage_idx = write_stage_idx ^ 1;
    int tile_idx = param.block_size_k;

    // 大循环逻辑
    do{
        // load next tile from global mem to share mem:步骤1
        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        ldg_b_reg[0] = B2PIN(G_x,G_y,param);
        ldg_b_reg[1] = B2PIN(G_x + 1,G_y,param);
        ldg_b_reg[2] = B2PIN(G_x + 2,G_y,param);
        ldg_b_reg[3] = B2PIN(G_x + 3,G_y,param);
        
        // load from share mem to register
        int lds_read_A0_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        int lds_read_A1_offset = (64*16 + load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        int lds_read_B_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + ((tid % 128) / 64) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA[0].vector4), "+v"(lds_read_A0_offset));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA[1].vector4), "+v"(lds_read_A1_offset));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B_offset));
        
        
        // load A 
        row = (tid % 64) / 2; 
        col = (tid % 2)*8; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 64) * 32 + row;
        FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(param.pweight[G_y*K + G_x]);
    

        
        asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00[0]), "+v"(fragA[0].vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01[0]), "+v"(fragA[0].vector_rear), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10[0]), "+v"(fragA[0].vector_front), "+v"(fragB.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11[0]), "+v"(fragA[0].vector_rear), "+v"(fragB.vector_rear));
            
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00[1]), "+v"(fragA[1].vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01[1]), "+v"(fragA[1].vector_rear), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10[1]), "+v"(fragA[1].vector_front), "+v"(fragB.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11[1]), "+v"(fragA[1].vector_rear), "+v"(fragB.vector_rear));
    

        // load next tile from global mem to share mem:步骤2
        // load A
        row = (tid % 64) / 2; 
        col = (tid % 2)*8; 
        G_x = col + tile_idx;
        G_y = param.block_size_m * blockIdx.y + (tid / 64) * 32 + row;
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + col*32 + row] = ldg_a_reg[0];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+1)*32 + row] = ldg_a_reg[1];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+2)*32 + row] = ldg_a_reg[2];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+3)*32 + row] = ldg_a_reg[3];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+4)*32 + row] = ldg_a_reg[4];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+5)*32 + row] = ldg_a_reg[5];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+6)*32 + row] = ldg_a_reg[6];
        lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 64) * 32*16 + (col+7)*32 + row] = ldg_a_reg[7];

        // load B
        row = (tid % 128) / 8;
        col = (tid % 8) * 4;
        G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
        G_y = row + tile_idx;
        FETCH_FLOAT2(lds[2 * param.block_size_k * param.block_size_m + write_stage_idx * param.block_size_k * param.block_size_n + (tid / 128) * 16 * 32 + row *32 + col]) = FETCH_FLOAT2(ldg_b_reg[0]);
        

        // switch  
        write_stage_idx ^= 1;
        load_stage_idx = write_stage_idx ^ 1;
        tile_idx += param.block_size_k;
        __syncthreads();
    }while(tile_idx< K);
    
    // 完成最后一次计算
    int lds_read_A0_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
    int lds_read_A1_offset = (64*16 + load_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
    int lds_read_B_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + ((tid % 128) / 64) * 16 * 32 + (tid % 64) * 8) * sizeof(_Float16);
    asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA[0].vector4), "+v"(lds_read_A0_offset));
    asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA[1].vector4), "+v"(lds_read_A1_offset));
    asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B_offset));

    // load from share mem to register
    asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00[0]), "+v"(fragA[0].vector_front), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01[0]), "+v"(fragA[0].vector_rear), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10[0]), "+v"(fragA[0].vector_front), "+v"(fragB.vector_rear));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11[0]), "+v"(fragA[0].vector_rear), "+v"(fragB.vector_rear));
        
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00[1]), "+v"(fragA[1].vector_front), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01[1]), "+v"(fragA[1].vector_rear), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10[1]), "+v"(fragA[1].vector_front), "+v"(fragB.vector_rear));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11[1]), "+v"(fragA[1].vector_rear), "+v"(fragB.vector_rear));   

    // store back to share mem:此处复用share mem
    uint32_t output_row = (tid / 128) * 32 + (tid % 64) %16; 
    uint32_t output_col = ((tid % 128) / 64) * 32 + ((tid % 64) / 16);
    //1
    __syncthreads();
    lds[ 64 * output_row + output_col] = fragC00[0].x;
    lds[  64 * output_row + output_col + 4] = fragC00[0].y;
    lds[  64 * output_row + output_col + 8] = fragC00[0].z;
    lds[  64 * output_row + output_col + 12] = fragC00[0].w;

    lds[  64 * (output_row + 16) + output_col] = fragC01[0].x;
    lds[  64 * (output_row + 16) + output_col + 4] = fragC01[0].y;
    lds[  64 * (output_row + 16) + output_col + 8] = fragC01[0].z;
    lds[  64 * (output_row + 16) + output_col + 12] = fragC01[0].w;

    lds[  64 * output_row + output_col + 16 ] = fragC10[0].x;
    lds[  64 * output_row + output_col + 16 + 4] = fragC10[0].y;
    lds[  64 * output_row + output_col + 16 + 8] = fragC10[0].z;
    lds[  64 * output_row + output_col + 16 + 12] = fragC10[0].w;

    lds[  64 * (output_row + 16) + output_col + 16] = fragC11[0].x;
    lds[  64 * (output_row + 16) + output_col + 16 + 4] = fragC11[0].y;
    lds[  64 * (output_row + 16) + output_col + 16 + 8] = fragC11[0].z;
    lds[  64 * (output_row + 16) + output_col + 16 + 12] = fragC11[0].w;
    __syncthreads();
    row = tid / 4; //每行由2个线程来处理，每个线程处理16个数据
    col = (tid % 4) * 16;
    FETCH_FLOAT4(param.pout[C2POUT(param.block_size_n * blockIdx.x + col, param.block_size_m * blockIdx.y  + row,param)]) = FETCH_FLOAT4(lds[64 * row + col]);
    FETCH_FLOAT4(param.pout[C2POUT(param.block_size_n * blockIdx.x + col + 8, param.block_size_m * blockIdx.y + row,param)]) = FETCH_FLOAT4(lds[64 * row + col + 8]);
    //2
    __syncthreads();
    lds[ 64 * output_row + output_col] = fragC00[1].x;
    lds[  64 * output_row + output_col + 4] = fragC00[1].y;
    lds[  64 * output_row + output_col + 8] = fragC00[1].z;
    lds[  64 * output_row + output_col + 12] = fragC00[1].w;

    lds[  64 * (output_row + 16) + output_col] = fragC01[1].x;
    lds[  64 * (output_row + 16) + output_col + 4] = fragC01[1].y;
    lds[  64 * (output_row + 16) + output_col + 8] = fragC01[1].z;
    lds[  64 * (output_row + 16) + output_col + 12] = fragC01[1].w;

    lds[  64 * output_row + output_col + 16 ] = fragC10[1].x;
    lds[  64 * output_row + output_col + 16 + 4] = fragC10[1].y;
    lds[  64 * output_row + output_col + 16 + 8] = fragC10[1].z;
    lds[  64 * output_row + output_col + 16 + 12] = fragC10[1].w;

    lds[  64 * (output_row + 16) + output_col + 16] = fragC11[1].x;
    lds[  64 * (output_row + 16) + output_col + 16 + 4] = fragC11[1].y;
    lds[  64 * (output_row + 16) + output_col + 16 + 8] = fragC11[1].z;
    lds[  64 * (output_row + 16) + output_col + 16 + 12] = fragC11[1].w;
    __syncthreads();
    row = tid / 4; //每行由2个线程来处理，每个线程处理16个数据
    col = (tid % 4) * 16;
    FETCH_FLOAT4(param.pout[C2POUT(param.block_size_n * blockIdx.x + col, param.block_size_m * blockIdx.y + 64 + row,param)]) = FETCH_FLOAT4(lds[64 * row + col]);
    FETCH_FLOAT4(param.pout[C2POUT(param.block_size_n * blockIdx.x + col + 8, param.block_size_m * blockIdx.y + 64 + row,param)]) = FETCH_FLOAT4(lds[64 * row + col + 8]);

}

// 第1组
extern "C" __global__ void t1_implicit_gemm_64x16x64_256(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{
    // shared memory
    __shared__ _Float16 lds[2*64*16 + 2*16*64];

    unsigned int K = param.r * param.s *param.c;

    const int tid = threadIdx.x;

    _Float16 ldg_a_reg[4];
    _Float16 ldg_b_reg[4];

    //transfer first tile from global mem to shared mem
    // load A from global memory to shared memory
    int row = (tid % 128) / 4; 
    int col = (tid % 4) * 4; 
    int G_x = col ;
    int G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
    if (tid < 108){
        FETCH_FLOAT2(ldg_a_reg[0]) = FETCH_FLOAT2(param.pweight[G_y*K + G_x]);
    }
    lds[(tid / 128) * 32 * 16 + col*32 + row] = ldg_a_reg[0];
    lds[(tid / 128) * 32 * 16 + (col+1)*32 + row] = ldg_a_reg[1];
    lds[(tid / 128) * 32 * 16 + (col+2)*32 + row] = ldg_a_reg[2];
    lds[(tid / 128) * 32 * 16 + (col+3)*32 + row] = ldg_a_reg[3];
    
    // load B from global memory to shared memory
    row = (tid % 128) / 8;
    col = (tid % 8) * 4;
    G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
    G_y = row;
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col] = B2PIN(G_x,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 1] = B2PIN(G_x + 1,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 2] = B2PIN(G_x + 2,G_y,param);
    lds[2*param.block_size_k*param.block_size_m + (tid / 128) * 32*16 + row *32 + col + 3] = B2PIN(G_x + 3,G_y,param);
    
    __syncthreads();

    RegisterUnion fragA, fragB;
    float4_ fragC00, fragC01, fragC10, fragC11;
    fragC00 = {0, 0, 0, 0};
    fragC01 = {0, 0, 0, 0};
    fragC10 = {0, 0, 0, 0};
    fragC11 = {0, 0, 0, 0};


    int write_stage_idx = 1;
    int load_stage_idx = write_stage_idx ^ 1;
    int tile_idx = 0;

    // 大循环逻辑
    do{
        tile_idx += param.block_size_k;
        // load from share mem to register
        if(tid < 128){
            if(tid < 64){       // A0B0
                int lds_read_A0_offset = (load_stage_idx * param.block_size_k * param.block_size_m + tid * 8) * sizeof(_Float16);
                int lds_read_B0_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + tid * 8) * sizeof(_Float16);
                asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector4), "+v"(lds_read_A0_offset));
                asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B0_offset));

            }
            else if(tid >= 64 && tid < 128){        //A0B1
                int lds_read_A0_offset = (load_stage_idx * param.block_size_k * param.block_size_m + (tid - 64) * 8) * sizeof(_Float16);
                int lds_read_B1_offset = (2 * param.block_size_k * param.block_size_m + load_stage_idx * param.block_size_k * param.block_size_n + 16 * 32 +  (tid - 64) * 8) * sizeof(_Float16);
                asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector4), "+v"(lds_read_A0_offset));
                asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector4), "+v"(lds_read_B1_offset));
            }
        }

        // load next tile from global mem to share mem:步骤1
        if(tile_idx< K){
            // load A 
            row = (tid % 128) / 4; 
            col = (tid % 4)*4; 
            G_x = col + tile_idx;
            G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
            if (tid < 108){
                 FETCH_FLOAT2(ldg_a_reg[0]) = FETCH_FLOAT2(param.pweight[G_y*K + G_x]);
            }
            
            // load B
            row = (tid % 128) / 8;
            col = (tid % 8) * 4;
            G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
            G_y = row + tile_idx;
            ldg_b_reg[0] = B2PIN(G_x,G_y,param);
            ldg_b_reg[1] = B2PIN(G_x + 1,G_y,param);
            ldg_b_reg[2] = B2PIN(G_x + 2,G_y,param);
            ldg_b_reg[3] = B2PIN(G_x + 3,G_y,param);

           

            if( tid < 128){
                asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
                asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
                asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01), "+v"(fragA.vector_rear), "+v"(fragB.vector_front));
                asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));
                asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11), "+v"(fragA.vector_rear), "+v"(fragB.vector_rear));
            }
            row = (tid % 128) / 4; 
            col = (tid % 4)*4; 
            G_x = col + tile_idx;
            G_y = param.block_size_m * blockIdx.y + (tid / 128) * 32 + row;
            lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + col*32 + row] = ldg_a_reg[0];
            lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+1)*32 + row] = ldg_a_reg[1];
            lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+2)*32 + row] = ldg_a_reg[2];
            lds[write_stage_idx * param.block_size_k * param.block_size_m + (tid / 128) * 32*16 + (col+3)*32 + row] = ldg_a_reg[3];
            // load B
            row = (tid % 128) / 8;
            col = (tid % 8) * 4;
            G_x = param.block_size_n * blockIdx.x + (tid / 128) * 32 + col;
            G_y = row + tile_idx;
            FETCH_FLOAT2(lds[2 * param.block_size_k * param.block_size_m + write_stage_idx * param.block_size_k * param.block_size_n + (tid / 128) * 16 * 32 + row *32 + col]) = FETCH_FLOAT2(ldg_b_reg[0]);

            __syncthreads();
        }
        // switch  
        write_stage_idx ^= 1;
        load_stage_idx = write_stage_idx ^ 1;
    }while(tile_idx< K);
    
    // 完成最后一次计算
    asm volatile("s_waitcnt lgkmcnt(0)\n\t"); 
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01), "+v"(fragA.vector_rear), "+v"(fragB.vector_front));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11), "+v"(fragA.vector_rear), "+v"(fragB.vector_rear));
    __syncthreads();

    // 直接写回
    uint32_t output_row,output_col;
    if(tid < 64){
        output_row = tid % 16;
        output_col = tid / 16;
    }
    else if( tid >= 64 && tid < 128){
        output_row = (tid - 64) & 15;
        output_col = ((tid - 64) >> 4) + 32;
    }
    else if( tid >= 128 && tid < 192){
        output_row = ((tid - 128) & 15) + 32;
        output_col = ((tid - 128) >> 4);
    }
    else if( tid >= 192 && tid < 256){
        output_row = ((tid - 192) & 15) + 32;
        output_col = ((tid - 192) >> 4) + 32;
    }
    if(tid < 128){
        param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col, param.block_size_m * blockIdx.y +  output_row,param)] =  (_Float16)fragC00.x;
        param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 4, param.block_size_m * blockIdx.y + output_row,param)] = (_Float16) fragC00.y;
        param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 8, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC00.z;
        param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 12, param.block_size_m * blockIdx.y + output_row,param)] = (_Float16) fragC00.w;


        param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC10.x;
        param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 4, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC10.y;
        param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 8, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC10.z;
        param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 12, param.block_size_m * blockIdx.y + output_row,param)] =  (_Float16)fragC10.w;

        if(output_row + 16 < 27 ){
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC01.x;
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 4, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC01.y;
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 8, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC01.z;
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 12, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC01.w;

            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC11.x;
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 4, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC11.y;
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 8, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC11.z;
            param.pout[C2POUT(param.block_size_n * blockIdx.x + output_col + 16 + 12, param.block_size_m * blockIdx.y + (output_row + 16),param)] =  (_Float16)fragC11.w;
        }
    }

}


/*需要返回自定义kernel入参结构体的size*/
int getParamsize(__in__ problem_t* problem, __out__ int* paramSize)
{
    *paramSize = sizeof(mykernelParamType);

    return 0;
}

/*需要返回自己优化的kernel的grid信息与kernel函数的指针*/
int getkernelInfo(__in__ problem_t* problem, __out__  kernelInfo_t* kernelInfo, __in_out__ void* param)
{
    mykernelParamType* pArgs = (mykernelParamType*)param;

    unsigned int n = problem->n;
    unsigned int c = problem->c;
    unsigned int h = problem->h;
    unsigned int w = problem->w;
    unsigned int k = problem->k;
    unsigned int r = problem->r;
    unsigned int s = problem->s;
    unsigned int u = problem->u;
    unsigned int v = problem->v;
    unsigned int p = problem->p;
    unsigned int q = problem->q;

    unsigned int outh = (h - r + 2*p)/u + 1;
    unsigned int outw = (w - s + 2*q)/v + 1;

    unsigned int block_size_m = 32;
    unsigned int block_size_k = 16;
    unsigned int block_size_n = 32;
    unsigned int a_block_size = 32*16;
    unsigned int b_block_size = 16*32;
    unsigned int thread_x_per_block = 8;
    unsigned int thread_y_per_block = 8;

    unsigned int BLOCK_X_PER_GRID = (outh*outw*n + BLOCK_SIZE_N -1) /BLOCK_SIZE_N;
    unsigned int BLOCK_Y_PER_GRID = (k + BLOCK_SIZE_M - 1)/BLOCK_SIZE_M;
    unsigned int ENABLE_DOUBLE_BUFFER = 1;

    kernelInfo->blockx   = BLOCK_X_PER_GRID;                    //blockx  number
    kernelInfo->blocky   = BLOCK_Y_PER_GRID;                    //blocky  number
    kernelInfo->blockz   = 1;                    //blockz  number
    kernelInfo->threadx  = THREAD_X_PER_BLOCK;              //threadx number per block
    kernelInfo->thready  = THREAD_Y_PER_BLOCK;                   //thready number per block
    kernelInfo->threadz  = 1;                   //threadz number per block
    kernelInfo->dynmicLdsSize = 0 ;

    if( k == 27){    //1
        kernelInfo->blockx   = (outh*outw*n + 64 -1) / 64;                    //blockx  number
        kernelInfo->blocky   = (k + 64 - 1) / 64;                    //blocky  number
        kernelInfo->blockz   = 1;                    //blockz  number
        kernelInfo->threadx  = 256;              //threadx number per block
        kernelInfo->thready  = 1;                   //thready number per block
        kernelInfo->threadz  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize = 0 ;
        kernelInfo->kernelPtr= (void*)t1_implicit_gemm_64x16x64_256;   

        block_size_m = 64;
        block_size_k = 16;
        block_size_n = 64;
        a_block_size = block_size_m*block_size_k;
        b_block_size = block_size_k*block_size_n;
        thread_x_per_block = 256;
        thread_y_per_block = 1;
    }
    else if ( n == 16 && c == 256){     //2
        kernelInfo->blockx   = (outh*outw*n + 64 -1) / 64;                    //blockx  number
        kernelInfo->blocky   = (k + 128 - 1) / 128;                    //blocky  number
        kernelInfo->blockz   = 1;                    //blockz  number
        kernelInfo->threadx  = 256;              //threadx number per block
        kernelInfo->thready  = 1;                   //thready number per block
        kernelInfo->threadz  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize = 0 ;
        kernelInfo->kernelPtr= (void*)t2_implicit_gemm_128x16x64_256;   

        block_size_m = 128;
        block_size_k = 16;
        block_size_n = 64;
        a_block_size = block_size_m*block_size_k;
        b_block_size = block_size_k*block_size_n;
        thread_x_per_block = 256;
        thread_y_per_block = 1;
    }
    else if (n == 16 && c == 64){     //3
        kernelInfo->blockx   = (outh*outw*n + 64 -1) / 64;                    //blockx  number
        kernelInfo->blocky   = (k + 64 - 1) / 64;                    //blocky  number
        kernelInfo->blockz   = 1;                    //blockz  number
        kernelInfo->threadx  = 256;              //threadx number per block
        kernelInfo->thready  = 1;                   //thready number per block
        kernelInfo->threadz  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize = 0 ;
        kernelInfo->kernelPtr= (void*)t3_implicit_gemm_64x16x64_256;   

        block_size_m = 64;
        block_size_k = 16;
        block_size_n = 64;
        a_block_size = block_size_m*block_size_k;
        b_block_size = block_size_k*block_size_n;
        thread_x_per_block = 256;
        thread_y_per_block = 1;
    }
    else if ( c == 1920){     //4
        kernelInfo->blockx   = (outh*outw*n + 64 -1) / 64;                    //blockx  number
        kernelInfo->blocky   = (k + 64 - 1) / 64;                    //blocky  number
        kernelInfo->blockz   = 1;                    //blockz  number
        kernelInfo->threadx  = 256;              //threadx number per block
        kernelInfo->thready  = 1;                   //thready number per block
        kernelInfo->threadz  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize = 0 ;
        kernelInfo->kernelPtr= (void*)t4_implicit_gemm_64x16x64_256;   

        block_size_m = 64;
        block_size_k = 16;
        block_size_n = 64;
        a_block_size = block_size_m*block_size_k;
        b_block_size = block_size_k*block_size_n;
        thread_x_per_block = 256;
        thread_y_per_block = 1;
    }
    else if(c == 640 ){     //5
        kernelInfo->blockx   = (outh*outw*n + 64 -1) / 64;                    //blockx  number
        kernelInfo->blocky   = (k + 128 - 1) / 128;                    //blocky  number
        kernelInfo->blockz   = 1;                    //blockz  number
        kernelInfo->threadx  = 256 ;              //threadx number per block
        kernelInfo->thready  = 1;                   //thready number per block
        kernelInfo->threadz  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize = 0 ;
        kernelInfo->kernelPtr= (void*)t5_implicit_gemm_128x16x64_256;   

        block_size_m = 128;
        block_size_k = 16;
        block_size_n = 64;
        a_block_size = block_size_m*block_size_k;
        b_block_size = block_size_k*block_size_n;
        thread_x_per_block = 256;
        thread_y_per_block = 1;
    }
    else if( k == 4){  //6
        kernelInfo->blockx   = (outh*outw*n + 64 -1) / 64;                    //blockx  number
        kernelInfo->blocky   = (k + 64 - 1) / 64;                    //blocky  number
        kernelInfo->blockz   = 1;                    //blockz  number
        kernelInfo->threadx  = 256;              //threadx number per block
        kernelInfo->thready  = 1;                   //thready number per block
        kernelInfo->threadz  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize = 0 ;
        kernelInfo->kernelPtr= (void*)t6_implicit_gemm_64x16x64_256;   

        block_size_m = 64;
        block_size_k = 16;
        block_size_n = 64;
        a_block_size = block_size_m*block_size_k;
        b_block_size = block_size_k*block_size_n;
        thread_x_per_block = 256;
        thread_y_per_block = 1;
    }

    pArgs->pin = problem->in;
    pArgs->pweight = problem->weight;
    pArgs->pout = problem->out;
    pArgs->n = n;                              //batch szie              default value 1
    pArgs->c = c;                              //channel number          default value 32
    pArgs->h = h;                              //数据高                  default value 32
    pArgs->w = w;                              //数据宽                  default value 32
    pArgs->k = k;                              //卷积核数量              default value 32
    pArgs->r = r;                              //卷积核高                default value 1
    pArgs->s = s;                              //卷积核宽                default value 1
    pArgs->u = u;                              //卷积在高方向上的步长     default value 1
    pArgs->v = v;                              //卷积在宽方向上的步长     default value 1
    pArgs->p = p;                              //卷积在高方向上的补边     default value 0
    pArgs->q = q;                              //卷积在宽方向上的补边     default value 0
    pArgs->Oh = outh;
    pArgs->Ow = outw; 
    pArgs->block_size_m = block_size_m;
    pArgs->block_size_k = block_size_k;
    pArgs->block_size_n = block_size_n;
    pArgs->a_block_size = a_block_size;
    pArgs->b_block_size = b_block_size;    
    pArgs->thread_x_per_block = thread_x_per_block;
    pArgs->thread_y_per_block = thread_y_per_block;  
    return 0;
}


