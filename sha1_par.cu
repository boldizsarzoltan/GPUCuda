#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "sha1.cuh"

#include <stdlib.h>
#include <memory.h>

/****************************** MACROS ******************************/
#define SHA1_BLOCK_SIZE 20              // SHA1 outputs a 20 byte digest

/**************************** DATA TYPES ****************************/
typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[5];
	WORD k[4];
} CUDA_SHA1_CTX;

/****************************** MACROS ******************************/
#ifndef ROTLEFT
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#endif

/*********************** FUNCTION DEFINITIONS ***********************/
__device__  __forceinline__ void cuda_sha1_transform(CUDA_SHA1_CTX *ctx, const BYTE data[])
{
	WORD a, b, c, d, e, i, j, t, m[80];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (data[j] << 24) + (data[j + 1] << 16) + (data[j + 2] << 8) + (data[j + 3]);
	for ( ; i < 80; ++i) {
		m[i] = (m[i - 3] ^ m[i - 8] ^ m[i - 14] ^ m[i - 16]);
		m[i] = (m[i] << 1) | (m[i] >> 31);
	}

	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];

	for (i = 0; i < 20; ++i) {
		t = ROTLEFT(a, 5) + ((b & c) ^ (~b & d)) + e + ctx->k[0] + m[i];
		e = d;
		d = c;
		c = ROTLEFT(b, 30);
		b = a;
		a = t;
	}
	for ( ; i < 40; ++i) {
		t = ROTLEFT(a, 5) + (b ^ c ^ d) + e + ctx->k[1] + m[i];
		e = d;
		d = c;
		c = ROTLEFT(b, 30);
		b = a;
		a = t;
	}
	for ( ; i < 60; ++i) {
		t = ROTLEFT(a, 5) + ((b & c) ^ (b & d) ^ (c & d))  + e + ctx->k[2] + m[i];
		e = d;
		d = c;
		c = ROTLEFT(b, 30);
		b = a;
		a = t;
	}
	for ( ; i < 80; ++i) {
		t = ROTLEFT(a, 5) + (b ^ c ^ d) + e + ctx->k[3] + m[i];
		e = d;
		d = c;
		c = ROTLEFT(b, 30);
		b = a;
		a = t;
	}

	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
}

__device__ void cuda_sha1_init(CUDA_SHA1_CTX *ctx)
{
	ctx->datalen = 0;
	ctx->bitlen = 0;
	ctx->state[0] = 0x67452301;
	ctx->state[1] = 0xEFCDAB89;
	ctx->state[2] = 0x98BADCFE;
	ctx->state[3] = 0x10325476;
	ctx->state[4] = 0xc3d2e1f0;
	ctx->k[0] = 0x5a827999;
	ctx->k[1] = 0x6ed9eba1;
	ctx->k[2] = 0x8f1bbcdc;
	ctx->k[3] = 0xca62c1d6;
}

__device__ void cuda_sha1_update(CUDA_SHA1_CTX *ctx, const BYTE data[], size_t len)
{
	size_t i;

	for (i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			cuda_sha1_transform(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}

__device__ void cuda_sha1_final(CUDA_SHA1_CTX *ctx, BYTE hash[])
{
	WORD i;

	i = ctx->datalen;

	// Pad whatever data is left in the buffer.
	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		cuda_sha1_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	// Append to the padding the total message's length in bits and transform.
	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
	cuda_sha1_transform(ctx, ctx->data);

	// Since this implementation uses little endian byte ordering and MD uses big endian,
	// reverse all the bytes when copying the final state to the output hash.
	for (i = 0; i < 4; ++i) {
		hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
	}
}

__global__ void kernel_sha1_hash(BYTE* indata, WORD inlen, BYTE* outdata, WORD n_batch)
{
	WORD thread = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread >= n_batch)
	{
		return;
	}
	BYTE* in = indata  + thread * inlen;
	BYTE* out = outdata  + thread * SHA1_BLOCK_SIZE;
	CUDA_SHA1_CTX ctx;
	cuda_sha1_init(&ctx);
	cuda_sha1_update(&ctx, in, inlen);
	cuda_sha1_final(&ctx, out);
}

void mcm_cuda_sha1_hash_batch(BYTE* in, WORD inlen, BYTE* out, WORD n_batch)
{
	BYTE *cuda_indata;
	BYTE *cuda_outdata;
	cudaMalloc(&cuda_indata, inlen * n_batch);
	cudaMalloc(&cuda_outdata, SHA1_BLOCK_SIZE * n_batch);
	cudaMemcpy(cuda_indata, in, inlen * n_batch, cudaMemcpyHostToDevice);

	WORD thread = 256;
	WORD block = (n_batch + thread - 1) / thread;
	kernel_sha1_hash << < block, thread >> > (cuda_indata, inlen, cuda_outdata, n_batch);
	cudaMemcpy(out, cuda_outdata, SHA1_BLOCK_SIZE * n_batch, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda sha1 hash:%d %s \n", error, cudaGetErrorString(error));
	}
	cudaFree(cuda_indata);
	cudaFree(cuda_outdata);
}

int get_hash(char *start, char *all, int depth)
{
    if(depth > 20) {
        return 0;
    }
    BYTE* cuda_sha1_start;
    WORD cuda_sha1_start_size;
    WORD cuda_sha1_res_size;
    BYTE* cuda_sha1_res;
    cuda_sha1_res_size = reinterpret_cast<WORD>((unsigned int)64);
    cuda_sha1_res = (BYTE*)malloc(64 * SHA1_BLOCK_SIZE * sizeof(BYTE));
    char buffer[40];
    char temp[40];
    char target[40];
    depth++;
    for(int i = 97; i < 123; i++) {
        memset(target, '\0', sizeof(target));
        memset(buffer, '\0', sizeof(buffer));
        memset(temp, '\0', sizeof(temp));
        sprintf(target, "%s%c", start, i);
        cuda_sha1_start = reinterpret_cast<BYTE*>(target);
        cuda_sha1_start_size = reinterpret_cast<WORD>((unsigned int)strlen(target));
        mcm_cuda_sha1_hash_batch(cuda_sha1_start, cuda_sha1_start_size, cuda_sha1_res, cuda_sha1_res_size);
        for(int z =0;z<20;z++) {
            sprintf(temp, "%02x", cuda_sha1_res[z]);
            strcat(buffer, temp);
        }
        printf("Comparing:%s %s %s %d\n", target, buffer, all, strcmp(buffer, all));
        if(strcmp(buffer, all) == 0 ){
            printf("Found target:%s\n", target);
            return 1;
        }
    }
    for(int i = 97; i < 123; i++) {
        sprintf(target, "%s%c", start, i);
        if(1 == get_hash(target, all, depth)) {
            return 1;
        }
    }
    return 0;
}

int main()
{
    char start[256], all[256];
    printf("Start:");
    fgets(start, sizeof(start), stdin);
    size_t ln = strlen(start)-1;
    if (start[ln] == '\n')
        start[ln] = '\0';
    printf("All:");
    fgets(all, sizeof(all), stdin);
    size_t ln_all = strlen(all)-1;
    if (all[ln_all] == '\n')
        all[ln_all] = '\0';
    get_hash(start, all, 0);
}