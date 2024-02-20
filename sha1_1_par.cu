#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "sha1.cuh"

#include <stdlib.h>
#include <memory.h>

/****************************** MACROS ******************************/
#define SHA1_BLOCK_SIZE 20              // SHA1 outputs a 20 byte digest
#define MAX_INPUT_LENGTH 128

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

void generate_words(char** generated_strings, int target_depth, int current_depth, char* current_string, int current_string_position) {
    if(target_depth == current_depth) {
        int correct_position = current_string_position + (int) current_string[strlen(current_string) - 1] - 97;
        printf("%d %s\n", correct_position , current_string);
//        generated_strings[correct_position] = current_string;
        strcpy(generated_strings[correct_position], current_string);
        return;
    }

    char temp[40];
    current_depth++;
    for(int current_char = 97; current_char < 123; current_char++ ) {
        sprintf(temp, "%s%c", current_string, current_char);
        generate_words(
                generated_strings,
                target_depth,
                current_depth,
                temp,
                current_string_position
                + (current_char - 97) * 26 * (target_depth - current_depth)
        );
    }
}

int main() {

    char start[256], all[256];

    BYTE* input_data;
    BYTE* output_hashes;

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

    char **original_strings;

    int character_number = 26;
    int output_size2;
    // dont touch it or the computer dies
    int max_calculation_length = 6;
    char temp[40];
    char buffer[40];
    unsigned int number_of_words;

    for(int i=strlen(start) + 1;i < max_calculation_length; i++) {
        printf("Length:%d\n", i);
        number_of_words = pow(character_number, (i - strlen(start)));
        original_strings  = (char**) malloc(number_of_words * sizeof(char*));
        input_data = (BYTE*) malloc(number_of_words * i * sizeof(BYTE));
        output_hashes = (BYTE*) malloc(number_of_words * SHA1_BLOCK_SIZE * sizeof(BYTE));
        for (int j = 0; j < number_of_words; j++) {
            original_strings[j] = (char*) malloc(j * sizeof(char));
        }
        generate_words(original_strings, i, 1, start, 0);
        for (int current_string = 0 ; current_string < number_of_words; current_string++) {
            strcat((char* )input_data, original_strings[current_string]);
            printf("%s %d\n", original_strings[current_string], i);
        }

        mcm_cuda_sha1_hash_batch(input_data, i, output_hashes, number_of_words);

        // Print the computed hashes
        for (int j = 0; j < number_of_words; j++) {
//            printf("Hash of input data %s: ", original_strings[j]);
            memset(buffer, '\0', sizeof(buffer));
            for (int k = 0; k < SHA1_BLOCK_SIZE; ++k) {
                memset(temp, '\0', sizeof(temp));
                sprintf(temp, "%02x", output_hashes[j * SHA1_BLOCK_SIZE + k]);
//                printf("%s\t", temp);
//                printf("%d\t", j * SHA1_BLOCK_SIZE + k);
                strcat(buffer, temp);
            }
            printf("\tComparing:%s %s %s %d\n", original_strings[j], buffer, all, strcmp(buffer, all));
            if(strcmp(buffer, all) == 0 ){
                printf("Found target:%s\n", original_strings[j]);
                return 1;
            }
//            printf("\n");
        }
    }
//    return 1;
//    // Input data
//
//
//    for (int i = 0; i < batch_size; ++i) {
//        strncpy((char*)(input_data + i * MAX_INPUT_LENGTH), original_strings[i], worldlength);
//    }
//    // Allocate memory for output hashes
//    BYTE output_hashes[output_size];
//
//    // Call the function to compute SHA-1 hashes in batches
//    mcm_cuda_sha1_hash_batch(input_data, worldlength, output_hashes, batch_size);
//
//    // Print the computed hashes
//    for (int i = 0; i < batch_size; ++i) {
//        printf("Hash of input data %s: ", original_strings[i]);
//        for (int j = 0; j < SHA1_BLOCK_SIZE; ++j) {
//            printf("%02x", output_hashes[i * SHA1_BLOCK_SIZE + j]);
//        }
//        printf("\n");
//    }

    return 0;
}