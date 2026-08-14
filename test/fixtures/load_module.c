/* test/fixtures/load_module.c
 * Generic HSACO module loader/launcher harness (atomgpu plan todo 6, stage 3).
 *
 * Usage: load_module <hsaco-path> <kernel-symbol> [N]
 *   N = total element count, default 1<<20; must be a positive multiple of
 *       the fixed block size 256.
 *
 * Kernel ABI contract (the todo-7 xchg kernel must match it):
 *   __global__ void <symbol>(const float* in, float* out)
 * The harness fills in[i] = (float)(i % 256), launches N/256 blocks of 256
 * threads with sharedMemBytes=0 (plan G9: static LDS only, no dynamic LDS),
 * then verifies the cross-thread-exchange pattern
 *   out[i] == (float)((i + 1) % 256)
 * which matches each thread reading shared slot (tid + 1) % block_dim.
 *
 * Every HIP API return code is checked; any failure prints a clear error to
 * stderr and exits non-zero. Missing/invalid input files exit cleanly (no
 * crash). Prints LOAD-MODULE-PASS only if verification fully passes.
 */

#include <hip/hip_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIP_CHECK(expr)                                                      \
  do {                                                                       \
    hipError_t hip_err_ = (expr);                                            \
    if (hip_err_ != hipSuccess) {                                            \
      fprintf(stderr, "HIP error %s:%d: %s: %s (%d)\n", __FILE__, __LINE__,  \
              #expr, hipGetErrorString(hip_err_), (int)hip_err_);            \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

/* Read the whole file into a malloc'd buffer. Returns NULL on any error
 * (with a message on stderr); *out_size receives the byte count. */
static void* read_file(const char* path, size_t* out_size) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "ERROR: cannot open HSACO file '%s'\n", path);
    return NULL;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fprintf(stderr, "ERROR: fseek failed on '%s'\n", path);
    fclose(f);
    return NULL;
  }
  long sz = ftell(f);
  if (sz < 0) {
    fprintf(stderr, "ERROR: ftell failed on '%s'\n", path);
    fclose(f);
    return NULL;
  }
  rewind(f);
  void* buf = malloc(sz > 0 ? (size_t)sz : 1u);
  if (!buf) {
    fprintf(stderr, "ERROR: host malloc of %ld bytes failed\n", sz);
    fclose(f);
    return NULL;
  }
  if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    fprintf(stderr, "ERROR: short read on '%s'\n", path);
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *out_size = (size_t)sz;
  return buf;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <hsaco-path> <kernel-symbol> [N]\n", argv[0]);
    return 2;
  }
  const char* path = argv[1];
  const char* symbol = argv[2];
  long N = 1L << 20;
  if (argc >= 4) {
    char* end = NULL;
    N = strtol(argv[3], &end, 10);
    if (!end || *end != '\0') {
      fprintf(stderr, "ERROR: N '%s' is not an integer\n", argv[3]);
      return 2;
    }
  }
  if (N <= 0 || (N % 256) != 0) {
    fprintf(stderr, "ERROR: N must be a positive multiple of 256 (got %ld)\n",
            N);
    return 2;
  }

  size_t image_size = 0;
  void* image = read_file(path, &image_size);
  if (!image) {
    return 1;
  }

  HIP_CHECK(hipInit(0));

  hipModule_t module = NULL;
  HIP_CHECK(hipModuleLoadData(&module, image));

  hipFunction_t fn = NULL;
  HIP_CHECK(hipModuleGetFunction(&fn, module, symbol));

  const size_t bytes = sizeof(float) * (size_t)N;
  float* h_in = (float*)malloc(bytes);
  float* h_out = (float*)malloc(bytes);
  if (!h_in || !h_out) {
    fprintf(stderr, "ERROR: host malloc of %zu bytes failed\n", bytes);
    return 1;
  }
  for (long i = 0; i < N; ++i) {
    h_in[i] = (float)(i % 256);
  }

  float* d_in = NULL;
  float* d_out = NULL;
  HIP_CHECK(hipMalloc((void**)&d_in, bytes));
  HIP_CHECK(hipMalloc((void**)&d_out, bytes));
  HIP_CHECK(hipMemcpy(d_in, h_in, bytes, hipMemcpyHostToDevice));
  /* Zero the output so unwritten/corrupted slots are detectable. */
  HIP_CHECK(hipMemset(d_out, 0, bytes));

  void* args[] = {&d_in, &d_out};
  const unsigned int grid = (unsigned int)(N / 256);
  HIP_CHECK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1,
                                  0,   /* sharedMemBytes=0 (plan G9) */
                                  0,   /* default stream */
                                  args, NULL));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_out, d_out, bytes, hipMemcpyDeviceToHost));

  long bad = 0;
  for (long i = 0; i < N; ++i) {
    const float want = (float)((i + 1) % 256);
    if (h_out[i] != want) {
      if (bad < 8) {
        fprintf(stderr, "MISMATCH i=%ld got=%f want=%f\n", i,
                (double)h_out[i], (double)want);
      }
      ++bad;
    }
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipModuleUnload(module));
  free(image);
  free(h_in);
  free(h_out);

  if (bad != 0) {
    fprintf(stderr, "LOAD-MODULE-FAIL: %ld/%ld elements incorrect\n", bad, N);
    return 1;
  }
  printf("LOAD-MODULE-PASS\n");
  return 0;
}
