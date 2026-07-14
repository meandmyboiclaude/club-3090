/*
 * greenboost-cuda12-bridge.c — CUDA 12.x versioned-symbol bridge for GreenBoost
 *
 * GreenBoost's LD_PRELOAD shim exports CUDA runtime hooks with @@libcudart.so.13
 * symbol versioning, but PyTorch + vLLM ship CUDA 12.x (libcudart.so.12).
 * In CUDA 12.8+ the runtime also resolves driver functions via cuGetProcAddress
 * instead of the PLT, so GreenBoost's driver API hooks are bypassed.
 *
 * This bridge provides cudaMemGetInfo, cudaMalloc, cudaMallocAsync, and cudaFree
 * with @@libcudart.so.12 version tags, forwarding to GreenBoost's driver-level
 * hooks that perform the actual VRAM oversubscription.
 *
 * Build:
 *   gcc -shared -fPIC \
 *       -Wl,--version-script=greenboost-cuda12-bridge.map \
 *       -o libgreenboost_cuda12_bridge.so \
 *       greenboost-cuda12-bridge.c -ldl
 *
 * LD_PRELOAD order: bridge FIRST, then greenboost shim, e.g.
 *   LD_PRELOAD="libgreenboost_cuda12_bridge.so libgreenboost_cuda.so"
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>

typedef int CUresult;
typedef int cudaError_t;
typedef unsigned long long CUdeviceptr;
typedef void *CUstream;

/* GreenBoost driver API hooks — resolved at load time via LD_PRELOAD */
extern CUresult cuMemGetInfo_v2(size_t *free, size_t *total);
extern CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
extern CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream stream);
extern CUresult cuMemFree_v2(CUdeviceptr dptr);

static cudaError_t cu2cuda(CUresult r) {
    return r == 0 ? 0 : 2; /* cudaErrorMemoryAllocation */
}

typedef cudaError_t (*real_cudaFree_t)(void *);

static cudaError_t ensure_context(void) {
    static real_cudaFree_t real_free = NULL;
    if (!real_free)
        real_free = (real_cudaFree_t)dlvsym(
            RTLD_NEXT, "cudaFree", "libcudart.so.12");
    return real_free ? real_free(0) : 0;
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
    typedef cudaError_t (*fn_t)(size_t *, size_t *);
    static fn_t real = NULL;
    if (!real)
        real = (fn_t)dlvsym(
            RTLD_NEXT, "cudaMemGetInfo", "libcudart.so.12");
    cudaError_t ret = real ? real(free, total) : ensure_context();
    if (ret != 0)
        return ret;
    cuMemGetInfo_v2(free, total);
    return 0;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    ensure_context();
    return cu2cuda(cuMemAlloc_v2((CUdeviceptr *)devPtr, size));
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, void *stream) {
    ensure_context();
    return cu2cuda(cuMemAllocAsync((CUdeviceptr *)devPtr, size,
                                   (CUstream)stream));
}

cudaError_t cudaFree(void *devPtr) {
    if (devPtr == NULL) {
        static real_cudaFree_t real = NULL;
        if (!real)
            real = (real_cudaFree_t)dlvsym(
                RTLD_NEXT, "cudaFree", "libcudart.so.12");
        return real ? real(devPtr) : 0;
    }
    return cu2cuda(cuMemFree_v2((CUdeviceptr)(uintptr_t)devPtr));
}
