#pragma once

#include <cstdint>

#define RESTRICT __restrict

#if defined(__aarch64__)

#include <arm_neon.h>

// AArch64 has Advanced SIMD (NEON) as part of the base architecture.
constexpr int SIMD_REGISTERS = 8;
constexpr int SIMD_ALIGN = 16;
constexpr int simd_reg_width = 128;

// Use an untyped 128-bit container so the layer code can share one vector
// storage type for both 16-bit accumulators and 32-bit dot-product sums.
using SIMDVector = uint8x16_t;

inline SIMDVector vec_add_epi32(SIMDVector a, SIMDVector b) {
    return vreinterpretq_u8_s32(vaddq_s32(vreinterpretq_s32_u8(a), vreinterpretq_s32_u8(b)));
}

inline SIMDVector vec_add_epi16(SIMDVector a, SIMDVector b) {
    return vreinterpretq_u8_s16(vaddq_s16(vreinterpretq_s16_u8(a), vreinterpretq_s16_u8(b)));
}

inline SIMDVector vec_sub_epi16(SIMDVector a, SIMDVector b) {
    return vreinterpretq_u8_s16(vsubq_s16(vreinterpretq_s16_u8(a), vreinterpretq_s16_u8(b)));
}

inline SIMDVector vec_madd_epi16(SIMDVector a, SIMDVector b) {
    const int16x8_t va = vreinterpretq_s16_u8(a);
    const int16x8_t vb = vreinterpretq_s16_u8(b);
    const int32x4_t lo = vmull_s16(vget_low_s16(va), vget_low_s16(vb));
    const int32x4_t hi = vmull_s16(vget_high_s16(va), vget_high_s16(vb));
    return vreinterpretq_u8_s32(vpaddq_s32(lo, hi));
}

inline SIMDVector vec_hadd_epi32(SIMDVector a, SIMDVector b) {
    return vreinterpretq_u8_s32(vpaddq_s32(vreinterpretq_s32_u8(a), vreinterpretq_s32_u8(b)));
}

inline SIMDVector vec_set1_epi16(int16_t value) {
    return vreinterpretq_u8_s16(vdupq_n_s16(value));
}

inline SIMDVector vec_min_epi16(SIMDVector a, SIMDVector b) {
    return vreinterpretq_u8_s16(vminq_s16(vreinterpretq_s16_u8(a), vreinterpretq_s16_u8(b)));
}

inline SIMDVector vec_max_epi16(SIMDVector a, SIMDVector b) {
    return vreinterpretq_u8_s16(vmaxq_s16(vreinterpretq_s16_u8(a), vreinterpretq_s16_u8(b)));
}

inline int32_t vec_hsum_epi32(SIMDVector x) {
    return vaddvq_s32(vreinterpretq_s32_u8(x));
}

#elif defined(__AVX2__)

#include <immintrin.h>
#define USE_AVX2
#else 
#define USE_SSSE3
#endif

#if defined(USE_AVX2)

constexpr int SIMD_REGISTERS = 16;
constexpr int SIMD_ALIGN = 32;
constexpr int simd_reg_width = 256;

using SIMDVector = __m256i;

#define vec_add_epi32 _mm256_add_epi32

#define vec_add_epi16 _mm256_add_epi16
#define vec_sub_epi16 _mm256_sub_epi16
#define vec_madd_epi16 _mm256_madd_epi16
#define vec_hadd_epi32 _mm256_hadd_epi32

#define vec_set1_epi16 _mm256_set1_epi16

#define vec_min_epi16 _mm256_min_epi16
#define vec_max_epi16 _mm256_max_epi16

inline int32_t vec_hsum_epi32(__m256i x) {
    __m128i lo128 = _mm256_castsi256_si128(x);
    __m128i hi128 = _mm256_extracti128_si256(x, 1);

    __m128i sum128 = _mm_add_epi32(lo128, hi128);

    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    __m128i sum64 = _mm_add_epi32(hi64, sum128);

    __m128i hi32 = _mm_shuffle_epi32(sum64, 0b00'00'00'01);

    return _mm_cvtsi128_si32(_mm_add_epi32(hi32, sum64));
}


#elif defined(USE_SSSE3)

#include <immintrin.h>

constexpr int SIMD_REGISTERS = 8;
constexpr int SIMD_ALIGN = 16;
constexpr int simd_reg_width = 128;

using SIMDVector = __m128i;

#define vec_add_epi32 _mm_add_epi32

#define vec_add_epi16 _mm_add_epi16
#define vec_sub_epi16 _mm_sub_epi16

#define vec_madd_epi16 _mm_madd_epi16
#define vec_hadd_epi32 _mm_hadd_epi32

#define vec_set1_epi16 _mm_set1_epi16

#define vec_min_epi16 _mm_min_epi16
#define vec_max_epi16 _mm_max_epi16

inline int32_t vec_hsum_epi32(__m128i sum128) {
    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    __m128i sum64 = _mm_add_epi32(hi64, sum128);

    __m128i hi32 = _mm_shuffle_epi32(sum64, 0b00'00'00'01);

    return _mm_cvtsi128_si32(_mm_add_epi32(hi32, sum64));
}

#endif
