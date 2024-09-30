/*
 * OpenCL RC4
 *
 * Copyright (c) 2014-2024, magnum
 * This software is hereby released to the general public under
 * the following terms: Redistribution and use in source and binary
 * forms, with or without modification, are permitted.
 *
 * NOTICE: After changes, you probably need to drop cached kernels to
 * ensure the changes take effect: "make kernel-cache-clean"
 *
 * NOTE: These functions assume 32-bit aligment - no assertions!
 */

#ifndef _OPENCL_RC4_H
#define _OPENCL_RC4_H

#ifndef RC4_KEY_TYPE
#define RC4_KEY_TYPE
#endif

#ifndef RC4_IN_TYPE
#define RC4_IN_TYPE
#endif

#ifndef RC4_OUT_TYPE
#define RC4_OUT_TYPE
#endif

#include "opencl_misc.h"

#if __GPU__
#define RC4_USE_LOCAL
#endif

#define GETCHAR_KEY(buf, index)	(((RC4_KEY_TYPE uchar*)(buf))[(index)])
#define GETCHAR_IN(buf, index)	(((RC4_IN_TYPE uchar*)(buf))[(index)])

typedef struct {
	uint state[32 * 256/4];
	uint x[32];
	uint y[32];
	uint len[32];
} RC4_CTX;

#define X8      (x & 255)
#define state   ctx->state
#define lid     get_local_id(0)

/* Local memory access pattern nicked from hashcat, 20% boost - Thank you! */
#define KEY8(t, k)  (((k) & 3) + (((k) / 4) * 128) + (((t) & 31) * 4) /* + (((t) / 32) * 8192) */)
#define KEY32(t, k) (((k) * 32) + ((t) & 31) /* + (((t) / 32) * 2048) */)

#ifdef RC4_USE_LOCAL
#define GETSTATE(state, n)      GETCHAR_L(state, KEY8(lid, n))
#define PUTSTATE(state, n, v)   PUTCHAR_L(state, KEY8(lid, n), v)
#else
#define GETSTATE(state, n)      GETCHAR(state, KEY8(lid, n))
#define PUTSTATE(state, n, v)   PUTCHAR(state, KEY8(lid, n), v)
#endif

#undef swap_byte
#define swap_byte(a, b) {	  \
		uint tmp = GETSTATE(state, a); \
		PUTSTATE(state, a, GETSTATE(state, b)); \
		PUTSTATE(state, b, tmp); \
	}
#undef swap_no_inc
#define swap_no_inc(n) {	  \
		index2 = (GETCHAR_KEY(key, index1) + GETSTATE(state, n) + index2) & 255; \
		swap_byte(n, index2); \
	}
#undef swap_state
#define swap_state(n) {	  \
		swap_no_inc(n); \
		if (++index1 == keylen) index1 = 0; \
	}
#undef swap_anc_inc
#define swap_and_inc(n) {	  \
		swap_no_inc(n); \
		index1++; n++; \
	}

/*
 * Set fixed IV
 */
inline void rc4_init(
#ifdef RC4_USE_LOCAL
                __local
#endif
                RC4_CTX *restrict ctx)
{
	uint v = 0x03020100;
	uint a = 0x04040404;

	for (uint i = 0; i < 64; i++) {
		state[KEY32(lid, i)] = v;
		v += a;
	}
}

/*
 * Arbitrary length key
 */
inline void rc4_set_key(
#ifdef RC4_USE_LOCAL
                __local
#endif
                RC4_CTX *restrict ctx,
                uint keylen,
                const uint *restrict key)
{
	rc4_init(ctx);

	uint index1 = 0;
	uint index2 = 0;

	for (uint x = 0; x < 256; x++)
		swap_state(x);

	ctx->x[lid] = 1;
	ctx->y[lid] = 0;
	ctx->len[lid] = 0;
}

/*
 * Unrolled fixed keylen of 5 (40-bit).
 */
inline void rc4_40_set_key(
#ifdef RC4_USE_LOCAL
                __local
#endif
                RC4_CTX *restrict ctx,
                const uint *restrict key)
{
	rc4_init(ctx);

	uint index1 = 0;
	uint index2 = 0;

	for (uint x = 0; x < 255; x++) {
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_no_inc(x);
		index1 = 0;
	}
	swap_no_inc(255);

	ctx->x[lid] = 1;
	ctx->y[lid] = 0;
	ctx->len[lid] = 0;
}

/*
 * Unrolled fixed keylen of 16 (128-bit).
 */
inline void rc4_128_set_key(
#ifdef RC4_USE_LOCAL
                __local
#endif
                RC4_CTX *restrict ctx,
                const uint *restrict key)
{
	rc4_init(ctx);

	uint index1 = 0;
	uint index2 = 0;

	for (uint x = 0; x < 256; x++) {
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_and_inc(x);
		swap_no_inc(x);
		index1 = 0;
	}

	ctx->x[lid] = 1;
	ctx->y[lid] = 0;
	ctx->len[lid] = 0;
}

/*
 * Len is given in bytes but must be multiple of 4.
 */
inline void rc4(
#ifdef RC4_USE_LOCAL
                __local
#endif
	                RC4_CTX *restrict ctx,
                RC4_IN_TYPE const uint *in,
                RC4_OUT_TYPE uint *out,
                uint len)
{
	uint x = ctx->x[lid];
	uint y = ctx->y[lid];

	len += ctx->len[lid];

	/* Unrolled to 32-bit xor */
	for (; x <= len; x++) {
		uint xor_word;

		y = (GETSTATE(state, X8) + y) & 255;
		swap_byte(X8, y);
		xor_word = GETSTATE(state, (GETSTATE(state, X8) + GETSTATE(state, y)) & 255);
		x++;

		y = (GETSTATE(state, X8) + y) & 255;
		swap_byte(X8, y);
		xor_word += GETSTATE(state, (GETSTATE(state, X8) + GETSTATE(state, y)) & 255) << 8;
		x++;

		y = (GETSTATE(state, X8) + y) & 255;
		swap_byte(X8, y);
		xor_word += GETSTATE(state, (GETSTATE(state, X8) + GETSTATE(state, y)) & 255) << 16;
		x++;

		y = (GETSTATE(state, X8) + y) & 255;
		swap_byte(X8, y);
		xor_word += GETSTATE(state, (GETSTATE(state, X8) + GETSTATE(state, y)) & 255) << 24;

		*out++ = *in++ ^ xor_word;
	}

	ctx->x[lid] = x;
	ctx->y[lid] = y;
	ctx->len[lid] = len;
}

#undef state
#undef X8

#endif /* _OPENCL_RC4_H */
