/*
 * This software is Copyright (c) 2024 magnum and it is hereby released to the
 * general public under the following terms:  Redistribution and use in source
 * and binary forms, with or without modification, are permitted.
 *
 * See CPU format for human-readable code ;-)
 */

#include "opencl_misc.h"
#include "opencl_mask.h"
#include "opencl_md5.h"
#include "md5x50.h"
#include "opencl_md5_ctx.h"
#include "opencl_sha2_ctx.h"
#include "opencl_rc4.h"
#include "opencl_aes.h"

typedef struct {
	uint V;             // populated but unused
	uint R;
	int P;
	uint encrypt_metadata;
	uchar u[128];
	uchar o[128];
	uchar id[128];
	uint key_length;    // key length in bits
	uint id_len;
	uint u_len;
	uint o_len;
} pdf_salt_type;

__constant uint padding[8] = {
	0x5e4ebf28, 0x418a754e, 0x564e0064, 0x0801faff,
	0xb6002e2e, 0x803e68d0, 0xfea90c2f, 0x7a695364
};

inline uint prepare234(__global const uchar *pwbuf, __global const uint *index, uint *password)
{
	uint i;
	uint gid = get_global_id(0);
	uint base = index[gid];
	uint len = index[gid + 1] - base;

	pwbuf += base;

	/* Work-around for self-tests not always calling set_key() like IRL */
	if (len > PLAINTEXT_LENGTH)
		len = 0;

	for (i = 0; i < len; i++)
		((uchar*)password)[i] = pwbuf[i];

	/* Pad for rev < 5, offloading the inner loop */
	if (len < 32)
		memcpy_cp((uchar*)password + len, (__constant uchar*)padding, 32 - len);

	return 32;
}

__kernel
#ifdef RC4_USE_LOCAL
__attribute__((work_group_size_hint(MAX_LOCAL_RC4, 1, 1)))
#endif
void pdf_r2(__global const uchar *pwbuf,
            __global const uint *index,
            __constant pdf_salt_type *pdf_salt,
            __global uint *result,
            volatile __global uint *crack_count_ret,
            __global uint *int_key_loc,
#if USE_CONST_CACHE
            __constant
#else
            __global
#endif
            uint *int_keys)
{
#ifdef RC4_USE_LOCAL
	__local
#endif
	RC4_CTX rc4_ctx;
	uint password[(PLAINTEXT_LENGTH + 3) / 4]; // Not null terminated
	uint gid = get_global_id(0);
#if NUM_INT_KEYS > 1 && !IS_STATIC_GPU_MASK
	uint ikl = int_key_loc[gid];
	uint loc0 = ikl & 0xff;
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
	uint loc1 = (ikl & 0xff00) >> 8;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
	uint loc2 = (ikl & 0xff0000) >> 16;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
	uint loc3 = (ikl & 0xff000000) >> 24;
#endif
#endif
#endif

#if !IS_STATIC_GPU_MASK
#define GPU_LOC_0 loc0
#define GPU_LOC_1 loc1
#define GPU_LOC_2 loc2
#define GPU_LOC_3 loc3
#else
#define GPU_LOC_0 LOC_0
#define GPU_LOC_1 LOC_1
#define GPU_LOC_2 LOC_2
#define GPU_LOC_3 LOC_3
#endif

	/* Prepare password */
	prepare234(pwbuf, index, password);

	for (uint mi = 0; mi < NUM_INT_KEYS; mi++) {

		/* Apply GPU-side mask */
#if NUM_INT_KEYS > 1
		password[GPU_LOC_0] = int_keys[mi] & 0xff;
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
		password[GPU_LOC_1] = (int_keys[mi] & 0xff00) >> 8;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
		password[GPU_LOC_2] = (int_keys[mi] & 0xff0000) >> 16;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
		password[GPU_LOC_3] = (int_keys[mi] & 0xff000000) >> 24;
#endif
#endif
#endif

		uint gidx = gid * NUM_INT_KEYS + mi;

		uint key[4];
		md5_init(key);

		/* Password already padded in prepare() and always len 32 */
		uint W[16];
		memcpy_macro(W, password, 32/4);
		memcpy_macro(W + 32/4, (__constant uint*)pdf_salt->o, 32/4);

		/* The above is always one M-D block */
		md5_block(uint, W, key);

		W[0] = (uint)pdf_salt->P;
		uint md5_len = 4;

		memcpy_macro(W + 1, (__constant uint*)pdf_salt->id, pdf_salt->id_len / 4);
		md5_len += pdf_salt->id_len;

		W[md5_len / 4] = 0x00000080;
		for (uint i = md5_len / 4 + 1; i < 14; i++)
			W[i] = 0;
		W[14] = (64 + md5_len) << 3;
		W[15] = 0;

		md5_block(uint, W, key);

		uint output[16/4];
		memcpy_macro(output, padding, 16/4);

		rc4_40_set_key(&rc4_ctx, key);
		rc4(&rc4_ctx, output, output, 16);

		if ((result[gidx] = !memcmp_pc(output, pdf_salt->u, 16)))
			atomic_max(crack_count_ret, gidx + 1);
	}
}

__kernel
#ifdef RC4_USE_LOCAL
__attribute__((work_group_size_hint(MAX_LOCAL_RC4, 1, 1)))
#endif
void pdf_r34(__global const uchar *pwbuf,
             __global const uint *index,
             __constant pdf_salt_type *pdf_salt,
             __global uint *result,
             volatile __global uint *crack_count_ret,
             __global uint *int_key_loc,
#if USE_CONST_CACHE
             __constant
#else
             __global
#endif
             uint *int_keys)
{
#ifdef RC4_USE_LOCAL
	__local
#endif
	RC4_CTX rc4_ctx;
	uint password[(PLAINTEXT_LENGTH + 3) / 4]; // Not null terminated
	uint gid = get_global_id(0);
#if NUM_INT_KEYS > 1 && !IS_STATIC_GPU_MASK
	uint ikl = int_key_loc[gid];
	uint loc0 = ikl & 0xff;
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
	uint loc1 = (ikl & 0xff00) >> 8;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
	uint loc2 = (ikl & 0xff0000) >> 16;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
	uint loc3 = (ikl & 0xff000000) >> 24;
#endif
#endif
#endif

#if !IS_STATIC_GPU_MASK
#define GPU_LOC_0 loc0
#define GPU_LOC_1 loc1
#define GPU_LOC_2 loc2
#define GPU_LOC_3 loc3
#else
#define GPU_LOC_0 LOC_0
#define GPU_LOC_1 LOC_1
#define GPU_LOC_2 LOC_2
#define GPU_LOC_3 LOC_3
#endif

	/* Prepare password */
	prepare234(pwbuf, index, password);

	for (uint mi = 0; mi < NUM_INT_KEYS; mi++) {

		/* Apply GPU-side mask */
#if NUM_INT_KEYS > 1
		password[GPU_LOC_0] = int_keys[mi] & 0xff;
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
		password[GPU_LOC_1] = (int_keys[mi] & 0xff00) >> 8;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
		password[GPU_LOC_2] = (int_keys[mi] & 0xff0000) >> 16;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
		password[GPU_LOC_3] = (int_keys[mi] & 0xff000000) >> 24;
#endif
#endif
#endif

		uint gidx = gid * NUM_INT_KEYS + mi;
		uint output[32 / 4];
		uint key[MAX_KEY_SIZE / 8 / 4];
		uint digest[16 / 4];
		uint W[(32 + sizeof(pdf_salt->id) + 63) / 64 * 16]; // three limbs

		md5_init(key);

		/* Password already padded in prepare() */
		memcpy_macro(W, password, 32/4);
		memcpy_macro(W + 32/4, (__constant uint*)pdf_salt->o, 32/4);

		/* The above is always one M-D block */
		md5_block(uint, W, key);

		W[0] = (uint)pdf_salt->P;
		uint md5_len = 4;

		memcpy_macro(W + 1, (__constant uint*)pdf_salt->id, pdf_salt->id_len / 4);
		md5_len += pdf_salt->id_len;

		if (pdf_salt->R >= 4 && !pdf_salt->encrypt_metadata) {
			W[md5_len / 4] = 0xffffffff;
			md5_len += 4;
		}

		W[md5_len / 4] = 0x00000080;
		for (uint i = md5_len / 4 + 1; i < 14; i++)
			W[i] = 0;
		W[14] = (64 + md5_len) << 3;
		W[15] = 0;

		md5_block(uint, W, key);

		if (pdf_salt->key_length == 40)
			md5x50_40(key);
		else
			md5x50_128(key);

		uint *WP = W;
		uint md5len = 32 + pdf_salt->id_len;

		memcpy_macro(W, padding, 8);
		memcpy_macro(W + 8, (__constant uint*)pdf_salt->id, pdf_salt->id_len / 4);
		LASTCHAR(W, md5len, 0x80);

		/* Clean the last M-D block */
		uint start_clean = md5len / 4 + 1;
		uint md_len_pos = (start_clean & 0xf0) + 14 + (((md5len & 63) > 55) ? 16 : 0);
		for (uint i = start_clean; i < md_len_pos; i++)
			W[i] = 0;
		W[md_len_pos] = md5len << 3;
		W[md_len_pos + 1] = 0;

		md5_init(digest);

		WP = W;
		for (uint left = md5len; left > 55; left -= 64, WP +=16)
			md5_block(uint, WP, digest);
		md5_block(uint, WP, digest);

		if (pdf_salt->key_length == 40) {
			rc4_40_set_key(&rc4_ctx, key);
			rc4(&rc4_ctx, digest, output, 16);
			for (uint x = 0x01010101; x <= 0x13131313; x += 0x01010101) {
				uint xor[2];

				xor[0] = key[0] ^ x;
				xor[1] = key[1] ^ x;

				rc4_40_set_key(&rc4_ctx, xor);
				rc4(&rc4_ctx, output, output, 16);
			}
		} else {
			rc4_128_set_key(&rc4_ctx, key);
			rc4(&rc4_ctx, digest, output, 16);
			for (uint x = 0x01010101; x <= 0x13131313; x += 0x01010101) {
				uint xor[16 / 4];

				xor[0] = key[0] ^ x;
				xor[1] = key[1] ^ x;
				xor[2] = key[2] ^ x;
				xor[3] = key[3] ^ x;

				rc4_128_set_key(&rc4_ctx, xor);
				rc4(&rc4_ctx, output, output, 16);
			}
		}

		if ((result[gidx] = !memcmp_pc(output, pdf_salt->u, 16)))
			atomic_max(crack_count_ret, gidx + 1);
	}
}

inline uint prepare56(__global const uchar *pwbuf, __global const uint *index, uint *password)
{
	uint i;
	uint gid = get_global_id(0);
	uint base = index[gid];
	uint len = index[gid + 1] - base;

	pwbuf += base;

	/* Work-around for self-tests not always calling set_key() like IRL */
	if (len > PLAINTEXT_LENGTH)
		len = 0;

	for (i = 0; i < len; i++)
		((uchar*)password)[i] = pwbuf[i];

	return len;
}

__kernel
void pdf_r5(__global const uchar *pwbuf,
            __global const uint *index,
            __constant pdf_salt_type *pdf_salt,
            __global uint *result,
            volatile __global uint *crack_count_ret,
            __global uint *int_key_loc,
#if USE_CONST_CACHE
            __constant
#else
            __global
#endif
            uint *int_keys)
{
	uint password[(PLAINTEXT_LENGTH + 3) / 4]; // Not null terminated
	uint gid = get_global_id(0);
#if NUM_INT_KEYS > 1 && !IS_STATIC_GPU_MASK
	uint ikl = int_key_loc[gid];
	uint loc0 = ikl & 0xff;
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
	uint loc1 = (ikl & 0xff00) >> 8;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
	uint loc2 = (ikl & 0xff0000) >> 16;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
	uint loc3 = (ikl & 0xff000000) >> 24;
#endif
#endif
#endif

#if !IS_STATIC_GPU_MASK
#define GPU_LOC_0 loc0
#define GPU_LOC_1 loc1
#define GPU_LOC_2 loc2
#define GPU_LOC_3 loc3
#else
#define GPU_LOC_0 LOC_0
#define GPU_LOC_1 LOC_1
#define GPU_LOC_2 LOC_2
#define GPU_LOC_3 LOC_3
#endif

	/* Prepare password */
	uint pw_len = prepare56(pwbuf, index, password);

	for (uint mi = 0; mi < NUM_INT_KEYS; mi++) {

		/* Apply GPU-side mask */
#if NUM_INT_KEYS > 1
		password[GPU_LOC_0] = int_keys[mi] & 0xff;
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
		password[GPU_LOC_1] = (int_keys[mi] & 0xff00) >> 8;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
		password[GPU_LOC_2] = (int_keys[mi] & 0xff0000) >> 16;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
		password[GPU_LOC_3] = (int_keys[mi] & 0xff000000) >> 24;
#endif
#endif
#endif

		uint gidx = gid * NUM_INT_KEYS + mi;
		uint validationkey[32 / 4];
		uint W[(128 + 8 + 48 + 63) / 64 * 16]; // up to three limbs
		uint *WP = W;
		uint sha256[8];
		const uint sha256len = pw_len + 8;

		uint i = 0;
		do {
			*WP++ = SWAP32(password[i]);
		} while (4 * ++i < pw_len);

		__constant uchar *c = (__constant uchar*)pdf_salt->u + 32;
		for (i = pw_len; i < pw_len + 8; i++)
			PUTCHAR_BE(W, i, *c++);

		LASTCHAR_BE(W, sha256len, 0x80);

		/* Clean the last M-D block */
		uint start_clean = sha256len / 4 + 1;
		uint md_len_pos = (start_clean & 0xf0) + 15 + (((sha256len & 63) > 55) ? 16 : 0);
		for (uint i = start_clean; i < md_len_pos; i++)
			W[i] = 0;
		W[md_len_pos] = sha256len << 3;

		sha256_init(sha256);

		WP = W;
		for (uint left = sha256len; left > 55; left -= 64, WP +=16)
			sha256_block(WP, sha256);
		sha256_block(WP, sha256);

		block_swap32(sha256, 8);
		memcpy_macro(validationkey, sha256, 8);

		if ((result[gidx] = !memcmp_pc(validationkey, pdf_salt->u, 16)))
			atomic_max(crack_count_ret, gidx + 1);
	}
}

__kernel
void pdf_r6(__global const uchar *pwbuf,
            __global const uint *index,
            __constant pdf_salt_type *pdf_salt,
            __global uint *result,
            volatile __global uint *crack_count_ret,
            __global uint *int_key_loc,
#if USE_CONST_CACHE
            __constant
#else
            __global
#endif
            uint *int_keys)
{
	uint password[(PLAINTEXT_LENGTH + 3) / 4]; // Not null terminated
	uint gid = get_global_id(0);
#if NUM_INT_KEYS > 1 && !IS_STATIC_GPU_MASK
	uint ikl = int_key_loc[gid];
	uint loc0 = ikl & 0xff;
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
	uint loc1 = (ikl & 0xff00) >> 8;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
	uint loc2 = (ikl & 0xff0000) >> 16;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
	uint loc3 = (ikl & 0xff000000) >> 24;
#endif
#endif
#endif

#if !IS_STATIC_GPU_MASK
#define GPU_LOC_0 loc0
#define GPU_LOC_1 loc1
#define GPU_LOC_2 loc2
#define GPU_LOC_3 loc3
#else
#define GPU_LOC_0 LOC_0
#define GPU_LOC_1 LOC_1
#define GPU_LOC_2 LOC_2
#define GPU_LOC_3 LOC_3
#endif

	/* Prepare password */
	uint pw_len = prepare56(pwbuf, index, password);

	for (uint mi = 0; mi < NUM_INT_KEYS; mi++) {

		/* Apply GPU-side mask */
#if NUM_INT_KEYS > 1
		password[GPU_LOC_0] = int_keys[mi] & 0xff;
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
		password[GPU_LOC_1] = (int_keys[mi] & 0xff00) >> 8;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
		password[GPU_LOC_2] = (int_keys[mi] & 0xff0000) >> 16;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
		password[GPU_LOC_3] = (int_keys[mi] & 0xff000000) >> 24;
#endif
#endif
#endif

		uint gidx = gid * NUM_INT_KEYS + mi;
		uint hash[32 / 4];
		ulong data64[(128 + 64 + 48) * 64 / sizeof(ulong)]; // 120/240 limbs
		//uint *data32 = (uint*)data64;
		uchar *data = (uchar*)data64;
		uint block[64 / 4];
		uint block_size = 32;
		uint data_len = 0;
		uint i, j, sum;
		SHA256_CTX sha256;
		SHA512_CTX sha384;
		SHA512_CTX sha512;
		AES_KEY aes;

		SHA256_Init(&sha256);
		SHA256_Update(&sha256, (uchar*)password, pw_len);
		uchar s[8];
		memcpy_cp(s, pdf_salt->u + 32, 8);
		SHA256_Update(&sha256, s, 8);
		SHA256_Final((uchar*)block, &sha256);

		for (i = 0; i < 64 || i < data[data_len * 64 - 1] + 32; i++) {
			memcpy_pp(data, password, pw_len);
			memcpy_pp(data + pw_len, block, block_size);
			data_len = pw_len + block_size;
			for (j = 1; j < 64; j++)
				memcpy_pp(data + j * data_len, data, data_len);

			AES_set_encrypt_key(block, 128, &aes);
			AES_cbc_encrypt(data, data, data_len * 64, &aes, block + (16 / 4));

			for (j = 0, sum = 0; j < 16; j++)
				sum += data[j];

			block_size = 32 + (sum % 3) * 16;
			switch (block_size)
			{
			case 32:
				SHA256_Init(&sha256);
				SHA256_Update(&sha256, data, data_len * 64);
				SHA256_Final((uchar*)block, &sha256);
				break;
			case 48:
				SHA384_Init(&sha384);
				SHA384_Update(&sha384, data, data_len * 64);
				SHA384_Final((uchar*)block, &sha384);
				break;
			case 64:
				SHA512_Init(&sha512);
				SHA512_Update(&sha512, data, data_len * 64);
				SHA512_Final((uchar*)block, &sha512);
				break;
			}
		}

		memcpy_macro(hash, block, 32 / 4);

		if ((result[gidx] = !memcmp_pc(hash, pdf_salt->u, 16)))
			atomic_max(crack_count_ret, gidx + 1);
	}
}
