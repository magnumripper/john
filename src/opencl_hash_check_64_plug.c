/*
 * This software is
 * Copyright (c) 2015 Sayantan Datta <std2048 at gmail dot com>
 * Copyright (c) 2023 magnum
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 */

#if HAVE_OPENCL

#include <math.h> // For calculating bitmap characteristics

#include "options.h"
#include "logger.h"
#include "opencl_hash_check.h"
#include "mask_ext.h"
#include "misc.h"

cl_uint ocl_hc_num_loaded_hashes;
cl_uint *ocl_hc_hash_ids = NULL;
unsigned int ocl_hc_hash_table_size = 0, ocl_hc_offset_table_size = 0;

static cl_uint *loaded_hashes = NULL;
static OFFSET_TABLE_WORD *offset_table = NULL;
static uint64_t bitmap_size_bits;
static cl_uint *bitmaps = NULL;
static cl_uint *zero_buffer = NULL;
static cl_mem buffer_offset_table, buffer_hash_table, buffer_hash_ids_64, buffer_bitmap_dupe, buffer_bitmaps;
static struct fmt_main *self;

void ocl_hc_64_init(struct fmt_main *_self)
{
	self = _self;
	bt_hash_table_64 = NULL;
}

void ocl_hc_64_prepare_table(struct db_salt *salt)
{
	unsigned int *bin, i;
	struct db_password *pw, *last;

	ocl_hc_num_loaded_hashes = (salt->count);

	if (loaded_hashes)
		MEM_FREE(loaded_hashes);
	if (ocl_hc_hash_ids)
		MEM_FREE(ocl_hc_hash_ids);
	if (offset_table)
		MEM_FREE(offset_table);
	if (bt_hash_table_64)
		MEM_FREE(bt_hash_table_64);

	loaded_hashes = (cl_uint*) mem_alloc(2 * ocl_hc_num_loaded_hashes * sizeof(cl_uint));
	ocl_hc_hash_ids = (cl_uint*) mem_calloc((3 * ocl_hc_num_loaded_hashes + 1), sizeof(cl_uint));

	last = pw = salt->list;
	i = 0;
	do {
		bin = (unsigned int*)pw->binary;
		if (bin == NULL) {
			if (last == pw)
				salt->list = pw->next;
			else
				last->next = pw->next;
		} else {
			last = pw;
			loaded_hashes[2 * i] = bin[0];
			loaded_hashes[2 * i + 1] = bin[1];
			i++;
		}
	} while ((pw = pw->next)) ;

	if (i != (salt->count)) {
		fprintf(stderr,
			"Something went wrong while preparing hashes..Exiting..\n");
		error();
	}

	ocl_hc_num_loaded_hashes =
		bt_create_perfect_hash_table(64, (void*)loaded_hashes,
		                          ocl_hc_num_loaded_hashes,
		                          &offset_table,
		                          &ocl_hc_offset_table_size,
		                          &ocl_hc_hash_table_size, 0);

	if (!ocl_hc_num_loaded_hashes) {
		MEM_FREE(bt_hash_table_64);
		MEM_FREE(offset_table);
		fprintf(stderr, "Failed to create Hash Table for cracking.\n");
		error();
	}
}

/* Naive implementation for simplicity - we don't need speed. */
static uint32_t log2_64(uint64_t val)
{
        uint32_t res = 0;

        while (val >>= 1)
                res++;

        return res;
}

/*
 * Rotate, for slicing k > 4.  The kernel will just shift instead,
 * whenever it knows the end result is the same after the mask.
 */
#define ror(a, b)	(((a) << (32 - (b))) | ((a) >> (b)))

static void prepare_bitmap_k(struct db_salt *salt, uint64_t m, uint32_t k, uint32_t **bitmap_ptr)
{
	const uint32_t mask = m - 1;
	struct db_password *pw = salt->list;
	uint32_t *hash;

	MEM_FREE(*bitmap_ptr);
	*bitmap_ptr = mem_calloc(m >> 5, sizeof(uint32_t));

	do {
		if (!(hash = (unsigned int*)pw->binary))
			continue;

		uint32_t bmp_idx;
		uint32_t a = hash[0];
		uint32_t b = hash[1];
		uint32_t c = hash[2];
		uint32_t d = hash[3];

		bmp_idx = b & mask;
		(*bitmap_ptr)[bmp_idx >> 5] |= (1U << (bmp_idx & 31));

		if (k >= 2) {
			bmp_idx = a & mask;
			(*bitmap_ptr)[bmp_idx >> 5] |= (1U << (bmp_idx & 31));
		}
		if (k >= 3) {
			bmp_idx = c & mask;
			(*bitmap_ptr)[bmp_idx >> 5] |= (1U << (bmp_idx & 31));
		}
		if (k >= 4) {
			bmp_idx = d & mask;
			(*bitmap_ptr)[bmp_idx >> 5] |= (1U << (bmp_idx & 31));
		}
		if (k >= 5) {
			bmp_idx = ror(b, 16) & mask;
			(*bitmap_ptr)[bmp_idx >> 5] |= (1U << (bmp_idx & 31));
		}
		if (k >= 6) {
			bmp_idx = ror(a, 16) & mask;
			(*bitmap_ptr)[bmp_idx >> 5] |= (1U << (bmp_idx & 31));
		}
		if (k >= 7) {
			bmp_idx = ror(c, 16) & mask;
			(*bitmap_ptr)[bmp_idx >> 5] |= (1U << (bmp_idx & 31));
		}
		if (k == 8) {
			bmp_idx = ror(d, 16) & mask;
			(*bitmap_ptr)[bmp_idx >> 5] |= (1U << (bmp_idx & 31));
		}
	} while ((pw = pw->next)) ;
}

#define RED "\x1b[0;31m"
#define YEL "\x1b[0;33m"
#define NRM "\x1b[0m"

static double calc_k(double m, double n)
{
	return MIN(8, MAX(1, round(m / n * log(2))));
}

static double calc_p(double m, double n, double k)
{
	if (!k)
		k = calc_k(m, n);

	return pow(1 - exp(-k / (m / n)), k);
}

#if 0
/*
 * Added for reference - these are not used.
 */
static double calc_n(double m, double k, double p)
{
	return ceil(m / (-k / log(1 - exp(log(p) / k))));
}

static double calc_m(double n, double p)
{
	uint m = ceil((n * log(p)) / log(1 / pow(2, log(2))));

	if (m & (m - 1))
		get_power_of_two(m);

	return (double)m;
}
#endif

#define NEW_ALGO	1

char* ocl_hc_64_select_bitmap(struct db_salt *salt)
{
	static char kernel_params[200];
	char *env = NULL;
	unsigned int k = 2, use_local = 0, n = salt->count;
	uint64_t max_local_mem_sz = get_local_memory_size(gpu_id);
	uint64_t max_global_mem_sz = get_max_mem_alloc_size(gpu_id);
	double m;
#if NEW_ALGO
	unsigned int old_k;
	double target_p;

#define FITS_LOCAL(m)	((m) < (max_local_mem_sz * 8))
#define FITS_GLOBAL(m)	((m) < (max_global_mem_sz * 8))

#endif

	if (max_global_mem_sz > 0x20000000)
		max_global_mem_sz = 0x20000000;
	else if (max_global_mem_sz & (max_global_mem_sz - 1)) {
		get_power_of_two(max_global_mem_sz);
		max_global_mem_sz >>= 1;
	}

	/* Buggy MacOS reports 64KB while it has only 32KB */
	if (platform_apple(platform_id) && max_local_mem_sz == 65536)
		max_local_mem_sz >>= 1;

#if NEW_ALGO
	/*
	 * nvidia is less hurt by FP than AMD, but benefits more from using
	 * local.
	 *
	 * nvidia sometimes benefit from disabling bitmap and have PHT in
	 * local memory.  AMD never does.
	 */
	if (n < 65536)
		target_p = 1.0 / 128;
	else
		target_p = 1.0 / 64;
#endif

	if ((env = getenv("BITMAP_SIZE"))) {
		m = bitmap_size_bits = (uint64_t)strtoull(env, NULL, 0);

		if (bitmap_size_bits & (bitmap_size_bits - 1))
			error_msg("\nError: BITMAP_SIZE=%s is not a log2 multiple\n", env);
		else
		if (bitmap_size_bits && bitmap_size_bits < 32)
			error_msg("\nError: BITMAP_SIZE=%s - must be >= 32, or 0 to disable\n", env);
		else if (bitmap_size_bits > 0x100000000ULL)
			error_msg("\nError: BITMAP_SIZE=%s - max. allowed is 0x100000000\n", env);
		else if (bitmap_size_bits / 8 > max_global_mem_sz)
			error_msg("\nError: BITMAP_SIZE=%s - device's max. is 0x%"PRIx64"\n", env, max_global_mem_sz * 8);
	} else
#if NEW_ALGO
	{
#if 1
		bitmap_size_bits = max_local_mem_sz * 8;
		if (bitmap_size_bits & (bitmap_size_bits - 1)) {
			get_power_of_two(bitmap_size_bits);
			bitmap_size_bits >>= 1;
		}
		m = bitmap_size_bits;
		old_k = k = calc_k(m, n);
		if (!ocl_any_test_running)
			fprintf(stderr, "Init: n=%u m=%.0f: k=%u p=1/%.0f\n", n, m, k, 1 / calc_p(m, n, k));
		do {
			k = calc_k(m, n);
			while (k > 1 && calc_p(m, n, k - 1) <= target_p)
				k--;
		} while (calc_p(m / 2, n, 0) <= target_p && (m = bitmap_size_bits /= 2));

		if (!ocl_any_test_running && k != old_k)
			fprintf(stderr, "Decreased to m=%.0f: k=%u p=1/%.0f\n", m, k, 1 / calc_p(m, n, k));
		if (m < 32)
			k = m = bitmap_size_bits = 0;
#else
		m = MAX(32, calc_m(n, target_p));
		bitmap_size_bits = MIN(m, max_global_mem_sz * 8);
		if (bitmap_size_bits & (bitmap_size_bits - 1)) {
			get_power_of_two(bitmap_size_bits);
			bitmap_size_bits >>= 1;
		}
		m = bitmap_size_bits;
#endif
	}
#else
	if ((env = getenv("BLOOM_K")) && strtoul(env, NULL, 0) == 0)
		bitmap_size_bits = 0;
	else
	if (n <= 5100) {
		if (amd_gcn_10(device_info[gpu_id]) || amd_vliw4(device_info[gpu_id]))
			bitmap_size_bits = 1024 * 1024;
		else if (amd_gcn_11(device_info[gpu_id]) || max_local_mem_sz < 16384 || cpu(device_info[gpu_id]))
			bitmap_size_bits = 512 * 1024;
		else {
			bitmap_size_bits = 128 * 1024;
			k = 4;
		}
	}
	else if (n <= 10100) {
		if (amd_gcn_10(device_info[gpu_id]) || amd_vliw4(device_info[gpu_id]))
			bitmap_size_bits = 1024 * 1024;
		else if (amd_gcn_11(device_info[gpu_id]) || max_local_mem_sz < 32768 || cpu(device_info[gpu_id]))
			bitmap_size_bits = 512 * 1024;
		else {
			bitmap_size_bits = 256 * 1024;
			k = 4;
		}
	}
	else if (n <= 20100) {
		if (amd_gcn_10(device_info[gpu_id]))
			bitmap_size_bits = 2 * 1024 * 1024;
		else if (amd_gcn_11(device_info[gpu_id]) || max_local_mem_sz < 32768)
			bitmap_size_bits = 1024 * 1024;
		else if (amd_vliw4(device_info[gpu_id]) || cpu(device_info[gpu_id])) {
			bitmap_size_bits = 1024 * 1024;
			k = 4;
		}
		else {
			/* The 128-bit version had this as k=8, we might want k=4 */
			bitmap_size_bits = 256 * 1024;
			k = 8;
		}
	}
	else if (n <= 250100)
		if (max_local_mem_sz < 65536)
			bitmap_size_bits = 4 * 1024 * 1024;
		else
			bitmap_size_bits = 256 * 1024;
	else if (n <= 1100100) {
		if (!amd_gcn_11(device_info[gpu_id]))
			bitmap_size_bits = 8 * 1024 * 1024;
		else
			bitmap_size_bits = 4 * 1024 * 1024;
	}
	else if (n <= 1500100) {
		bitmap_size_bits = 4096 * 1024 * 2;
		k = 1;
	}
	else if (n <= 2700100) {
		bitmap_size_bits = 4096 * 1024 * 2 * 2;
		k = 1;
	}
	else {
		cl_ulong mult = n / 2700100;
		bitmap_size_bits = 4096 * 4096;
		get_power_of_two(mult);
		bitmap_size_bits *= mult;
		if ((bitmap_size_bits / 8) > max_global_mem_sz)
			bitmap_size_bits = max_global_mem_sz * 8;
	}
#endif

	m = bitmap_size_bits;

	if (m && (env = getenv("BLOOM_K"))) {
		k = (unsigned int)strtoul(env, NULL, 0);

		if (k > 8)
			error_msg("\nError: BLOOM_K=%s - must be 1..8, or 0 to disable\n", env);
	}
#if NEW_ALGO
	else if (bitmap_size_bits) {
		old_k = k;
		while (calc_p(2 * m, n, 0) > (FITS_LOCAL(m) ? 1.0 / 64 : target_p) && FITS_GLOBAL(2 * m)) {
			m = bitmap_size_bits *= 2;
			k = MIN(8, MAX(1, calc_k(m, n)));
		};

		if (!ocl_any_test_running && old_k != k)
			fprintf(stderr, "Target is 1/%.0f, trying m=%.0f k=%u p=1/%.0f\n", 1 / target_p, m, k, 1 / calc_p(m, n, k));

		/* Decrease too good FP ratio (due to rounding up m to nearest log2) */
		while (k > 8 || (k > 1 && calc_p(m, n, k) < target_p && calc_p(m, n, k - 1) <= target_p))
			k--;

		if (old_k != k && !ocl_any_test_running)
			fprintf(stderr, "Decreased to m=%.0f k=%u p=1/%.0f\n", m, k, 1 / calc_p(m, n, k));

		old_k = k;
		/* Increase too poor FP ratio or, k out of bounds */
		while (!k || (k < 8 && calc_p(m, n, k) > target_p && calc_p(m, n, k + 1) <= calc_p(m, n, k)))
			k++;

		if (old_k != k && !ocl_any_test_running)
			fprintf(stderr, "Increased to m=%.0f k=%u p=1/%.0f\n", m, k, 1 / calc_p(m, n, k));
	}
#endif

	/* Setting size or k to zero disables bitmap */
	if (bitmap_size_bits < 32)
		bitmap_size_bits = k = 0;
	else if (!k)
		bitmap_size_bits = 0;

	if ((env = getenv("USE_LOCAL"))) {
		use_local = (unsigned int)strtoul(env, NULL, 0);

		if (use_local > 1U)
			error_msg("\nError: USE_LOCAL=%s - must be 0 or 1\n", env);
		else if (use_local && (bitmap_size_bits / 8) > max_local_mem_sz)
			error_msg("\nError: BITMAP_SIZE=%"PRIu64" - device's local memory allows 0x%"PRIx64"\n",
			          bitmap_size_bits, (uint64_t)max_local_mem_sz * 8);
	}
	else {
		if (bitmap_size_bits)
			use_local = ((bitmap_size_bits / 8) < max_local_mem_sz);
		else
			use_local = ((ocl_hc_offset_table_size + 2 * ocl_hc_hash_table_size) * sizeof(int) < max_local_mem_sz);
	}

	uint32_t bitmap_size_log = log2_64(bitmap_size_bits);

	if (!ocl_any_test_running) {
#define b_stats	"Bloom filter: n=%u, m=%"PRIu64" (%u-bit, %sB %s) k=%u, expected fp: 1/%.0f (p=%f).", \
			(uint32_t)n, bitmap_size_bits, bitmap_size_log, human_prefix(bitmap_size_bits / 8), \
			use_local ? "local": "global", k, 1.0 / calc_p(m, n, k), calc_p(m, n, k)

#define h_stats	"%u hashes: Hash table in %s memory (%sB); bloom filter disabled.", \
			(uint32_t)n, use_local ? "local" : "global", \
			human_prefix((ocl_hc_offset_table_size + 2 * ocl_hc_hash_table_size) * sizeof(OFFSET_TABLE_WORD))

		if (options.verbosity /* >= VERB_DEBUG */) {
			if (bitmap_size_bits)
				fprintf(stderr, YEL b_stats);
			else
				fprintf(stderr, YEL h_stats);
			fprintf(stderr, "\n" NRM);

			fprintf(stderr,
			        "Offset tbl %sB, Hash tbl %sB, Results %sB, Dupe bmp %sB, TOTAL on GPU: %sB\n",
			        human_prefix(ocl_hc_offset_table_size * sizeof(OFFSET_TABLE_WORD)),
			        human_prefix(ocl_hc_hash_table_size * 2 * sizeof(unsigned int)),
			        human_prefix((3 * ocl_hc_num_loaded_hashes + 1) * sizeof(cl_uint)),
			        human_prefix((ocl_hc_hash_table_size / 32 + 1) * sizeof(cl_uint)),
			        human_prefix((ocl_hc_offset_table_size + ocl_hc_hash_table_size * 2 + 3 * ocl_hc_num_loaded_hashes + 1 + ocl_hc_hash_table_size / 32 + 1) * sizeof(cl_uint) + (bitmap_size_bits / 8)));
		}

		if (bitmap_size_bits)
			log_event(b_stats);
		else
			log_event(h_stats);
	}

	if (bitmap_size_bits)
		prepare_bitmap_k(salt, bitmap_size_bits, k, &bitmaps);
	else
		MEM_FREE(bitmaps);

	sprintf(kernel_params,
	        "-D BLOOM_K=%u -D BITMAP_SIZE_LOG=%u%s", k, bitmap_size_log, use_local ? " -D USE_LOCAL" : "");

	return kernel_params;
}

void ocl_hc_64_crobj(cl_kernel kernel)
{
	cl_ulong max_alloc_size_bytes = 0;

	HANDLE_CLERROR(clGetDeviceInfo(devices[gpu_id], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc_size_bytes, 0), "failed to get CL_DEVICE_MAX_MEM_ALLOC_SIZE.");

	if (max_alloc_size_bytes & (max_alloc_size_bytes - 1)) {
		get_power_of_two(max_alloc_size_bytes);
		max_alloc_size_bytes >>= 1;
	}
	if (max_alloc_size_bytes >= 0x20000000)
		max_alloc_size_bytes = 0x20000000;

	zero_buffer = (cl_uint*) mem_calloc(ocl_hc_hash_table_size/32 + 1, sizeof(cl_uint));

	buffer_hash_ids_64 = clCreateBuffer(context[gpu_id], CL_MEM_READ_WRITE, (3 * ocl_hc_num_loaded_hashes + 1) * sizeof(cl_uint), NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_buffer_hash_ids_64.");

	buffer_bitmap_dupe = clCreateBuffer(context[gpu_id], CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (ocl_hc_hash_table_size/32 + 1) * sizeof(cl_uint), zero_buffer, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_bitmap_dupe.");

	buffer_bitmaps = clCreateBuffer(context[gpu_id], CL_MEM_READ_WRITE, max_alloc_size_bytes, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_bitmaps.");

	buffer_offset_table = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY, ocl_hc_offset_table_size * sizeof(OFFSET_TABLE_WORD), NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_offset_table.");

	buffer_hash_table = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY, ocl_hc_hash_table_size * sizeof(unsigned int) * 2, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_hash_table.");

	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_hash_ids_64, CL_TRUE, 0, sizeof(cl_uint), zero_buffer, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_hash_ids_64.");
	if (bitmap_size_bits)
		HANDLE_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_bitmaps, CL_TRUE, 0, (size_t)(bitmap_size_bits / 8), bitmaps, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_bitmaps.");
	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_offset_table, CL_TRUE, 0, sizeof(OFFSET_TABLE_WORD) * ocl_hc_offset_table_size, offset_table, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_offset_table.");
	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_hash_table, CL_TRUE, 0, sizeof(cl_uint) * ocl_hc_hash_table_size * 2, bt_hash_table_64, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_hash_table.");

	HANDLE_CLERROR(clSetKernelArg(kernel, 4, sizeof(buffer_bitmaps), (void*) &buffer_bitmaps), "Error setting argument 5.");
	HANDLE_CLERROR(clSetKernelArg(kernel, 5, sizeof(buffer_offset_table), (void*) &buffer_offset_table), "Error setting argument 6.");
	HANDLE_CLERROR(clSetKernelArg(kernel, 6, sizeof(buffer_hash_table), (void*) &buffer_hash_table), "Error setting argument 7.");
	HANDLE_CLERROR(clSetKernelArg(kernel, 7, sizeof(buffer_hash_ids_64), (void*) &buffer_hash_ids_64), "Error setting argument 8.");
	HANDLE_CLERROR(clSetKernelArg(kernel, 8, sizeof(buffer_bitmap_dupe), (void*) &buffer_bitmap_dupe), "Error setting argument 9.");
}

int ocl_hc_64_extract_info(struct db_salt *salt, void (*set_kernel_args)(void), void (*set_kernel_args_kpc)(void), void (*init_kernel)(char*), size_t gws, size_t *lws, int *pcount)
{
	if (salt != NULL && salt->count > 4500 &&
		(ocl_hc_num_loaded_hashes - ocl_hc_num_loaded_hashes / 10) > salt->count) {
		size_t old_ot_sz_bytes, old_ht_sz_bytes;

		ocl_hc_64_prepare_table(salt);
		init_kernel(ocl_hc_64_select_bitmap(salt));

		BENCH_CLERROR(clGetMemObjectInfo(buffer_offset_table, CL_MEM_SIZE, sizeof(size_t), &old_ot_sz_bytes, NULL), "failed to query buffer_offset_table.");

		if (old_ot_sz_bytes < ocl_hc_offset_table_size * sizeof(OFFSET_TABLE_WORD)) {
			BENCH_CLERROR(clReleaseMemObject(buffer_offset_table), "Error Releasing buffer_offset_table.");

			buffer_offset_table = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY, ocl_hc_offset_table_size * sizeof(OFFSET_TABLE_WORD), NULL, &ret_code);
			BENCH_CLERROR(ret_code, "Error creating buffer argument buffer_offset_table.");
		}

		BENCH_CLERROR(clGetMemObjectInfo(buffer_hash_table, CL_MEM_SIZE, sizeof(size_t), &old_ht_sz_bytes, NULL), "failed to query buffer_hash_table.");

		if (old_ht_sz_bytes < ocl_hc_hash_table_size * sizeof(cl_uint) * 2) {
			BENCH_CLERROR(clReleaseMemObject(buffer_hash_table), "Error Releasing buffer_hash_table.");
			BENCH_CLERROR(clReleaseMemObject(buffer_bitmap_dupe), "Error Releasing buffer_bitmap_dupe.");
			MEM_FREE(zero_buffer);

			zero_buffer = (cl_uint*) mem_calloc(ocl_hc_hash_table_size/32 + 1, sizeof(cl_uint));
			buffer_bitmap_dupe = clCreateBuffer(context[gpu_id], CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ((ocl_hc_hash_table_size - 1) / 32 + 1) * sizeof(cl_uint), zero_buffer, &ret_code);
			BENCH_CLERROR(ret_code, "Error creating buffer argument buffer_bitmap_dupe.");
			buffer_hash_table = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY, ocl_hc_hash_table_size * sizeof(cl_uint) * 2, NULL, &ret_code);
			BENCH_CLERROR(ret_code, "Error creating buffer argument buffer_hash_table.");
		}

		if (bitmap_size_bits)
			BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_bitmaps, CL_TRUE, 0, (bitmap_size_bits / 8), bitmaps, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_bitmaps.");
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_offset_table, CL_TRUE, 0, sizeof(OFFSET_TABLE_WORD) * ocl_hc_offset_table_size, offset_table, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_offset_table.");
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_hash_table, CL_TRUE, 0, sizeof(cl_uint) * ocl_hc_hash_table_size * 2, bt_hash_table_64, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_hash_table.");

		BENCH_CLERROR(clSetKernelArg(crypt_kernel, 4, sizeof(buffer_bitmaps), (void*) &buffer_bitmaps), "Error setting argument 5.");
		BENCH_CLERROR(clSetKernelArg(crypt_kernel, 5, sizeof(buffer_offset_table), (void*) &buffer_offset_table), "Error setting argument 6.");
		BENCH_CLERROR(clSetKernelArg(crypt_kernel, 6, sizeof(buffer_hash_table), (void*) &buffer_hash_table), "Error setting argument 7.");
		BENCH_CLERROR(clSetKernelArg(crypt_kernel, 7, sizeof(buffer_hash_ids_64), (void*) &buffer_hash_ids_64), "Error setting argument 8.");
		BENCH_CLERROR(clSetKernelArg(crypt_kernel, 8, sizeof(buffer_bitmap_dupe), (void*) &buffer_bitmap_dupe), "Error setting argument 9.");
		set_kernel_args();
		set_kernel_args_kpc();
	}

	BENCH_CLERROR(clFinish(queue[gpu_id]), "clFinish");
	WAIT_INIT(gws)

	BENCH_CLERROR(clEnqueueNDRangeKernel(queue[gpu_id], crypt_kernel, 1, NULL, &gws, lws, 0, NULL, multi_profilingEvent[2]), "failed in clEnqueueNDRangeKernel");

	BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], buffer_hash_ids_64, CL_FALSE, 0, sizeof(cl_uint), ocl_hc_hash_ids, 0, NULL, multi_profilingEvent[3]), "failed in reading back num cracked hashes.");

	BENCH_CLERROR(clFlush(queue[gpu_id]), "failed in clFlush");
	WAIT_SLEEP
	BENCH_CLERROR(clFinish(queue[gpu_id]), "failed in clFinish");
	WAIT_UPDATE
	WAIT_DONE

	if (ocl_hc_hash_ids[0] > ocl_hc_num_loaded_hashes)
		error_msg("Error, crypt_all kernel.\n");

	if (ocl_hc_hash_ids[0]) {
		BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], buffer_hash_ids_64, CL_TRUE, 0, (3 * ocl_hc_hash_ids[0] + 1) * sizeof(cl_uint), ocl_hc_hash_ids, 0, NULL, NULL), "failed in reading data back ocl_hc_hash_ids.");
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_bitmap_dupe, CL_FALSE, 0, ((ocl_hc_hash_table_size - 1) / 32 + 1) * sizeof(cl_uint), zero_buffer, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_bitmap_dupe.");
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_hash_ids_64, CL_FALSE, 0, sizeof(cl_uint), zero_buffer, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_hash_ids_64.");
	}

	*pcount *= mask_int_cand.num_int_cand;
	return ocl_hc_hash_ids[0];
}

void ocl_hc_64_rlobj(void)
{
	if (buffer_bitmaps) {
		HANDLE_CLERROR(clReleaseMemObject(buffer_offset_table), "Error Releasing buffer_offset_table.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_hash_table), "Error Releasing buffer_hash_table.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_bitmap_dupe), "Error Releasing buffer_bitmap_dupe.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_hash_ids_64), "Error Releasing buffer_hash_ids_64.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_bitmaps), "Error Releasing buffer_bitmap.");
		MEM_FREE(zero_buffer);
		buffer_bitmaps = NULL;
	}

	if (loaded_hashes)
		MEM_FREE(loaded_hashes);
	if (ocl_hc_hash_ids)
		MEM_FREE(ocl_hc_hash_ids);
	if (bitmaps)
		MEM_FREE(bitmaps);
	if (offset_table)
		MEM_FREE(offset_table);
	if (bt_hash_table_64)
		MEM_FREE(bt_hash_table_64);
}

int ocl_hc_64_cmp_all(void *binary, int count)
{
	return count;
}

int ocl_hc_64_cmp_one(void *binary, int index)
{
	int result = (((unsigned int*)binary)[0] == bt_hash_table_64[ocl_hc_hash_ids[3 + 3 * index]] &&
	              ((unsigned int*)binary)[1] == bt_hash_table_64[ocl_hc_hash_table_size + ocl_hc_hash_ids[3 + 3 * index]]);
	return result;
}

#endif /* HAVE_OPENCL */
