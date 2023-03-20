/*
Author: Siddharth Bhat
FFI bindings to GGWP
*/

#include <lean/lean.h>
#include "ggml/ggml.h"


// -- Context functions -- 
lean_object * lean_ggml_init(size_t mem_size, lean_object * /* w */) {
	struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
    };
    struct ggml_context *out = ggml_init(params);
    return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
}

lean_object *lean_ggml_print_objects(size_t ctx, lean_object * /* w */) {
	ggml_print_objects((struct ggml_context*)ctx);
	return lean_io_result_mk_ok(lean_box(0));	
}

lean_object *lean_ggml_free(size_t ctx, lean_object * /* w */) {
	ggml_free((struct ggml_context*)ctx);
	return lean_io_result_mk_ok(lean_box(0));
}

// -- Tensor functions --
lean_object *lean_ggml_new_tensor_1d (size_t ctx, size_t type, size_t nelem, lean_object * /* w */) {
	struct ggml_tensor *out = ggml_new_tensor_1d(ctx, type, nelem);
    return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
}

lean_object *lean_ggml_add(
    size_t ctx, size_t a, size_t b, lean_object * /* w */) {
	struct ggml_tensor *out = ggml_add(
		(struct ggml_context*) ctx,
		(struct ggml_tensor*)a,
		(struct ggml_tensor*)b);
    return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
};

