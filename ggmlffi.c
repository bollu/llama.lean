/*
Author: Siddharth Bhat
FFI bindings to GGWP
*/

#include <lean/lean.h>
#include "ggml/ggml.h"


// unbox an integer.
int64_t lean_unbox_int64(lean_object *o) {
  assert(lean_is_scalar(o) && "given too large integer");
  return lean_scalar_to_int64(o);
}

// box an integer.
lean_object *lean_box_int64(int64_t i) {
  return lean_int64_to_int(i);
}

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
lean_object *lean_ggml_new_f32(size_t ctx, float value, lean_object * /* w */) {
  struct ggml_tensor *out = ggml_new_f32((struct ggml_context *)ctx, value);
  return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
}


lean_object *lean_ggml_new_tensor_1d (size_t ctx, size_t type, size_t nelem, lean_object * /* w */) {
	struct ggml_tensor *out = ggml_new_tensor_1d((struct ggml_context*)ctx,
								type, nelem);
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

lean_object *lean_ggml_mul(
    size_t ctx, size_t a, size_t b, lean_object * /* w */) {
	struct ggml_tensor *out = ggml_mul(
		(struct ggml_context*) ctx,
		(struct ggml_tensor*)a,
		(struct ggml_tensor*)b);
    return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
};

lean_object *lean_ggml_cpy(
    size_t ctx, size_t a, size_t b, lean_object * /* w */) {
	struct ggml_tensor *out = ggml_cpy(
		(struct ggml_context*) ctx,
		(struct ggml_tensor*)a,
		(struct ggml_tensor*)b);
    return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
};


lean_object *lean_ggml_scale(
    size_t ctx, size_t a, size_t b, lean_object * /* w */) {
	struct ggml_tensor *out = ggml_scale(
		(struct ggml_context*) ctx,
		(struct ggml_tensor*)a,
		(struct ggml_tensor*)b);
    return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
};

lean_object *lean_ggml_repeat(
    size_t ctx, size_t a, size_t b, lean_object * /* w */) {
	struct ggml_tensor *out = ggml_repeat(
		(struct ggml_context*) ctx,
		(struct ggml_tensor*)a,
		(struct ggml_tensor*)b);
    return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
};

lean_object *lean_ggml_diag_mask_inf(size_t ctx, size_t a, lean_object *npast_boxed) {
  int npast = lean_unbox_int64(npast_boxed);
  struct ggml_tensor *out = ggml_diag_mask_inf((struct ggml_context*)ctx,
					       (struct ggml_tensor *)a,
					       npast);
  return lean_io_result_mk_ok(lean_box_usize((size_t)out));
}

lean_object *lean_ggml_reshape_3d(
    size_t ctx, size_t tensor,
    lean_object *ne0_boxed, lean_object *ne1_boxed, lean_object *ne2_boxed,
    lean_object * /* w */) {
  int ne0 = lean_unbox_int64(ne0_boxed);
  int ne1 = lean_unbox_int64(ne1_boxed);
  int ne2 = lean_unbox_int64(ne2_boxed);
  struct ggml_tensor *out =
    ggml_reshape_3d((struct ggml_context*) ctx,
		    (struct ggml_tensor*)tensor,
		    ne0, ne1, ne2);
  return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
};


// -- unary ops ---

lean_object *lean_ggml_rms_norm(
    size_t ctx, size_t tensor,
    lean_object * /* w */) {
  struct ggml_tensor *out = ggml_rms_norm(
		(struct ggml_context*) ctx,
		(struct ggml_tensor*)tensor);
  return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
};

lean_object *lean_ggml_silu(
    size_t ctx, size_t tensor,
    lean_object * /* w */) {
  struct ggml_tensor *out = ggml_silu(
		(struct ggml_context*) ctx,
		(struct ggml_tensor*)tensor);
  return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
};

lean_object *lean_ggml_soft_max(
    size_t ctx, size_t tensor,
    lean_object * /* w */) {
  struct ggml_tensor *out = ggml_soft_max(
		(struct ggml_context*) ctx,
		(struct ggml_tensor*)tensor);
  return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
};



lean_object *lean_ggml_mul_mat(
    size_t ctx, size_t a, size_t b, lean_object * /* w */) {
	struct ggml_tensor *out = ggml_mul_mat(
		(struct ggml_context*) ctx,
		(struct ggml_tensor*)a,
		(struct ggml_tensor*)b);
    return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
};

lean_object *lean_ggml_new_tensor_2d (size_t ctx,
				size_t type, size_t ne0, size_t ne1, lean_object * /* w */) {
  // struct ggml_tensor *out = ggml_new_tensor_2d((struct ggml_context*)ctx,
  //								type, ne0, ne1);
  struct ggml_tensor *out = NULL;
  return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
}

lean_object *lean_ggml_new_tensor_3d (size_t ctx,
				size_t type, size_t ne0, size_t ne1, size_t ne2, lean_object * /* w */) {
	struct ggml_tensor *out = ggml_new_tensor_3d((struct ggml_context*)ctx,
								type, ne0, ne1, ne2);
    return lean_io_result_mk_ok(lean_box_usize((size_t)(out)));
}

lean_object *lean_ggml_permute(size_t ctx,
			       size_t tensor,
			       size_t ax0,
			       size_t ax1,
			       size_t ax2,
			       size_t ax3) {
  struct ggml_tensor *out =
    ggml_permute((struct ggml_context *)ctx,
		 (struct ggml_tensor *)tensor, ax0, ax1, ax2, ax3);
  return lean_io_result_mk_ok(lean_box_usize((size_t)out));

}


// TODO: Fix int handling.
lean_object *lean_ggml_rope(size_t ctx,
			    size_t tensor,
			    lean_object *npast_boxed,
			    lean_object *ndims_boxed,
			    lean_object *mode_boxed) {
  int npast = lean_unbox_int64(npast_boxed);
  int ndims = lean_unbox_int64(ndims_boxed);
  int mode = lean_unbox_int64(mode_boxed);
  struct ggml_tensor *out =
    ggml_rope((struct ggml_context *)ctx,
	      (struct ggml_tensor *)tensor, npast, ndims, mode);
  return lean_io_result_mk_ok(lean_box_usize((size_t)out));

}

lean_object *lean_ggml_view_1d(size_t ctx,
			       size_t tensor,
			       lean_object* ne0_boxed,
			       size_t offset) { // TODO: check how Int64 is sent on the wire.
  int ne0 = lean_unbox_int64(ne0_boxed);
  struct ggml_tensor *out =
    ggml_view_1d((struct ggml_context *)ctx,
		 (struct ggml_tensor *)tensor, ne0, offset);
    return lean_io_result_mk_ok(lean_box_usize((size_t)out));
}

lean_object *lean_ggml_blck_size (size_t ty) {
  int out =  ggml_blck_size(ty);
  return lean_io_result_mk_ok(lean_int64_to_int(out));
}

lean_object* lean_ggml_nbytes (size_t ty) {
  uint64_t out =  ggml_nbytes((struct ggml_tensor*)ty);
  return lean_io_result_mk_ok(lean_box_uint64(out));
}

lean_object* lean_ggml_nelements (size_t ty) {
  int out =  ggml_nelements((struct ggml_tensor*)ty);
  return lean_io_result_mk_ok(lean_int64_to_int(out));
}

lean_object *lean_ggml_build_forward_expand (size_t ctx, size_t graph, size_t tensor) {
    ggml_build_forward_expand((void*)graph, (void*)tensor);
    return lean_io_result_mk_ok(lean_box(0));
}

lean_object *lean_ggml_used_mem (size_t ctx) {
  int out = ggml_used_mem((void*)ctx);
  return lean_io_result_mk_ok(lean_int64_to_int(out));
}

lean_object *lean_ggml_time_init() {
  ggml_time_init();
  return lean_io_result_mk_ok(lean_box(0));
}

lean_object *lean_ggml_time_us() {
  int result = ggml_time_us();
  return lean_io_result_mk_ok(lean_int64_to_int(result));
}


lean_object *lean_ggml_type_size(size_t type) {
  int result = ggml_type_size(type);
  return lean_io_result_mk_ok(lean_int64_to_int(result));
}

lean_object *lean_ggml_type_sizef(size_t type) {
  int result = ggml_type_sizef(type);
  return lean_io_result_mk_ok(lean_box_float(result));
}

size_t lean_ggml_element_size(lean_object *ctx, lean_object *a) {
  return ggml_element_size((struct ggml_tensor *)a);
}
