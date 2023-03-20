import «Llama»
import Cli
import Lean
import Std 

open Cli
open Std

section Util


namespace ggml
-- bindings to the GGML library

-- enum ggml_type {
--     GGML_TYPE_I8,
--     GGML_TYPE_I16,
--     GGML_TYPE_I32,
--     GGML_TYPE_F16,
--     GGML_TYPE_F32,
--     GGML_TYPE_COUNT,
-- };

inductive type
| i8
| i16
| i32
| f16
| f32
| count

def type.marshal : type -> Int
| .i8 => 0
| .i16 => 1
| .i32 => 2
| .f16 => 3
| .f32 => 4
| .count => 5

-- // available tensor operations:
-- enum ggml_op {
--     GGML_OP_NONE = 0,
-- 
--     GGML_OP_DUP,
--     GGML_OP_ADD,
--     GGML_OP_SUB,
--     GGML_OP_MUL,
--     GGML_OP_DIV,
--     GGML_OP_SQR,
--     GGML_OP_SQRT,
--     GGML_OP_SUM,
--     GGML_OP_MEAN,
--     GGML_OP_REPEAT,
--     GGML_OP_ABS,
--     GGML_OP_SGN,
--     GGML_OP_NEG,
--     GGML_OP_STEP,
--     GGML_OP_RELU,
--     GGML_OP_GELU,
--     GGML_OP_NORM, // normalize
-- 
--     GGML_OP_MUL_MAT,
-- 
--     GGML_OP_SCALE,
--     GGML_OP_CPY,
--     GGML_OP_RESHAPE,
--     GGML_OP_VIEW,
--     GGML_OP_PERMUTE,
--     GGML_OP_TRANSPOSE,
--     GGML_OP_GET_ROWS,
--     GGML_OP_DIAG_MASK_INF,
--     GGML_OP_SOFT_MAX,
--     GGML_OP_ROPE,
--     GGML_OP_CONV_1D_1S,
--     GGML_OP_CONV_1D_2S,
-- 
--     GGML_OP_FLASH_ATTN,
--     GGML_OP_FLASH_FF,
-- 
--     GGML_OP_COUNT,
-- };

inductive op
| none
| dup
| add
| sub
| mul
| div
| sqr
| sqrt
| sum
| mean
| repeat_
| abs
| sgn
| neg
| step
| relu
| gelu
| norm -- NORMALIZE
| mul_mat
| scale
| cpy
| reshape
| view
| permute
| transpose
| get_rows
| diag_mask_inf
| soft_max
| rope
| conv_1d_1s
| conv_1d_2s
| flash_attn
| flash_ff
| count


def op.marshal : op -> Int
| none => 0
| dup => 1
| add => 2
| sub => 3
| mul => 4
| div => 5
| sqr => 6
| sqrt => 7
| sum => 8
| mean => 9
| repeat_ => 10 -- TODO: why is 'repeat' taken?
| abs => 11
| sgn => 12
| neg => 13
| step => 14
| relu => 15
| gelu => 16
| norm => 17 -- NORMALIZE
| mul_mat => 18
| scale => 19
| cpy => 20
| reshape => 21
| view => 22
| permute => 23
| transpose => 24
| get_rows => 25
| diag_mask_inf => 26
| soft_max => 27
| rope => 28
| conv_1d_1s => 29
| conv_1d_2s => 30
| flash_attn => 31
| flash_ff => 32
| count => 33

end ggml

-- void    ggml_time_init(void); // call this once at the beginning of the program
-- int64_t ggml_time_ms(void);
-- int64_t ggml_time_us(void);
-- int64_t ggml_cycles(void);
-- int64_t ggml_cycles_per_ms(void);

-- void ggml_print_object (const struct ggml_object * obj);
-- void ggml_print_objects(const struct ggml_context * ctx);

-- int    ggml_nelements(const struct ggml_tensor * tensor);
-- size_t ggml_nbytes   (const struct ggml_tensor * tensor);

-- size_t ggml_type_size   (enum ggml_type type);
-- size_t ggml_element_size(const struct ggml_tensor * tensor);

-- struct ggml_context * ggml_init(struct ggml_init_params params);
-- void ggml_free(struct ggml_context * ctx);

-- size_t ggml_used_mem(const struct ggml_context * ctx);

-- size_t ggml_set_scratch(struct ggml_context * ctx, struct ggml_scratch scratch);

-- struct ggml_tensor * ggml_new_tensor(
--         struct ggml_context * ctx,
--         enum   ggml_type type,
--         int    n_dims,
--         const int *ne);

-- struct ggml_tensor * ggml_new_tensor_1d(
--         struct ggml_context * ctx,
--         enum   ggml_type type,
--         int    ne0);

-- struct ggml_tensor * ggml_new_tensor_2d(
--         struct ggml_context * ctx,
--         enum   ggml_type type,
--         int    ne0,
--         int    ne1);

-- struct ggml_tensor * ggml_new_tensor_3d(
--         struct ggml_context * ctx,
--         enum   ggml_type type,
--         int    ne0,
--         int    ne1,
--         int    ne2);

-- struct ggml_tensor * ggml_new_tensor_4d(
--         struct ggml_context * ctx,
--         enum   ggml_type type,
--         int    ne0,
--         int    ne1,
--         int    ne2,
--         int    ne3);

-- struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
-- struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

-- struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
-- struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, const struct ggml_tensor * src);

-- struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
-- struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
-- struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

-- int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
-- void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

-- float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
-- void  ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

--  void * ggml_get_data    (const struct ggml_tensor * tensor);
-- float * ggml_get_data_f32(const struct ggml_tensor * tensor);

-- //
-- // operations on tensors with backpropagation
-- //

-- struct ggml_tensor * ggml_dup(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- struct ggml_tensor * ggml_add(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b);

-- struct ggml_tensor * ggml_sub(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b);

-- struct ggml_tensor * ggml_mul(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b);

-- struct ggml_tensor * ggml_div(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b);

-- struct ggml_tensor * ggml_sqr(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- struct ggml_tensor * ggml_sqrt(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- // return scalar
-- // TODO: compute sum along rows
-- struct ggml_tensor * ggml_sum(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- // mean along rows
-- struct ggml_tensor * ggml_mean(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- // if a is the same shape as b, and a is not parameter, return a
-- // otherwise, return a new tensor: repeat(a) to fit in b
-- struct ggml_tensor * ggml_repeat(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b);

-- struct ggml_tensor * ggml_abs(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- struct ggml_tensor * ggml_sgn(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- struct ggml_tensor * ggml_neg(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- struct ggml_tensor * ggml_step(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- struct ggml_tensor * ggml_relu(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- // TODO: double-check this computation is correct
-- struct ggml_tensor * ggml_gelu(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- // normalize along rows
-- // TODO: eps is hardcoded to 1e-5 for now
-- struct ggml_tensor * ggml_norm(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- // A: m rows, n columns
-- // B: p rows, n columns (i.e. we transpose it internally)
-- // result is m columns, p rows
-- struct ggml_tensor * ggml_mul_mat(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b);

-- //
-- // operations on tensors without backpropagation
-- //

-- // in-place, returns view(a)
-- struct ggml_tensor * ggml_scale(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b);

-- // a -> b, return view(b)
-- struct ggml_tensor * ggml_cpy(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b);

-- // return view(a), b specifies the new shape
-- // TODO: when we start computing gradient, make a copy instead of view
-- struct ggml_tensor * ggml_reshape(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b);

-- // return view(a)
-- // TODO: when we start computing gradient, make a copy instead of view
-- struct ggml_tensor * ggml_reshape_2d(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         int                   ne0,
--         int                   ne1);

-- // return view(a)
-- // TODO: when we start computing gradient, make a copy instead of view
-- struct ggml_tensor * ggml_reshape_3d(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         int                   ne0,
--         int                   ne1,
--         int                   ne2);

-- // offset in bytes
-- struct ggml_tensor * ggml_view_1d(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         int                   ne0,
--         size_t                offset);

-- struct ggml_tensor * ggml_view_2d(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         int                   ne0,
--         int                   ne1,
--         size_t                nb1, // row stride in bytes
--         size_t                offset);

-- struct ggml_tensor * ggml_permute(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         int                   axis0,
--         int                   axis1,
--         int                   axis2,
--         int                   axis3);

-- // alias for ggml_permute(ctx, a, 1, 0, 2, 3)
-- struct ggml_tensor * ggml_transpose(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- struct ggml_tensor * ggml_get_rows(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b);

-- // set elements above the diagonal to -INF
-- // in-place, returns view(a)
-- struct ggml_tensor * ggml_diag_mask_inf(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         int                   n_past);

-- // in-place, returns view(a)
-- struct ggml_tensor * ggml_soft_max(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a);

-- // rotary position embedding
-- // in-place, returns view(a)
-- // if mode == 1, skip n_past elements
-- // TODO: avoid creating a new tensor every time
-- struct ggml_tensor * ggml_rope(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         int                   n_past,
--         int                   n_dims,
--         int                   mode);

-- // padding = 1
-- // TODO: we don't support extra parameters for now
-- //       that's why we are hard-coding the stride, padding, and dilation
-- //       not great ..
-- struct ggml_tensor * ggml_conv_1d_1s(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b);

-- struct ggml_tensor * ggml_conv_1d_2s(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b);

-- struct ggml_tensor * ggml_flash_attn(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * q,
--         struct ggml_tensor  * k,
--         struct ggml_tensor  * v,
--         bool                  masked);

-- struct ggml_tensor * ggml_flash_ff(
--         struct ggml_context * ctx,
--         struct ggml_tensor  * a,
--         struct ggml_tensor  * b0,
--         struct ggml_tensor  * b1,
--         struct ggml_tensor  * c0,
--         struct ggml_tensor  * c1);

-- //
-- // automatic differentiation
-- //

-- void ggml_set_param(
--         struct ggml_context * ctx,
--         struct ggml_tensor * tensor);

-- void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);

-- struct ggml_cgraph ggml_build_forward (struct ggml_tensor * tensor);
-- struct ggml_cgraph ggml_build_backward(struct ggml_context * ctx, struct ggml_cgraph * gf, bool keep);

-- void ggml_graph_compute(struct ggml_context * ctx, struct ggml_cgraph * cgraph);
-- void ggml_graph_reset  (struct ggml_cgraph * cgraph);

-- // print info and performance information for the graph
-- void ggml_graph_print(const struct ggml_cgraph * cgraph);

-- // dump the graph into a file using the dot format
-- void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);

-- struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type);

-- // optimize the function defined by the tensor f
-- enum ggml_opt_result ggml_opt(
--         struct ggml_context * ctx,
--         struct ggml_opt_params params,
--         struct ggml_tensor * f);



open Lean in
def String.toFloat? (s: String) : Option Float :=
  (Syntax.decodeScientificLitVal? s).map (fun ⟨m, s, e⟩ => Float.ofScientific m s e)

instance : ParseableType Float where
  name := "Int"
  parse?
    | "" => none
    | s  => s.toFloat?


-- https://github.com/ggerganov/llama.cpp/blob/da5303c1ea68aa19db829c634f1e10d08d409680/utils.h#L15
structure GptParams where
  seed : Int := -1
  nthreads : Nat := 4
  npredict : Nat := 128
  repeatLastN : Nat := 64
  nCtx : Nat := 512

  -- sampling parameters
  topK : Nat := 40 
  topP : Float := 0.95
  temp : Float := 0.8
  repeatPenalty : Float := 1.30

  nbatch : Nat := 8 -- batch size

  model : String := "models/lamma-7B/ggml-model.bin"
  prompt : String := ""

  randomPrompt : Bool := False
  useColor : Bool := False
  interactive : Bool := False -- interactive mode
  interactiveStart : Bool := False -- reverse prompt immediately
  antiprompt : List String -- string upon seeing which more user input is prompted
  instruct : Bool := False -- instruct mode.
  ignoreEos : Bool := False -- do not stop generating after EOS.


structure gptvocab where 
  token : String 
  token2id : HashMap Int String
  id2token : HashMap String Int 

-- TODO: do I need 'int32'?
abbrev gptvocab.id : Type := UInt32

def gptvocab.gpt_tokenize (vocab: gptvocab) (text: String): Array gptvocab := sorry

-- sentencepiece tokenization. 
def gptvocab.llama_tokenize (vocab: gptvocab) (text: String) (bos: Bool): Array gptvocab := sorry 

-- load the tokens from encoder.json
def gptvocab.init (fname : String): Option gptvocab := sorry 

-- random generation monad.
abbrev LlamaM := ReaderT GptParams IO 


-- sample next token given probabilities for each embedding
-- - consider only the top K tokens
-- - from them, consider only the top tokens with cumulative probability > P
def gptvocab.llama_sample_top_p_top_k (vocab: gptvocab) (logits: Array Float)
  (lastNTokens: Array gptvocab.id)
  -- (repeatPenalty : Double)
  -- (topK : Nat)
  -- (topP : Double)
  -- (temp : Double) 
  : LlamaM gptvocab.id := sorry 
 
-- filter to top K tokens from list of logits
def gptvocab.sample_top_k (vocab: gptvocab) 
  (logitsId : Array (Float × gptvocab.id))
  -- (topK: Nat)
  : LlamaM gptvocab.id := sorry 
end Util


section Main 


-- https://github.com/ggerganov/llama.cpp/blob/da5303c1ea68aa19db829c634f1e10d08d409680/utils.cpp#L18
def runLlama (p : Parsed) : IO UInt32 := do
  return 0

-- https://github.com/ggerganov/llama.cpp/blob/da5303c1ea68aa19db829c634f1e10d08d409680/utils.cpp#L18
def llama : Cmd := `[Cli|
  llama VIA runLlama; ["0.0.1"]
  "llama language model runner (powered by L∃∀N4)."

  FLAGS:
    i, interactive;                  "Declares a flag `--invert` with an associated short alias `-i`."
    ins, instruct;                "Declares a flag `--optimize` with an associated short alias `-o`."
    r, "reverse-prompt" : Array String; "Declares a flag `--set-paths` " ++
                                "that takes an argument of type `Array Nat`. " ++
                                "Quotation marks allow the use of hyphens."
    "color" : String; "colorize output"
    s, seed : Int; "seed"
    t, threads : Nat ; "number of threads to use during computation"
    p, prompt : String ; "prompt to start with"
    "random-prompt" : String ; "start with a randomized prompt"
    f, file : String ; "prompt file to start generation"
    n, n_predict : Nat ; "number of tokens to predict"
    top_k : Nat ; "top-k sampling"
    top_p : Nat ; "top-p sampling"
    repeat_last_n : Nat  ; "last n tokens to consider for penalize"
    repeat_last_p : Nat  ; "penalize repeat sequence of tokens"
    c, ctx_size : Nat ; "size of prompt context"
    "ignore-eos" ; "ignore end of stream token and continue generating"
    memory_f16 ; "use f16 instead of f32 for memory key+value"
    temp : Float ; "temperature"
    b, batch_size : Nat ; "batch size for prompt processing"
    m, model : String; "model path"


  ARGS:
    input : String;      "Declares a positional argument <input> " ++
                         "that takes an argument of type `String`."
    ...outputs : String; "Declares a variable argument <output>... " ++
                         "that takes an arbitrary amount of arguments of type `String`."

  -- SUBCOMMANDS:
  --   installCmd;
  --   testCmd

  -- The EXTENSIONS section denotes features that
  -- were added as an external extension to the library.
  -- `./Cli/Extensions.lean` provides some commonly useful examples.
  -- EXTENSIONS:
  --   author "mhuisi";
  --   defaultValues! #[("priority", "0")]


]


def main (args : List String) : IO UInt32 :=
  llama.validate args

end Main
