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

def type.marshal : type -> USize
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
| dup --Identity function : Tensor k → Tensor k
| add --Addition : Tensor k → Tensor k → Tensor k
| sub --Subtraction : Tensor k → Tensor k → Tensor k
| mul --Pointwise Multiplication : Tensor k → Tensor k → Tensor k
| div --Pointwise Division : Tensor k → Tensor k → Tensor k
| sqr --Pointwise Squaring : Tensor k → Tensor k
| sqrt --Pointwise Square Root : Tensor k → Tensor k
| sum -- Add 'em up : Tensor k → Tensor (identity of tensor product)
| mean -- Average 'em out : Tensor k → Tensor (identity of tensor product)
| repeat_ -- Complicated
| abs -- Pointwise
| sgn -- Pointwise
| neg -- Pointwise
| step -- Pointwise fun x => if x < 0 then 0 else 1
| relu -- Pointwise fun x => if x < 0 then 0 else x
| gelu -- Pointwise fun x => x * 0.5 * (1 + erf(x/sqrt(2)))
| silu -- Pointwise fun x => x * 1/ (1+exp(-x))
| norm -- not the sqrt of sum of squares
| rms_norm -- Don't know yet : Tensor k -> Tensor k rms(a) = sqrt(1/n * sum_i (a_i^2)), rms_norm i = a_i
| mul_mat -- Matrix multiplication, undefined on stuff that isn't a matrix
| scale -- Tensor 0 → Tensor k → Tensor k
| cpy -- identity function
| reshape -- Tensor → Shape → Tensor --Changes the data to fit new layout of same (linear algebra sense) dimension
| view1D -- Tensor → (Num_elements : ℕ) → (offset : ℕ) → Tensor --Output at i = input at i+offset output
| view 2D -- Number of elements in 2 axes and offsets in 2 axes
| permute4D -- take a permutation of 4 numbers and Return a 4d tesnro with everything permuted
| transpose -- Matrix transpose
| get_rows -- (x : Matrix) (I : vector of Nats)  out(i,j) = x(I(i),j)
| diag_mask_inf -- Take a matrix and set upper triangle to -infinity
| soft_max -- Take a vector and exponentiate pointwise and normalize by L1-norm
| rope -- Screwed up rotatory positional embedding
| conv_1d_1s -- Polynomial multiplication of vectors
| conv_1d_2s -- Convolve 1d 2d
| flash_attn -- 3 input matrices Q K V, compute softmax (QKᵀ / sqrt(number of columns of q)) V
| flash_ff -- TODO
| count


def op.marshal : op -> USize
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
| silu => 17
| norm => 18 -- NORMALIZE
| rms_norm => 19
| mul_mat => 20
| scale => 21
| cpy => 22
| reshape => 23
| view => 24
| permute => 25
| transpose => 26
| get_rows => 27
| diag_mask_inf => 28
| soft_max => 29
| rope => 30
| conv_1d_1s => 31
| conv_1d_2s => 32
| flash_attn => 33
| flash_ff => 34
| count => 35

structure Context where
  private mk :: ptr : USize
instance : Nonempty Context := ⟨{ ptr := default }⟩

-- must use same context everyhere
structure Tensor (ctx: Context) where
  private mk :: ptr : USize
deriving Inhabited

structure Cgraph (ctx : Context) where
  private mk :: ptr : USize
deriving Inhabited

-- functions to be bound in GGML:
-- master ~/papers/llama/llama.cpp> rg "ggml_[a-zA-Z0-9_]*\(" main.cpp -o --no-line-number | sort | uniq
@[extern "lean_ggml_add"]
opaque ggml_add (a : Tensor ctx) (b : Tensor ctx) : BaseIO (Tensor ctx)

-- ggml_blck_size(
@[extern "lean_ggml_blck_size"]
opaque ggml_blck_size_ (t : USize) : BaseIO Int

def ggml_blck_size (t : type) : BaseIO Int := ggml_blck_size_ t.marshal

-- ggml_build_forward_expand(
@[extern "lean_ggml_build_forward_expand"]
opaque ggml_build_forward_expand (graph : Cgraph ctx) (tensor : Tensor ctx)  : BaseIO Unit
-- ggml_cpu_has_arm_fma(
-- ggml_cpu_has_avx(
-- ggml_cpu_has_avx2(
-- ggml_cpu_has_avx512(
-- ggml_cpu_has_blas(
-- ggml_cpu_has_f16c(
-- ggml_cpu_has_fma(
-- ggml_cpu_has_fp16_va(
-- ggml_cpu_has_neon(
-- ggml_cpu_has_sse3(
-- ggml_cpu_has_vsx(
-- ggml_cpu_has_wasm_simd(
-- ggml_cpy(
@[extern "lean_ggml_cpy"]
opaque ggml_cpy (a b : Tensor ctx) : BaseIO (Tensor ctx)

-- set elements above the diagonal to -INF
-- in-place, returns view(a).
-- ggml_diag_mask_inf(
@[extern "lean_ggml_diag_mask_inf"]
opaque ggml_diag_mask_inf (a : Tensor ctx) (n_past : Int) : BaseIO (Tensor ctx)

-- ggml_element_size(
@[extern "lean_ggml_element_size"]
opaque ggml_element_size (a : Tensor ctx) : BaseIO UInt64


-- void pointer?
-- ggml_get_data(
-- opaque ggml_get_data (a : Tensor ctx) : BaseIO VoidPtr

-- ggml_get_rows(
opaque ggml_get_rows (a b : Tensor ctx) : Tensor ctx

-- ggml_graph_dump_dot(
-- ggml_init(
@[extern "lean_ggml_init"]
opaque ggml_init (size : USize) : BaseIO (Context)

-- ggml_free(
@[extern "lean_ggml_free"]
opaque ggml_free (ctx : Context) : BaseIO (Unit)

@[extern "lean_ggml_print_objects"]
opaque ggml_print_objects (ctx : Context) : BaseIO (Unit)

-- ggml_mul(
@[extern "lean_ggml_mul"]
opaque ggml_mul (a b : Tensor ctx) : BaseIO (Tensor ctx)

-- ggml_mul_mat(
@[extern "lean_ggml_mul_mat"]
opaque ggml_mul_mat (a b : Tensor ctx) : BaseIO (Tensor ctx)

#check Int
-- ggml_nbytes(
@[extern "lean_ggml_nbytes"]
opaque ggml_nbytes (a : Tensor ctx) : BaseIO (UInt64)



-- ggml_nelements(
@[extern "lean_ggml_nelements"]
opaque ggml_nelements (a : Tensor ctx) : BaseIO Int

-- ggml_new_f32(
@[extern "lean_ggml_new_f32"]
opaque ggml_new_f32 (ctx : Context) (value : Float) : BaseIO (Tensor ctx)

-- ggml_new_tensor_1d(
@[extern "lean_ggml_new_tensor_1d"]
opaque ggml_new_tensor_1d_ (ctx : Context) (type : USize) (ne0 : USize) : BaseIO (Tensor ctx)

def ggml_new_tensor_1d (ctx: Context) (t : type) (ne0 : USize) : BaseIO (Tensor ctx) :=
  ggml_new_tensor_1d_ ctx t.marshal ne0

-- ggml_new_tensor_2d(
@[extern "lean_ggml_new_tensor_2d"]
opaque ggml_new_tensor_2d_ (ctx : Context) (type : USize) (ne0 ne1 : USize) : BaseIO (Tensor ctx)

def ggml_new_tensor_2d (ctx: Context) (t : type) (ne0 ne1 : USize) : BaseIO (Tensor ctx) :=
  ggml_new_tensor_2d_ ctx t.marshal ne0 ne1

-- ggml_new_tensor_3d(
@[extern "lean_ggml_new_tensor_3d"]
opaque ggml_new_tensor_3d_ (ctx : Context) (type : USize) (ne0 ne1 ne2 : USize) : BaseIO (Tensor ctx)

def ggml_new_tensor_3d (ctx: Context) (t : type) (ne0 ne1 ne2 : USize) : BaseIO (Tensor ctx) :=
  ggml_new_tensor_3d_ ctx t.marshal ne0 ne1 ne2

-- TODO: what does this actually do?
-- ggml_permute(
@[extern "lean_ggml_permute"]
opaque ggml_permute (a : Tensor ctx) (ax0 ax1 ax2 ax3 : Int) : BaseIO (Tensor ctx)

-- if a is the same shape as b, and a is not parameter, return a
-- otherwise, return a new tensor: repeat(a) to fit in b
-- ggml_repeat(
@[extern "lean_ggml_repeat"]
opaque ggml_repeat (a b : Tensor ctx) : BaseIO (Tensor ctx)

-- review view(a)
-- ggml_reshape_3d(
@[extern "lean_ggml_reshape_3d"]
opaque ggml_reshape_3d (a : Tensor ctx) (ne0 ne1 ne2 : Int)  : BaseIO (Tensor ctx)

-- TODO: where is this fro
@[extern "lean_ggml_rms_norm"]
opaque ggml_rms_norm (a : Tensor ctx) : BaseIO (Tensor ctx)

-- rotary position embedding
-- ggml_rope(
@[extern "lean_ggml_rope"]
opaque ggml_rope (a : Tensor ctx) (npast ndims mode : Int) : BaseIO (Tensor ctx)

-- ggml_scale(
@[extern "lean_ggml_scale"]
opaque ggml_scale (a b : Tensor ctx) : BaseIO (Tensor ctx)


-- TODO: what is silu? [OK, it's x ↦ xσ(x)]
@[extern "lean_ggml_silu"]
opaque ggml_silu (a : Tensor ctx) : BaseIO (Tensor ctx)

-- ggml_soft_max(
@[extern "lean_ggml_soft_max"]
opaque ggml_soft_max (a : Tensor ctx) : BaseIO (Tensor ctx)

-- ggml_time_init(
@[extern "lean_ggml_time_init"]
opaque ggml_time_init : BaseIO Unit

-- ggml_time_us(
@[extern "lean_ggml_time_us"]
opaque ggml_time_us : BaseIO Int

-- size in bytes for all elements in a block.
-- ggml_type_size(
@[extern "lean_ggml_type_size"]
opaque ggml_type_size_ (i : USize) : BaseIO Int
def ggml_type_size (t : type) : BaseIO Int := ggml_type_size_ t.marshal

-- return number of bytes as float
@[extern "lean_ggml_type_sizef"]
opaque ggml_type_sizef_ (i : USize) : BaseIO Float
opaque ggml_type_sizef (t : type) : BaseIO Float := ggml_type_sizef_ t.marshal

-- ggml_used_mem(
@[extern "lean_ggml_used_mem"]
opaque ggml_used_mem (ctx: Context): BaseIO Int

-- ggml_view_1d(
@[extern "lean_ggml_view_1d"]
opaque ggml_view_1d (t : Tensor ctx) (ne0 : Int) (offset : UInt64) : BaseIO (Tensor ctx)

end ggml




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

open ggml in
def main (args : List String) : IO UInt32 := do
  -- llama.validate args
  let ctx <- ggml_init (1024 * 1024 * 1024)
  let t0 <- ggml_new_tensor_1d ctx type.i32 10
  ggml_print_objects ctx
  ggml_free ctx
  return 0
end Main
