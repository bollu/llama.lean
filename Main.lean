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

-- functions to be bound in GGML:
-- master ~/papers/llama/llama.cpp> rg "ggml_[a-zA-Z0-9_]*\(" main.cpp -o --no-line-number | sort | uniq
-- ggml_add(
-- ggml_blck_size(
-- ggml_build_forward_expand(
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
-- ggml_diag_mask_inf(
-- ggml_element_size(
-- ggml_free(
-- ggml_get_data(
-- ggml_get_rows(
-- ggml_graph_dump_dot(
-- ggml_init(
-- ggml_mul(
-- ggml_mul_mat(
-- ggml_nbytes(
-- ggml_nelements(
-- ggml_new_f32(
-- ggml_new_tensor_1d(
-- ggml_new_tensor_2d(
-- ggml_new_tensor_3d(
-- ggml_permute(
-- ggml_repeat(
-- ggml_reshape_3d(
-- ggml_rms_norm(
-- ggml_rope(
-- ggml_scale(
-- ggml_silu(
-- ggml_soft_max(
-- ggml_time_init(
-- ggml_time_us(
-- ggml_type_size(
-- ggml_type_sizef(
-- ggml_used_mem(
-- ggml_view_1d(

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


def main (args : List String) : IO UInt32 :=
  llama.validate args

end Main
