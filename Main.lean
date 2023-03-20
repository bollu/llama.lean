import «Llama»
import Cli
import Lean
import Std 

open Cli
open Std

section Util

section GGML
end GGML

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