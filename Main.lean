import Cli
import «Llama»
import Lean

open Cli

open Lean in
def String.toFloat? (s: String) : Option Float :=
  (Syntax.decodeScientificLitVal? s).map (fun ⟨m, s, e⟩ => Float.ofScientific m s e)

instance : ParseableType Float where
  name := "Int"
  parse?
    | "" => none
    | s  => s.toFloat?


structure GptParams where 
  nthreads : Nat
  npredict : Nat
  topk : Nat
  topp : Nat
  repeatLastN : Nat
  repeatPenalty : Float
  nctx : Nat
  temp : Float
  nbatch : Nat

def runLlama (p : Parsed) : IO UInt32 := do
  return 0

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
