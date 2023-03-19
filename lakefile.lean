import Lake
open Lake DSL

require Alloy from git "https://github.com/tydeu/lean4-alloy.git"@"334407"
require Cli from git "https://github.com/mhuisi/lean4-cli.git"@"nightly"
require Std from git "https://github.com/leanprover/std4"@"529a6"
package «llama» {
  -- add package configuration options here
}

lean_lib «Llama» {
  -- add library configuration options here
}

@[default_target]
lean_exe «llama» {
  root := `Main
}
