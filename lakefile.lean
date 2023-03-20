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

extern_lib «ggml» (pkg : Package) := do
  -- build with cmake and make
  let ggmlBaseDir : FilePath := pkg.dir / "ggml"
  let ggmlBuildDir := ggmlBaseDir / "build"
  IO.FS.createDirAll ggmlBuildDir
  proc { cmd := "cmake", args := #["../", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"], cwd := ggmlBuildDir }
  proc { cmd := "cmake", args := #["--build", "src/"], cwd := ggmlBuildDir }
  -- copy library
  let tgtPath := pkg.libDir / "libggml.a"
  IO.FS.writeBinFile tgtPath (← IO.FS.readBinFile (ggmlBuildDir / "src" / "libggml.a"))
  -- give library to lake
  pure (BuildJob.pure tgtPath)
