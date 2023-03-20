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
  IO.FS.createDirAll pkg.libDir
  IO.FS.writeBinFile tgtPath (← IO.FS.readBinFile (ggmlBuildDir / "src" / "libggml.a"))
  -- give library to lake
  pure (BuildJob.pure tgtPath)

extern_lib «ggml-ffi» (pkg : Package) := do
  -- see also: https://github.com/yatima-inc/RustFFI.lean/blob/2a397cbc0904e2d575862c4067b512b6cc6b65f8/lakefile.lean
  let srcFileName := "ggmlffi.c"
  let oFilePath := pkg.oleanDir / "libggmlffi.o"
  let srcJob ← inputFile srcFileName
  buildFileAfterDep oFilePath srcJob fun srcFile => do
    let flags := #["-I", (← getLeanIncludeDir).toString, 
                   "-I", (pkg.dir / "ggml" / "include").toString,
                   "-fPIC"]
    compileO srcFileName oFilePath srcFile flags -- build static archive
