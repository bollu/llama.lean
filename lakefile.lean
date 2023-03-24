import Lake
open Lake DSL



require Cli from git "https://github.com/mhuisi/lean4-cli.git"@"nightly"
require Std from git "https://github.com/leanprover/std4"@"529a6"
package «llama» {
  -- add package configuration options here
  extraDepTargets := #["ggmlffi-shim", "ggml"]
  moreLinkArgs := #["-L./build/lib/", "-lggmlffishim", "-lggml"]
}

lean_lib «Llama» {
  -- add library configuration options here
}

@[default_target]
lean_exe «llama» {
  root := `Main
  supportInterpreter := true
}

target «ggmlffi» (pkg : Package) : FilePath := do
  -- see also: https://github.com/yatima-inc/RustFFI.lean/blob/2a397cbc0904e2d575862c4067b512b6cc6b65f8/lakefile.lean
  let srcFileName := "ggmlffi.c"
  let oFilePath := pkg.oleanDir / "libggmlffi.o"
  let srcJob ← inputFile srcFileName
  buildFileAfterDep oFilePath srcJob fun srcFile => do
    let flags := #["-I", (← getLeanIncludeDir).toString,
                   "-I", (pkg.dir / "ggml" / "include").toString,
                   "-fPIC"]
    compileO srcFileName oFilePath srcFile flags -- build static archive

#check IndexBuildM
#check buildStaticLib

target «ggmlffi-shim» (pkg : Package) : FilePath := do
  let name := nameToStaticLib "ggmlffishim"
  let ffiO ← fetch <| pkg.target ``ggmlffi
  buildStaticLib (pkg.buildDir / "lib" / name) #[ffiO]


target «ggml» (pkg : Package) : FilePath := do
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
  pure (BuildJob.pure tgtPath)
  -- let outPath := pkg.libDir / "libggmlplusffi.a"
  -- buildStaticLib outPath #[ggmlO, BuildJob.pure tgtPath]

meta if get_config? env = some "dev" then -- dev is so not everyone has to build it
  require «doc-gen4» from git "https://github.com/leanprover/doc-gen4" @ "main"
