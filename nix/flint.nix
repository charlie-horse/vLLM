# Flint (RecursiveSI/flint, GLM-style MoE) on the 1Cat-vLLM source fork.
#
#   tree        this repo (the flake's own source, minus nix/), built in place
#               (`setup.py build_ext --inplace`) for sm_70 with CUDA 12.8
#               against the wheel env's torch. package.nix wraps it as
#               `vllmFlint`: the stock venv with this tree first on PYTHONPATH,
#               so `vllm`, `flash_attn_v100` and `flash_qla` come from the
#               fork. The fork's runtime deps must stay equal to the 1.5.1
#               wheel's, which ./pyenv/uv.lock pins.
#   checkpoint  the NVFP4 GGUF converted to a ModelOpt W4A16_NVFP4 checkpoint
#               by the fork's converter (CPU only), served from the store. The
#               converter has its own pin (`converterFork`), so a change to
#               this repo rebuilds only the serving tree, not the 23 GB
#               checkpoint.
{
  pkgs,
  lib,
  python,
  venv,
  cuda,
  pythonSet,
  # The serving tree's source (./source.nix).
  src,
}:
let
  # Converter pin: the tree whose vllm.models.flint.convert_gguf produced the
  # checkpoint. It is fetched, not taken from this repo, because the
  # checkpoint is input-addressed: any change to the converter's build inputs
  # gives it a new store path and reconverts (~4 min, a new 23 GB path). Only
  # bump it when the converter or the checkpoint format changes. To bump: set
  # rev, then `nix flake prefetch --json github:charlie-horse/1Cat-vLLM/<rev>`
  # and paste its "hash".
  converterFork = {
    rev = "f66728d9b95e44dd6526b0f2033832477e6a79de"; # claude/flint-v100-pcie
    hash = "sha256-PmWgtyrqJ6KMllN/NqYjhNgrwSypGtQ4xk1BP1ADQfk=";
  };

  # Weights pins (Hugging Face commits; neither repo is gated).
  ggufRepo = {
    repo = "RecursiveSI/flint-nvfp4-gguf";
    rev = "6ec43e19b565fc8d560e2d04985d7724f20673bf";
  };
  configRepo = {
    repo = "RecursiveSI/flint";
    rev = "2a247be0c0aec895ccb0ac7ca92229ec6df37288";
  };

  fetchFork =
    pin:
    pkgs.fetchFromGitHub {
      owner = "charlie-horse";
      repo = "1Cat-vLLM";
      inherit (pin) rev hash;
    };

  # CMake FetchContent can't clone in the sandbox: pre-fetch what the sm_70
  # build pulls (CUTLASS_REVISION, DEFAULT_TRITON_KERNELS_TAG and
  # VLLM_FLASH_ATTN_SM70_COMMIT in the fork's cmake files).
  cutlassSrc = pkgs.fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cutlass";
    rev = "v4.4.2";
    hash = "sha256-0q9Ad0Z6E/rO2PdM4uQc8H0E0qs9uKc3reHepiHhjEc=";
  };
  tritonSrc = pkgs.fetchFromGitHub {
    owner = "triton-lang";
    repo = "triton";
    rev = "v3.5.1";
    hash = "sha256-dyNRtS1qtU8C/iAf0Udt/1VgtKGSvng1+r2BtvT9RB4=";
  };
  flashAttnSrc = pkgs.fetchFromGitHub {
    owner = "zhinianqin";
    repo = "flash-attention-v100";
    rev = "c2eda5e6115b98c3ba4bfd181570668742eece22";
    hash = "sha256-PKF32PXYl6xsMFAG6+AgER0ABNFYNl5l3TpcaR9Dv+U=";
  };
  # Its csrc/cutlass submodule (fetched on its own: the repo also has a
  # composable_kernel gitlink without a .gitmodules entry).
  flashAttnCutlassSrc = pkgs.fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cutlass";
    rev = "62750a2b75c802660e4894434dc55e839f322277";
    hash = "sha256-eL64JYiFAqrn6OJDzijpMR4DbEB4El/8yT7++iCZBlE=";
  };
  # Applied by cmake/external_projects/vllm_flash_attn.cmake when it clones,
  # but not when VLLM_FLASH_ATTN_SRC_DIR is given, so apply them here (same
  # order). Keep in sync with that file when bumping the fork.
  flashAttnPatches = [
    "sm70_flash_attn_d256_pipeline"
    "sm70_flash_attn_d256_splitkv3"
    "sm70_flash_attn_d256_k_pingpong"
    "sm70_flash_attn_d256_gqa_arch"
    "sm70_flash_attn_d256_splitkv_n"
  ];

  site = "${venv}/${python.sitePackages}";

  mkTree =
    { src, version }:
    cuda.backendStdenv.mkDerivation {
      pname = "1cat-vllm-flint";
      inherit src version;

      nativeBuildInputs = [
        cuda.cudatoolkit
        pkgs.cmake
        pkgs.ninja
        pkgs.patchelf
        pkgs.git
        pkgs.which
      ];
      dontUseCmakeConfigure = true;
      dontUseNinjaBuild = true;
      dontUseNinjaInstall = true;
      dontUseNinjaCheck = true;
      dontConfigure = true;
      # Keep the tree exactly as an in-place build leaves it.
      dontStrip = true;
      dontPatchELF = true;
      dontPatchShebangs = true;

      env = {
        TORCH_CUDA_ARCH_LIST = "7.0";
        CMAKE_CUDA_ARCHITECTURES = "70";
        FLASH_ATTN_V100_CUDA_ARCH_LIST = "7.0";
        VLLM_TARGET_DEVICE = "cuda";
        CMAKE_BUILD_TYPE = "Release";
        NVCC_THREADS = "1";
        SETUPTOOLS_SCM_PRETEND_VERSION = "1.5.1.dev0+flint";
        CUDA_HOME = "${cuda.cudatoolkit}";
        CUDA_PATH = "${cuda.cudatoolkit}";
        NVCC_PREPEND_FLAGS = "-I${cuda.cudatoolkit}/include";
        # The Nix ld wrapper adds its own RPATH entries, so CMake's install-time
        # RPATH rewrite fails; skip CMake's RPATH handling (an unsandboxed build
        # ends up with just the wrapper's glibc/gcc/CUDA entries too).
        CMAKE_ARGS = lib.concatStringsSep " " [
          "-DPython3_EXECUTABLE=${venv}/bin/python3.12"
          "-DPython_EXECUTABLE=${venv}/bin/python3.12"
          "-DCMAKE_SKIP_RPATH=ON"
        ];
        # setup.py imports setuptools_rust (the Rust frontend itself is optional
        # and not built by build_ext).
        PYTHONPATH = lib.concatMapStringsSep ":" (p: "${p}/${python.sitePackages}") [
          pythonSet.setuptools-rust
          pythonSet.semantic-version
        ];
      };

      buildPhase = ''
        runHook preBuild
        export HOME=$TMPDIR
        # torch's CUDA libs ship in the nvidia-* wheels; setup.py imports torch.
        export LD_LIBRARY_PATH="$(find -L ${site}/nvidia ${site}/torch/lib \
          -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | paste -sd:)"

        deps=$TMPDIR/deps
        mkdir -p $deps
        cp -r --no-preserve=mode ${cutlassSrc} $deps/cutlass
        cp -r --no-preserve=mode ${tritonSrc} $deps/triton
        cp -r --no-preserve=mode ${flashAttnSrc} $deps/flash-attn
        rm -rf $deps/flash-attn/csrc/cutlass
        cp -r --no-preserve=mode ${flashAttnCutlassSrc} $deps/flash-attn/csrc/cutlass
        for p in ${toString flashAttnPatches}; do
          patch -d $deps/flash-attn --batch --forward -p1 -l -i $PWD/cmake/patches/$p.patch
        done
        export VLLM_CUTLASS_SRC_DIR=$deps/cutlass
        export TRITON_KERNELS_SRC_DIR=$deps/triton/python/triton_kernels/triton_kernels
        export VLLM_FLASH_ATTN_SRC_DIR=$deps/flash-attn

        # nvcc on CUTLASS kernels needs a few GB per job; cap at 16.
        export MAX_JOBS=$(( NIX_BUILD_CORES > 16 ? 16 : NIX_BUILD_CORES ))
        ${venv}/bin/python setup.py build_ext --inplace
        runHook postBuild
      '';

      installPhase = ''
        runHook preInstall
        rm -rf build flash-attention-v100/build .deps
        mkdir -p $out
        cp -r . $out/
        # Bytecode in the read-only store (hash-checked, so mtimes don't matter).
        ${venv}/bin/python -m compileall -q -j $NIX_BUILD_CORES \
          --invalidation-mode unchecked-hash \
          $out/vllm $out/flash_qla $out/flash-attention-v100/flash_attn_v100 || true
        runHook postInstall
      '';

      passthru = { inherit src; };
    };

  hf =
    repo: rev: file:
    "https://huggingface.co/${repo}/resolve/${rev}/${file}";

  gguf = pkgs.fetchurl {
    name = "flint-nvfp4.gguf";
    url = hf ggufRepo.repo ggufRepo.rev "flint-nvfp4.gguf";
    sha256 = "97cc8009ff9b132d5cf3c488e6767962bb5a5dc158a90ed661bbbddf876b8bde";
  };

  # The config/tokenizer files convert_gguf copies (its _EXTRA_FILES that
  # exist in the repo, plus config.json).
  hfConfig = pkgs.linkFarm "flint-hf-config" (
    lib.mapAttrsToList
      (name: sha256: {
        inherit name;
        path = pkgs.fetchurl {
          url = hf configRepo.repo configRepo.rev name;
          inherit sha256;
        };
      })
      {
        "config.json" = "a57c33d11f725099105b657043c113149763066d204fd472d0b2ee399abc2ba7";
        "generation_config.json" = "a499c71895e008abd572558a5ad5904feaed92cf2a3b6e2cb0e9cabf03b96ed3";
        "tokenizer.json" = "a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68";
        "tokenizer_config.json" = "159ac88217e90ffd0a4a294726048bb302829e82e1958dc85bb203aa79138fbf";
        "chat_template.jinja" = "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466";
        "LICENSE" = "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1";
      }
  );
in
{
  inherit gguf hfConfig;
  # No commit in the version: a commit that only touches nix/ leaves the
  # source, and so the build, unchanged. The deployed commit is the host's
  # `vllm-fork` lock entry.
  tree = mkTree {
    inherit src;
    version = "1.5.1.dev0+flint";
  };
  converterTree = mkTree {
    src = fetchFork converterFork;
    version = "1.5.1.dev0+flint.${builtins.substring 0 7 converterFork.rev}";
  };

  # ~23 GB, ~2 min on 14 cores, ~25 GB peak RAM.
  # Takes the wrapped converterTree (package.nix). Input-addressed: its store
  # path changes whenever converterTree's derivation does.
  mkCheckpoint =
    vllmFlint:
    pkgs.runCommand "flint-nvfp4-modelopt"
      {
        passthru = { inherit gguf hfConfig; };
      }
      ''
        export HOME=$TMPDIR HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
        ${vllmFlint}/bin/python -m vllm.models.flint.convert_gguf \
          ${gguf} $out --hf-config ${hfConfig} --device cpu
        # Run metadata (duration, absolute paths): not reproducible.
        rm -f $out/conversion_report.json
      '';
}
