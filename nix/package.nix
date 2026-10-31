# 1Cat-vLLM (V100 / sm_70 fork) as a Nix-built Python environment.
#
# Python deps are pinned by ./pyenv/uv.lock (torch 2.10.0+cu128, 1Cat-vLLM
# 1.5.1 SM70 release wheel) and built with uv2nix. Wheels are used as-is;
# native libraries are found through the wrapper's LD_LIBRARY_PATH.
#
# Returns { vllm, vllmFlint, lmcacheUi, flintCheckpoint, ... }: `vllm` wraps
# the stock wheel; `vllmFlint` is the same venv with this repo's sm_70 build
# (./flint.nix) first on PYTHONPATH; `lmcacheUi` is a python for LMCache's
# dashboard; `flintCheckpoint` is Flint's converted weights.
#
# `inputs` are the flake inputs (pyproject-nix, uv2nix, pyproject-build-systems);
# `src` is the serving tree's source (./source.nix).
{ pkgs, lib, inputs, src }:
let
  inherit (inputs) pyproject-nix uv2nix pyproject-build-systems;

  python = pkgs.python312;
  cuda = pkgs.cudaPackages_12_8;

  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./pyenv; };

  # Wheels reference optional/system libs (libcuda, MPI/UCX/libfabric
  # transports in nvshmem, ...). Don't fail autoPatchelf on them; whatever is
  # actually needed at runtime comes from the wrapper's LD_LIBRARY_PATH.
  lockedNames = map (p: p.name) (lib.importTOML ./pyenv/uv.lock).package;
  ignoreMissingLibs =
    final: prev:
    lib.genAttrs (lib.filter (n: n != "c4130-vllm-env" && prev ? ${n}) lockedNames) (
      n: prev.${n}.overrideAttrs (_: { autoPatchelfIgnoreMissingDeps = true; })
    );

  # LMCache has no CUDA 12 wheels, so build it from the pinned git source:
  # CUDA 12.8 for sm_70, against this set's torch 2.10 (not the torch==2.13 its
  # pyproject build-system pins for release wheels).
  lmcacheFromSource = final: prev: {
    lmcache = (prev.lmcache.override { sourcePreference = "sdist"; }).overrideAttrs (old: {
      nativeBuildInputs =
        (old.nativeBuildInputs or [ ])
        ++ final.resolveBuildSystem {
          setuptools = [ ];
          setuptools-scm = [ ];
          wheel = [ ];
          packaging = [ ];
          # Imported by setup.py for torch.utils.cpp_extension (with its deps).
          torch = [ ];
          ninja = [ ];
          grpcio-tools = [ ];
        }
        ++ [ cuda.cudatoolkit ];
      env = (old.env or { }) // {
        LMCACHE_CUDA_MAJOR = "12";
        TORCH_CUDA_ARCH_LIST = "7.0";
        CUDA_HOME = "${cuda.cudatoolkit}";
        NVCC_PREPEND_FLAGS = "-I${cuda.cudatoolkit}/include";
        CC = "${cuda.backendStdenv.cc}/bin/cc";
        CXX = "${cuda.backendStdenv.cc}/bin/c++";
        SETUPTOOLS_SCM_PRETEND_VERSION = "0.5.6";
        MAX_JOBS = "16";
      };
      # torch's CUDA libs ship in the nvidia-* wheels; the build imports torch.
      preBuild = (old.preBuild or "") + ''
        export LD_LIBRARY_PATH="$(find -L ${
          lib.concatMapStringsSep " " (n: "${final.${n}}/${python.sitePackages}") (
            lib.filter (lib.hasPrefix "nvidia-") (lib.attrNames final)
          )
        } -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | paste -sd:)''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      '';
    });
  };

  pythonSet =
    (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope
      (lib.composeManyExtensions [
        pyproject-build-systems.overlays.wheel
        (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
        ignoreMissingLibs
        lmcacheFromSource
      ]);

  venv = (pythonSet.mkVirtualEnv "c4130-vllm-env" workspace.deps.default).overrideAttrs (_: {
    # flashinfer-python and torch-c-dlpack-ext both ship a stray top-level
    # build_backend.py (packaging leftover, never imported at runtime).
    venvIgnoreCollisions = [ "lib/python3.12/site-packages/build_backend.py" ];
  });

  # TileLang / Triton / FlashInfer JIT-compile kernels at runtime and need a
  # CUDA 12.8 nvcc (CUDA 13 cannot target sm_70) plus a compatible host gcc.
  # nvcc's own profile only points at the cuda_nvcc include dir, so the
  # wrapper adds the merged toolkit's headers (cuda_runtime.h etc.).
  jitTools = [
    cuda.cudatoolkit
    cuda.backendStdenv.cc
    pkgs.ninja
    pkgs.coreutils
    pkgs.which
  ];

  hostLibs = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.libGL
    pkgs.glib
    pkgs.numactl
  ];

  # Wraps venv executables with the runtime setup. `extraWrapperArgs` lets an
  # env put a source tree in front of the venv (vllmFlint).
  mkWrapped =
    {
      name,
      exes,
      extraWrapperArgs ? "",
      passthru ? { },
    }:
    pkgs.runCommand name
      {
        nativeBuildInputs = [ pkgs.makeWrapper ];
        passthru = { inherit venv python pythonSet; } // passthru;
      }
      ''
        site=${venv}/${python.sitePackages}

        # Every shared-library dir shipped by the pip CUDA wheels (nvidia-*, torch),
        # plus the runtime libs TileLang links against (libtvm_ffi, libz3).
        libdirs=$(find -L "$site/nvidia" "$site/torch/lib" "$site/tvm_ffi/lib" "$site/z3/lib" \
          -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | paste -sd:)

        mkdir -p $out/bin
        for exe in ${toString exes}; do
          # A venv python finds its pyvenv.cfg through argv[0].
          argv0=()
          [ $exe = python ] && argv0=(--argv0 ${venv}/bin/python)
          makeWrapper ${venv}/bin/$exe $out/bin/$exe "''${argv0[@]}" \
            --set CUDA_HOME ${cuda.cudatoolkit} \
            --set CUDA_PATH ${cuda.cudatoolkit} \
            --set NVCC_PREPEND_FLAGS "-I${cuda.cudatoolkit}/include" \
            --set TRITON_LIBCUDA_PATH /run/opengl-driver/lib \
            --set CC ${cuda.backendStdenv.cc}/bin/cc \
            --set CXX ${cuda.backendStdenv.cc}/bin/c++ \
            --prefix PATH : ${lib.makeBinPath jitTools} \
            --prefix LD_LIBRARY_PATH : "/run/opengl-driver/lib:$libdirs:${hostLibs}" \
            ${extraWrapperArgs}
        done
      '';

  # Stock 1Cat 1.5.1 release wheel (the Qwen models).
  vllm = mkWrapped {
    name = "vllm-c4130";
    exes = [ "vllm" "hf" "lmcache" ];
  };

  # LMCache's dashboard is started as `python -m lmcache.lmcache_frontend.app`.
  # The app itself is only FastAPI, but importing `lmcache` pulls in torch, so it
  # needs the same runtime setup. Separate from `vllm` so a bare `python` doesn't
  # land on the system PATH through environment.systemPackages.
  lmcacheUi = mkWrapped {
    name = "lmcache-ui-c4130";
    exes = [ "python" ];
  };

  # This repo's sm_70 build (Flint); built in ./flint.nix.
  flint = import ./flint.nix { inherit pkgs lib python venv cuda pythonSet src; };
  wrapFlint =
    tree:
    mkWrapped {
      name = "vllm-flint-c4130";
      exes = [ "vllm" "python" ];
      extraWrapperArgs = "--prefix PYTHONPATH : ${tree}:${tree}/flash-attention-v100";
      passthru = { inherit tree; };
    };
  # Serving env (this repo) and the converter env (flint.nix `converterFork`),
  # which only builds the checkpoint.
  vllmFlint = wrapFlint flint.tree;
  flintConverter = wrapFlint flint.converterTree;
in
{
  inherit vllm vllmFlint lmcacheUi flintConverter venv;
  inherit (flint) gguf hfConfig;
  flintTree = flint.tree;
  flintCheckpoint = flint.mkCheckpoint flintConverter;
}
