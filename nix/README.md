# Nix packaging

This repo is a flake (`flake.nix` at the root, code here). It builds the fork
for V100 (sm_70, CUDA 12.8) and provides the NixOS module that runs it.

## Outputs (`x86_64-linux`)

- `packages.vllm-flint` (`default`): this repo built in place, wrapped with the
  runtime setup (`vllm`, `python`). `vllm-flint-tree` is the unwrapped build.
- `packages.vllm-wheel`: the stock 1Cat 1.5.1 wheel env (`vllm`, `hf`,
  `lmcache`); `lmcache-ui`: a `python` for LMCache's dashboard;
  `python-env`: the uv2nix venv from `pyenv/`.
- `packages.flint-gguf`, `flint-hf-config`, `flint-converter`,
  `flint-checkpoint`: Flint's weights and their GGUF -> ModelOpt conversion.
- `nixosModules.default`: `services.vllm-sm70` (`vllm`, `vllm-lmcache`,
  `vllm-lmcache-ui`, `vllm-model-fetch`, `vllm-lmcache-l2-prune`).
- `checks.module-eval`: evaluates the module's units without building.

## Files

- `source.nix`: the repo minus `nix/`, `flake.nix` and `flake.lock`. Commits
  that only touch those don't rebuild the ~25 min CUDA tree.
- `flint.nix`: the in-place sm_70 build, the pre-fetched CMake deps, the
  weights fetches and the conversion. The converter has its own fetched pin
  (`converterFork`) so the 23 GB checkpoint keeps its store path; bump it only
  when the converter or the checkpoint format changes.
- `package.nix`: the uv2nix Python env (`pyenv/`) and the wrappers.
- `module.nix`: the service module.
- `pyenv/`: `pyproject.toml` + `uv.lock` for the env (torch 2.10+cu128, 1Cat
  1.5.1 wheel, LMCache 0.5.6 from source). Re-lock with
  `uv lock --python 3.12` in this directory.

## Host use

```nix
# flake.nix
inputs.vllm-fork.url = "github:charlie-horse/vLLM";
inputs.vllm-fork.inputs.nixpkgs.follows = "nixpkgs";

# a host module
{ inputs, ... }: {
  imports = [ inputs.vllm-fork.nixosModules.default ];
  services.vllm-sm70 = {
    enable = true;
    activeModel = "flint";
    firewallInterfaces = [ "eno1" "tailscale0" ];
  };
}
```

Deploy a new fork commit with `nix flake update vllm-fork` on the host, then
build and switch. `flake.lock` here pins the same nixpkgs, pyproject-nix,
uv2nix and build-system-pkgs revisions as the host.
