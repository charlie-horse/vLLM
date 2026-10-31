# Nix packaging

This repo is a flake (`flake.nix` at the root, code here). It builds the fork
for V100 (sm_70, CUDA 12.8) and provides the NixOS module that runs it.

## Outputs

- `packages.x86_64-linux.vllm` (`default`): this repo built in place for
  sm_70 on the pinned wheel env, wrapped with the runtime setup (`vllm`,
  `python`).
- `nixosModules.vllm` (`default`): `services.vllm-sm70` (`vllm`,
  `vllm-model-fetch`, the model table). Imports `nixosModules.lmcache`.
- `nixosModules.lmcache`: `services.lmcache-sm70` (`vllm-lmcache`,
  `vllm-lmcache-ui`, `vllm-lmcache-l2-prune`). The vllm module enables it
  when the active model has `lmcache = true`, sets the chunk size and runs it
  as the vllm user (KV moves over CUDA IPC).

Both modules build their packages with the host's `pkgs`, once
(`packages-module.nix`).

## Files

- `source.nix`: the repo minus `nix/`, `flake.nix` and `flake.lock`. Commits
  that only touch those don't rebuild the ~25 min CUDA tree.
- `flint.nix`: the in-place sm_70 build, the pre-fetched CMake deps, the
  weights fetches and the conversion. The converter has its own fetched pin
  (`converterFork`) so the 23 GB checkpoint keeps its store path; bump it only
  when the converter or the checkpoint format changes.
- `package.nix`: the uv2nix Python env (`pyenv/`) and the wrappers.
- `module.nix`, `lmcache.nix`: the service modules.
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
  services.lmcache-sm70.ui.firewallInterfaces = [ "tailscale0" ];
}
```

Deploy a new fork commit with `nix flake update vllm-fork` on the host, then
build and switch. `flake.lock` here pins the same nixpkgs, pyproject-nix,
uv2nix and build-system-pkgs revisions as the host.
