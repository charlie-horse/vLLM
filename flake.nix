{
  description = "1Cat-vLLM for V100 (sm_70): Nix build and NixOS service";

  # Locked to the same revisions as the c4130 host flake, so the vLLM closure
  # doesn't move when the host consumes this flake. The host should still set
  # `vllm-fork.inputs.nixpkgs.follows = "nixpkgs"`.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

    pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix.url = "github:pyproject-nix/uv2nix";
    uv2nix.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix.inputs.pyproject-nix.follows = "pyproject-nix";
    pyproject-build-systems.url = "github:pyproject-nix/build-system-pkgs";
    pyproject-build-systems.inputs.nixpkgs.follows = "nixpkgs";
    pyproject-build-systems.inputs.pyproject-nix.follows = "pyproject-nix";
    pyproject-build-systems.inputs.uv2nix.follows = "uv2nix";
  };

  outputs =
    inputs@{ nixpkgs, ... }:
    let
      system = "x86_64-linux";
      inherit (nixpkgs) lib;
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      src = import ./nix/source.nix { inherit lib; };
      p = import ./nix/package.nix { inherit pkgs lib inputs src; };
      vllm = import ./nix/module.nix { inherit inputs src; };
    in
    {
      # This repo built for sm_70 (CUDA 12.8) on the pinned wheel env, wrapped
      # with the runtime setup (`vllm`, `python`).
      packages.${system} = {
        default = p.vllmFlint;
        vllm = p.vllmFlint;
      };

      # `vllm` imports `lmcache`; use `lmcache` alone to run only the cache
      # server. Both build their packages with the host's `pkgs`.
      nixosModules = {
        default = vllm;
        inherit vllm;
        lmcache = import ./nix/lmcache.nix { inherit inputs src; };
      };
    };
}
