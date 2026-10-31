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
    inputs@{ self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      inherit (nixpkgs) lib;
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      src = import ./nix/source.nix { inherit lib; };
      p = import ./nix/package.nix { inherit pkgs lib inputs src; };

      module = import ./nix/module.nix { inherit inputs src; };

      # Evaluates the module on a minimal system without building anything.
      evalModule = lib.nixosSystem {
        inherit system;
        modules = [
          module
          {
            nixpkgs.config.allowUnfree = true;
            boot.loader.grub.enable = false;
            fileSystems."/".device = "nodev";
            system.stateVersion = "26.05";
            services.vllm-sm70 = {
              enable = true;
              firewallInterfaces = [ "eth0" ];
            };
          }
        ];
      };
      units = lib.mapAttrs (_: u: u.serviceConfig.ExecStart or u.script) {
        inherit (evalModule.config.systemd.services)
          vllm
          vllm-lmcache
          vllm-lmcache-ui
          vllm-model-fetch
          ;
      };
    in
    {
      packages.${system} = {
        default = p.vllmFlint;
        # This repo built for sm_70, wrapped with the runtime setup.
        vllm-flint = p.vllmFlint;
        # The unwrapped in-place build of this repo.
        vllm-flint-tree = p.flintTree;
        # The stock 1Cat 1.5.1 wheel (the Qwen models), plus `hf` and `lmcache`.
        vllm-wheel = p.vllm;
        lmcache-ui = p.lmcacheUi;
        # The Python environment (torch, LMCache, 1Cat wheel) from nix/pyenv.
        python-env = p.venv;
        flint-gguf = p.gguf;
        flint-hf-config = p.hfConfig;
        flint-converter = p.flintConverter;
        flint-checkpoint = p.flintCheckpoint;
      };

      nixosModules.default = module;
      nixosModules.vllm-sm70 = module;

      checks.${system}.module-eval = pkgs.writeText "vllm-sm70-units" (
        builtins.unsafeDiscardStringContext (builtins.toJSON units)
      );
    };
}
