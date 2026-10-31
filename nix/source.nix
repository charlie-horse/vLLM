# The fork's source tree for the sm_70 build: the whole repo minus the Nix
# packaging, so a change under nix/ (or to flake.nix/flake.lock) doesn't
# rebuild the ~25 min CUDA tree. Content-addressed like a fetchFromGitHub
# "source", so the same commit gives the same store path either way.
{ lib }:
let
  root = ../.;
  nixOnly = lib.fileset.unions (
    map lib.fileset.maybeMissing [
      ../nix
      ../flake.nix
      ../flake.lock
    ]
  );
in
lib.fileset.toSource {
  inherit root;
  fileset = lib.fileset.difference root nixOnly;
}
