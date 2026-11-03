# The fork's source tree for the sm_70 build: the whole repo minus the Nix
# packaging, .claude/ and .gitignore, so a change under nix/ (or to flake.nix/flake.lock,
# agent settings or ignore rules) doesn't rebuild the ~25 min CUDA tree. Content-addressed
# like a fetchFromGitHub "source", so the same content gives the same store
# path either way.
{ lib }:
let
  root = ../.;
  nixOnly = lib.fileset.unions (
    map lib.fileset.maybeMissing [
      ../nix
      ../flake.nix
      ../flake.lock
      # Agent/editor settings and VCS metadata, not part of the build.
      ../.claude
      ../.gitignore
    ]
  );
in
lib.fileset.toSource {
  inherit root;
  fileset = lib.fileset.difference root nixOnly;
}
