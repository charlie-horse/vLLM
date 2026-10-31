# vLLM inference service on 2x V100 (sm_70, TP=2), one model at a time.
#
# Host-agnostic: the host enables it and sets its own values, e.g.
#
#   services.vllm-sm70 = {
#     enable = true;
#     activeModel = "flint";
#     firewallInterfaces = [ "eno1" "tailscale0" ];
#   };
#
# Switch models by setting `activeModel` and running nixos-rebuild switch.
# `inputs` are this flake's inputs and `src` the serving tree's source; both
# are bound by flake.nix, packages are built with the host's `pkgs`.
{ inputs, src }:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.vllm-sm70;

  inherit (import ./package.nix { inherit pkgs lib inputs src; })
    vllm
    vllmFlint
    lmcacheUi
    flintCheckpoint
    ;

  inherit (cfg) port;
  # systemd StateDirectory of the vllm user (the units say "vllm").
  stateDir = "/var/lib/vllm";
  modelsDir = "${stateDir}/models";

  # KV cache tiers below the GPU: LMCache L1 in pinned CPU RAM, L2 on local disk.
  # LMCache doesn't evict L2 on its own (that needs its fleet coordinator), so
  # vllm-lmcache-l2-prune keeps the directory under l2SizeGb, least recently
  # accessed first; fs_native checks files on every lookup, so pruned chunks
  # are just misses.
  kvTiers = {
    inherit (cfg.kvTiers) l1SizeGb l2SizeGb;
    l2Dir = "/var/lib/lmcache/l2";
    port = 5555;
    # The MP server's admin API (no authentication, so loopback only) and the
    # dashboard that renders it.
    httpPort = 8080;
    uiPort = 8081;
  };

  # DFlash2 speculative decoding with a local draft checkpoint, per 1Cat's
  # release profile (draft KV follows the target's FP16 KV).
  dflashConfig =
    draftDir:
    builtins.toJSON {
      method = "dflash";
      model = draftDir;
      num_speculative_tokens = 7;
      kv_cache_dtype = "auto";
      attention_backend = "FLASH_ATTN_V100";
      draft_sample_method = "probabilistic";
      enforce_eager = false;
    };

  # Built-in models (see the `models` option for the fields).
  builtinModels = {
    # Flint NVFP4 on this fork, per scripts/serve_flint_nvfp4_v100_pcie.sh.
    # Weights are converted from the GGUF at build time (./flint.nix).
    # LMCache's MP connector supports the hybrid (sliding-window + full
    # attention) KV cache manager; its ~2.2 GiB per GPU fits under the 0.90
    # utilization below.
    flint = {
      path = flintCheckpoint;
      package = vllmFlint;
      lmcache = true;
      lmcacheChunkSize = 256;
      asyncScheduling = true;
      maxNumSeqs = 2;
      gpuMemoryUtilization = 0.90;
      env = {
        CUDA_DEVICE_ORDER = "PCI_BUS_ID";
        # Run the real CUDA IPC peer-write test; PCIe pairs that report P2P
        # but drop peer stores otherwise hang at the first decode.
        VLLM_SKIP_P2P_CHECK = "0";
        # No PYTORCH_ALLOC_CONF=expandable_segments: its CUDA IPC goes through
        # pidfd_getfd, which kernel.yama.ptrace_scope=1 denies to the LMCache
        # server (not an ancestor of vLLM), so register_kv_caches times out.
      };
      args = [
        "--kv-cache-dtype" "auto"
        # TurboMind NVFP4 MoE route: ~40% faster decode than the skinny route.
        "--kernel-config.sm70_skinny_moe" "false"
        "--enable-prompt-tokens-details"
        "--tool-call-parser" "glm47"
        "--reasoning-parser" "glm47"
        "--exclude-tools-when-tool-choice-none"
        "--override-generation-config" ''{"temperature":1.0,"top_p":0.95,"top_k":20}''
      ];
    };

    "qwen3.6-35b-a3b" =
      let
        draftDir = "${modelsDir}/Qwen3.6-35B-A3B-DFlash2";
      in
      {
        downloads = [
          {
            repo = "Qwen/Qwen3.6-35B-A3B-FP8";
            revision = "95a723d08a9490559dae23d0cff1d9466213d989";
            dir = "${modelsDir}/Qwen3.6-35B-A3B-FP8";
          }
          {
            repo = "incoai/Qwen3.6-35B-A3B-DFlash2";
            revision = "51ef7b6923ad6c14cb1bb41c37a9041446496ab6";
            dir = draftDir;
          }
        ];
        # vLLM's attention block size for this model (aligned to the GDN page).
        lmcacheChunkSize = 1056;
        args = [
          "--tool-call-parser" "qwen3_coder"
          "--reasoning-parser" "qwen3"
          "--language-model-only"
          "--mamba-cache-mode" "align"
          "--speculative-config" (dflashConfig draftDir)
        ];
      };

    "qwen3.8-27b" =
      let
        draftDir = "${modelsDir}/Qwen3.8-27B-DFlash2";
      in
      {
        downloads = [
          {
            repo = "Qwen/Qwen3.8-27B-FP8";
            revision = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a";
            dir = "${modelsDir}/Qwen3.8-27B-FP8";
          }
          {
            repo = "incoai/Qwen3.8-27B-DFlash2";
            revision = "dedf8df68adfb1afeaf7b7480c0a0243108177b4";
            dir = draftDir;
          }
        ];
        maxModelLen = 150000;
        gpuMemoryUtilization = 0.91;
        # Must be a multiple of vLLM's block size, which --mamba-block-size sets.
        lmcacheChunkSize = 8192;
        args = [
          "--tool-call-parser" "qwen3_coder"
          "--reasoning-parser" "qwen3"
          "--language-model-only"
          "--attention-backend" "FLASH_ATTN_V100"
          "--block-size" "2048"
          "--mamba-block-size" "8192"
          "--mamba-cache-mode" "align"
          "--seed" "0"
          "--trust-remote-code"
          "--speculative-config" (dflashConfig draftDir)
        ];
      };
  };

  active = cfg.activeModel;
  model = cfg.models.${active};
  marker = d: "${d.dir}/.complete-${d.revision}";
  downloads = model.downloads or [ ];
  modelPath = model.path or (lib.head downloads).dir;
  package = model.package or vllm;
  lmcache = model.lmcache or true;

  serveArgs = [
    modelPath
    "--served-model-name" active
    "--host" "0.0.0.0"
    "--port" (toString port)
    "--tensor-parallel-size" "2"
    "--dtype" "half"
    "--max-model-len" (toString (model.maxModelLen or 262144))
    "--max-num-seqs" (toString (model.maxNumSeqs or 4))
    "--max-num-batched-tokens" "8192"
    # Leaves ~2.2 GiB per GPU outside vLLM's budget for the LMCache server's
    # CUDA context and its KV transfer staging during stores.
    "--gpu-memory-utilization" (toString (model.gpuMemoryUtilization or 0.88))
  ]
  ++ lib.optional (!(model.asyncScheduling or false)) "--no-async-scheduling"
  ++ [
    "--enable-prefix-caching"
    "--enable-auto-tool-choice"
  ]
  ++ lib.optionals lmcache [
    # LMCache's own MP connector; the copy bundled in 1Cat targets an older
    # LMCache API. Failed loads fall back to recomputing the prefix.
    "--kv-transfer-config"
    (builtins.toJSON {
      kv_connector = "LMCacheMPConnector";
      kv_connector_module_path = "lmcache.integration.vllm.lmcache_mp_connector";
      kv_role = "kv_both";
      kv_load_failure_policy = "recompute";
      kv_connector_extra_config = {
        "lmcache.mp.host" = "tcp://127.0.0.1";
        "lmcache.mp.port" = kvTiers.port;
      };
    })
  ]
  ++ model.args;

  lmcacheArgs = [
    "server"
    "--host" "127.0.0.1"
    "--port" (toString kvTiers.port)
    "--http-host" "127.0.0.1"
    # Already the default, but vllm-lmcache-ui points at this port.
    "--http-port" (toString kvTiers.httpPort)
    # One object group per attention type, required for hybrid GDN models.
    "--separate-object-groups"
    # Must be a multiple of the active model's vLLM block size.
    "--chunk-size" (toString model.lmcacheChunkSize)
    "--l1-size-gb" (toString kvTiers.l1SizeGb)
    "--eviction-policy" "LRU"
    "--l2-adapter"
    (builtins.toJSON {
      type = "fs_native";
      base_path = kvTiers.l2Dir;
      num_workers = 8;
      max_capacity_gb = kvTiers.l2SizeGb;
    })
  ];

  # LMCache's dashboard is a read-only proxy in front of the MP server's admin
  # API: `--nodes` is both the node list it renders and the allowlist its
  # /proxy/{host}/{port}/... route checks against.
  uiArgs = [
    "--host" "0.0.0.0"
    "--port" (toString kvTiers.uiPort)
    "--nodes"
    (builtins.toJSON [
      {
        name = "lmcache_server";
        host = "127.0.0.1";
        port = toString kvTiers.httpPort;
      }
    ])
    "--log-level" "warning"
  ];

  env = {
    HOME = stateDir;
    HF_HOME = "${stateDir}/hf";
    XDG_CACHE_HOME = "${stateDir}/cache";
    VLLM_CACHE_ROOT = "${stateDir}/cache/vllm";
    TRITON_CACHE_DIR = "${stateDir}/cache/triton";
    TILELANG_CACHE_DIR = "${stateDir}/cache/tilelang";
    VLLM_NO_USAGE_STATS = "1";
    DO_NOT_TRACK = "1";
  };
in
{
  imports = [
    # The options lived under c4130.vllm while this module was host config.
    (lib.mkRenamedOptionModule
      [ "c4130" "vllm" "activeModel" ]
      [ "services" "vllm-sm70" "activeModel" ]
    )
    (lib.mkRenamedOptionModule
      [ "c4130" "vllm" "kvTiers" "l1SizeGb" ]
      [ "services" "vllm-sm70" "kvTiers" "l1SizeGb" ]
    )
    (lib.mkRenamedOptionModule
      [ "c4130" "vllm" "kvTiers" "l2SizeGb" ]
      [ "services" "vllm-sm70" "kvTiers" "l2SizeGb" ]
    )
  ];

  options.services.vllm-sm70 = {
    enable = lib.mkEnableOption "the vLLM server on two V100s (TP=2) with LMCache";

    activeModel = lib.mkOption {
      type = lib.types.enum (lib.attrNames cfg.models);
      default = "flint";
      description = "Model served by vllm.service. Only one fits on the GPUs at a time.";
    };

    # Served model name -> weights, either pinned HF `downloads` (a list of
    # { repo, revision, dir }; the first is the served checkpoint) or a Nix
    # store `path`; optional maxModelLen / gpuMemoryUtilization / maxNumSeqs
    # overrides; `lmcache` (default true) and its `lmcacheChunkSize`; `package`
    # (the wrapped vllm to run, default the stock wheel); extra service `env`;
    # `asyncScheduling` (default false); and model-specific `vllm serve` `args`.
    models = lib.mkOption {
      type = lib.types.attrsOf (lib.types.attrsOf lib.types.raw);
      description = ''
        Models vllm.service can serve. Built-in: flint, qwen3.6-35b-a3b and
        qwen3.8-27b. Add entries, or override single fields of a built-in one.
      '';
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "Port of the OpenAI-compatible API (no authentication).";
    };

    firewallInterfaces = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "eno1" "tailscale0" ];
      description = ''
        Interfaces on which the API port (and the LMCache dashboard, when
        LMCache is on) is opened. Neither authenticates.
      '';
    };

    # Lower both to run LMCache alongside something else that wants the RAM or
    # the disk; L1 is pinned and allocated in full at startup.
    kvTiers.l1SizeGb = lib.mkOption {
      type = lib.types.ints.positive;
      default = 64;
      description = "LMCache L1 size in GB (pinned CPU RAM).";
    };

    kvTiers.l2SizeGb = lib.mkOption {
      type = lib.types.ints.positive;
      default = 180;
      description = "LMCache L2 cap in GB (local disk, enforced by vllm-lmcache-l2-prune).";
    };
  };

  config = lib.mkIf cfg.enable {
    services.vllm-sm70.models = lib.mapAttrs (_: lib.mkDefault) builtinModels;

    environment.systemPackages = [ vllm ];

    hardware.nvidia.nvidiaPersistenced = true;

    users.users.vllm = {
      isSystemUser = true;
      group = "vllm";
      extraGroups = [ "video" ];
      home = stateDir;
    };
    users.groups.vllm = { };

    systemd.services.vllm-model-fetch = {
      description =
        if downloads == [ ] then "Check Nix-store weights for ${active}" else "Download weights for ${active}";
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      environment = env;
      # Runs only while some pinned revision is missing (triggering "|" conditions
      # are OR-ed), so boots with everything on disk don't depend on Hugging Face.
      # A model served from the Nix store has no downloads: its weights are in
      # the system closure, so the unit only checks they are there.
      unitConfig.ConditionPathExists =
        if downloads == [ ] then [ "!${modelPath}" ] else map (d: "|!${marker d}") downloads;
      script =
        if downloads == [ ] then
          ''
            echo "${modelPath} is missing" >&2
            exit 1
          ''
        else
          lib.concatMapStrings (d: ''
            if [ ! -e ${marker d} ]; then
              ${vllm}/bin/hf download ${d.repo} --revision ${d.revision} --local-dir ${d.dir}
              touch ${marker d}
            fi
          '') downloads;
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "vllm";
        Group = "vllm";
        StateDirectory = "vllm";
        TimeoutStartSec = "6h";
      };
    };

    # LMCache MP server: holds L1/L2 and copies KV to/from vLLM over CUDA IPC,
    # so it runs as the same user with GPU access.
    systemd.services.vllm-lmcache = lib.mkIf lmcache {
      description = "LMCache KV cache server (L1 ${toString kvTiers.l1SizeGb} GB RAM, L2 ${toString kvTiers.l2SizeGb} GB disk)";
      after = [ "nvidia-persistenced.service" ];
      environment = env;
      serviceConfig = {
        User = "vllm";
        Group = "vllm";
        SupplementaryGroups = [ "video" ];
        StateDirectory = [ "vllm" "lmcache" "lmcache/l2" ];
        ExecStart = "${vllm}/bin/lmcache ${lib.escapeShellArgs lmcacheArgs}";
        Restart = "on-failure";
        RestartSec = 10;
        LimitMEMLOCK = "infinity";
      };
    };

    # LMCache's own dashboard (lmcache.lmcache_frontend). It only talks HTTP to
    # the MP server's admin API, so it needs no GPU and no cache state of its own.
    systemd.services.vllm-lmcache-ui = lib.mkIf lmcache {
      description = "LMCache dashboard (port ${toString kvTiers.uiPort})";
      wantedBy = [ "multi-user.target" ];
      requires = [ "vllm-lmcache.service" ];
      after = [ "vllm-lmcache.service" ];
      environment = env;
      serviceConfig = {
        User = "vllm";
        Group = "vllm";
        # Importing lmcache probes for a compute backend.
        SupplementaryGroups = [ "video" ];
        StateDirectory = "vllm";
        ExecStart = "${lmcacheUi}/bin/python -m lmcache.lmcache_frontend.app ${lib.escapeShellArgs uiArgs}";
        Restart = "on-failure";
        RestartSec = 10;
      };
    };

    systemd.services.vllm-lmcache-l2-prune = {
      description = "Keep LMCache L2 under ${toString kvTiers.l2SizeGb} GB";
      path = [ pkgs.coreutils pkgs.findutils pkgs.gawk ];
      serviceConfig = {
        Type = "oneshot";
        User = "vllm";
        Group = "vllm";
      };
      # Above 90% of the cap, delete least recently accessed files down to 80%.
      script = ''
        dir=${kvTiers.l2Dir}
        [ -d "$dir" ] || exit 0
        cap=$(( ${toString kvTiers.l2SizeGb} * 1024 * 1024 * 1024 ))
        used=$(du -sb "$dir" | cut -f1)
        [ "$used" -gt $(( cap * 9 / 10 )) ] || exit 0
        excess=$(( used - cap * 8 / 10 ))
        find "$dir" -type f -printf '%A@ %s %p\n' | sort -n \
          | awk -v excess="$excess" 'freed >= excess { exit } { print $3; freed += $2 }' \
          | xargs -r -d '\n' rm -f --
      '';
    };

    systemd.timers.vllm-lmcache-l2-prune = {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnBootSec = "2min";
        OnUnitActiveSec = "1min";
      };
    };

    systemd.services.vllm = {
      description = "vLLM OpenAI-compatible server (${active})";
      wantedBy = [ "multi-user.target" ];
      requires = [ "vllm-model-fetch.service" ] ++ lib.optional lmcache "vllm-lmcache.service";
      after = [
        "vllm-model-fetch.service"
        "nvidia-persistenced.service"
      ] ++ lib.optional lmcache "vllm-lmcache.service";
      environment = env // { HF_HUB_OFFLINE = "1"; } // (model.env or { });
      serviceConfig = {
        User = "vllm";
        Group = "vllm";
        SupplementaryGroups = [ "video" ];
        StateDirectory = "vllm";
        ExecStart = "${package}/bin/vllm serve ${lib.escapeShellArgs serveArgs}";
        Restart = "on-failure";
        RestartSec = 30;
        TimeoutStopSec = 60;
        LimitMEMLOCK = "infinity";
      };
    };

    # Neither port authenticates; the host relies on network policy.
    networking.firewall.interfaces = lib.genAttrs cfg.firewallInterfaces (_: {
      allowedTCPPorts = [ port ] ++ lib.optional lmcache kvTiers.uiPort;
    });
  };
}
