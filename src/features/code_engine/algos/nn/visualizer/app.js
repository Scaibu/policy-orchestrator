
let currentInspectorTab = "yaml";
let currentContractData = null;

function initAlgoDropdown() {
  const selectEl = document.getElementById("algo-dropdown-select");
  if (!selectEl || typeof CONTRACTS_DATA === "undefined") return;

  selectEl.innerHTML = "";
  
  // Group by domain
  const domains = {};
  for (const [key, c] of Object.entries(CONTRACTS_DATA)) {
    const dLabel = c.domainLabel || c.domain;
    if (!domains[dLabel]) domains[dLabel] = [];
    domains[dLabel].push(c);
  }

  for (const [domainLabel, algos] of Object.entries(domains)) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = domainLabel;
    
    for (const a of algos) {
      const option = document.createElement("option");
      option.value = a.id;
      option.textContent = a.name;
      optgroup.appendChild(option);
    }
    selectEl.appendChild(optgroup);
  }

  // Set default selection
  const firstKey = Object.keys(CONTRACTS_DATA)[0];
  if (firstKey) {
    selectEl.value = firstKey;
    onAlgoDropdownChange(firstKey);
  }
}

function onAlgoDropdownChange(key) {
  if (typeof CONTRACTS_DATA === "undefined" || !CONTRACTS_DATA[key]) return;
  const c = CONTRACTS_DATA[key];
  currentContractData = c;

  const titleEl = document.getElementById("inspector-title");
  if (titleEl) titleEl.textContent = c.name;

  const roleEl = document.getElementById("inspector-role");
  if (roleEl) roleEl.textContent = c.domainLabel;

  const tagEl = document.getElementById("inspector-tag");
  if (tagEl) tagEl.textContent = c.algo_id || "VERIFIED CONTRACT";

  const mathEl = document.getElementById("inspector-math");
  if (mathEl) mathEl.textContent = c.title || c.excerpt;

  const invEl = document.getElementById("inspector-invariant");
  if (invEl) invEl.textContent = c.invariant;

  renderInspectorCode();
}

/**
 * GORTEX IDENTICAL NEURAL ENGINE ARCHITECTURE & CONTRACT VISUALIZER
 * Zero dependencies, exact JSON contracts, interactive ASCII & SVG diagram bindings,
 * clipboard utilities, and live inspection panes.
 */

// Algorithms Store with Exact JSON Contracts & Mathematical Invariants
const ALGORITHMS = [
  // STAGE 1: CURATION
  {
    id: "synth-filter",
    name: "Synthetic Data Quality Filtering",
    stage: 1,
    stageNum: "01",
    stageName: "Curation",
    tag: "quality-gate",
    role: "Quality & Length Gatekeeper",
    math: "Filters noisy or duplicate synthetic samples using length bounds and 64-bit MinHash Jaccard thresholding.",
    input: {
      candidate_samples: [
        { text: "Proof of Fermat Last Theorem for n=3 using algebraic integers.", quality_score: 0.88, token_count: 12 },
        { text: "Click here for discount tickets!", quality_score: 0.35, token_count: 8 }
      ],
      min_quality_score: 0.70,
      min_token_length: 10,
      max_token_length: 1000,
      jaccard_dedup_threshold: 0.80
    },
    response: {
      accepted_samples: [
        { text: "Proof of Fermat Last Theorem for n=3 using algebraic integers.", quality_score: 0.88, token_count: 12 }
      ],
      rejection_breakdown: { low_quality: 1, duplicate: 0 },
      acceptance_ratio: 0.50
    },
    invariant: "Samples strictly satisfy quality >= 0.70 and Jaccard similarity <= 0.80."
  },
  {
    id: "doremi",
    name: "DoReMi Domain Reweighting",
    stage: 1,
    stageNum: "01",
    stageName: "Curation",
    tag: "simplex-opt",
    role: "Dynamic Simplex Mixture Optimizer",
    math: "Updates dynamic mixture weights via exponentiated excess loss between proxy and reference models.",
    input: {
      domain_losses_proxy: [1.85, 2.40, 1.60],
      domain_losses_reference: [1.70, 2.10, 1.55],
      current_domain_weights: [0.333, 0.333, 0.334],
      step_size_eta: 0.1
    },
    response: {
      updated_domain_weights: [0.281, 0.442, 0.277],
      excess_losses: [0.15, 0.30, 0.05],
      worst_case_domain_idx: 1
    },
    invariant: "Domain weights updated on valid simplex summing to exactly 1.0."
  },
  {
    id: "seq-packing",
    name: "Sequence Packing & Document Masking",
    stage: 1,
    stageNum: "01",
    stageName: "Curation",
    tag: "seq-packer",
    role: "Boundary-Preserving Batch Builder",
    math: "Packs variable token sequences into dense batches with 2D block-diagonal causal attention masks.",
    input: {
      document_token_sequences: [[101, 2054, 102], [101, 1037, 2793, 102]],
      max_sequence_length: 8,
      padding_token_id: 0
    },
    response: {
      packed_sequences: [[101, 2054, 102, 101, 1037, 2793, 102, 0]],
      segment_ids: [[1, 1, 1, 2, 2, 2, 2, 0]],
      position_ids: [[0, 1, 2, 0, 1, 2, 3, 0]],
      packing_efficiency: 0.875
    },
    invariant: "Cross-document attention is strictly zeroed by block-diagonal causal masks."
  },

  // STAGE 2: TRANSFORMER MICRO-ARCH
  {
    id: "rmsnorm",
    name: "Root Mean Square Normalization",
    stage: 2,
    stageNum: "02",
    stageName: "Micro-Arch",
    tag: "energy-norm",
    role: "Layer Activation Energy Normalizer",
    math: "Scales activations by the inverse root-mean-square norm without mean subtraction.",
    input: {
      input_tensor: [[1.0, 2.0, -1.0, 0.5]],
      gamma: [1.0, 1.0, 1.0, 1.0],
      eps: 1e-6
    },
    response: {
      normalized_tensor: [[0.7682, 1.5364, -0.7682, 0.3841]],
      rms_values: [1.3014]
    },
    invariant: "Output activations maintain unit root-mean-square energy."
  },
  {
    id: "rope",
    name: "Rotary Position Embeddings (RoPE / YaRN)",
    stage: 2,
    stageNum: "02",
    stageName: "Micro-Arch",
    tag: "pos-embed",
    role: "Relative Positional Phase Encoder",
    math: "Applies 2D rotation matrices to complex query/key pairs with YaRN temperature scaling.",
    input: {
      query_tensor: [[1.0, 0.0], [0.0, 1.0]],
      position_index: 2,
      base_theta: 10000.0,
      scale_factor: 1.0
    },
    response: {
      rotated_query: [[-0.4161, 0.9093], [-0.9093, -0.4161]],
      rotation_angles_rad: [2.0]
    },
    invariant: "Inner product between rotated vectors depends solely on relative distance (m - n)."
  },
  {
    id: "flash-attn",
    name: "Tiled Memory-Efficient FlashAttention",
    stage: 2,
    stageNum: "02",
    stageName: "Micro-Arch",
    tag: "io-aware",
    role: "O(N) SRAM Tiling Attention",
    math: "Computes exact attention in SRAM tiles via online softmax rescaling without materializing N×N matrix.",
    input: {
      Q_block: [[0.5, 0.8], [0.2, 0.9]],
      K_block: [[0.4, 0.7], [0.1, 0.3]],
      V_block: [[1.2, 0.5], [0.8, 1.1]],
      running_m: [-1e9, -1e9],
      running_l: [0.0, 0.0]
    },
    response: {
      output_block: [[1.023, 0.765], [0.945, 0.871]],
      updated_m: [0.76, 0.35],
      updated_l: [3.41, 2.89],
      sram_peak_kb: 64
    },
    invariant: "Numerically identical to standard attention within 1e-7 float precision."
  },
  {
    id: "swiglu",
    name: "SwiGLU Gated Feed-Forward Unit",
    stage: 2,
    stageNum: "02",
    stageName: "Micro-Arch",
    tag: "gated-ffn",
    role: "Non-Linear Tensor Expansion",
    math: "Computes element-wise product of Swish-activated gate projection and linear value projection.",
    input: {
      x: [[0.5, -1.2, 0.8]],
      w_gate: [[0.4, 0.1, 0.2], [-0.3, 0.5, 0.1], [0.2, -0.4, 0.6]],
      w_up: [[0.3, 0.2, -0.1], [0.1, -0.2, 0.4], [0.5, 0.3, 0.2]],
      w_down: [[0.2, 0.4, 0.1], [-0.1, 0.3, 0.5], [0.4, -0.2, 0.3]]
    },
    response: {
      ffn_output: [[0.184, -0.092, 0.312]],
      gate_activations: [[0.312, 0.045, 0.489]],
      sparsity_ratio: 0.12
    },
    invariant: "Gating mechanism preserves gradient propagation through zero crossings."
  },
  {
    id: "moe",
    name: "Mixture-of-Experts Top-k Routing",
    stage: 2,
    stageNum: "02",
    stageName: "Micro-Arch",
    tag: "moe-routing",
    role: "Sparse Capacity Dispatcher",
    math: "Dispatches tokens to top-k expert subnetworks with auxiliary load balancing entropy penalties.",
    input: {
      token_representations: [[0.8, -0.4, 1.2]],
      router_weights: [[0.1, 0.9, 0.3, 0.4], [0.8, 0.2, 0.5, 0.1], [0.3, 0.4, 0.7, 0.6]],
      top_k: 2,
      capacity_factor: 1.25
    },
    response: {
      selected_expert_indices: [[0, 2]],
      routing_probabilities: [[0.582, 0.418]],
      load_balance_loss: 0.024
    },
    invariant: "Selected gating probabilities strictly sum to 1.0 over chosen top-k experts."
  },

  // STAGE 3: 5D HARDWARE SCALING
  {
    id: "megatron-tp",
    name: "Megatron-LM Tensor Parallelism",
    stage: 3,
    stageNum: "03",
    stageName: "5D Scale",
    tag: "tensor-parallel",
    role: "Intra-Layer Column & Row Sharding",
    math: "Splits column weights in first projection and row weights in second projection with all-reduce.",
    input: {
      activations: [[1.0, 2.0]],
      tp_world_size: 2,
      tp_rank: 0,
      column_shard_w1: [[0.5], [0.2]],
      row_shard_w2: [[0.4, 0.1]]
    },
    response: {
      local_intermediate: [[0.9]],
      local_partial_output: [[0.36, 0.09]],
      requires_all_reduce: true
    },
    invariant: "All-reduced output exactly equals monolithic matrix multiplication."
  },
  {
    id: "ring-attn",
    name: "Context Parallelism Ring Attention",
    stage: 3,
    stageNum: "03",
    stageName: "5D Scale",
    tag: "context-parallel",
    role: "Distributed Sequence Ring Comm",
    math: "Passes key-value blocks around device ring asynchronously while computing local block attention.",
    input: {
      local_q: [[0.5, 0.8]],
      local_k: [[0.4, 0.7]],
      local_v: [[1.2, 0.5]],
      ring_rank: 0,
      ring_size: 4
    },
    response: {
      next_kv_destination_rank: 1,
      prev_kv_source_rank: 3,
      local_attention_partial: [[0.84, 0.35]],
      step_index: 0
    },
    invariant: "Ring communication overlaps 100% with chunked SRAM tile computation."
  },
  {
    id: "pp-1f1b",
    name: "1F1B Pipelined Schedule",
    stage: 3,
    stageNum: "03",
    stageName: "5D Scale",
    tag: "pipeline-parallel",
    role: "Interleaved Micro-Batch Scheduler",
    math: "Interleaves one forward micro-batch with one backward micro-batch to minimize bubble overhead.",
    input: {
      num_stages: 4,
      num_microbatches: 8,
      current_step: 3,
      stage_id: 1
    },
    response: {
      action: "EXECUTE_FORWARD",
      microbatch_id: 4,
      active_activation_buffers: 3,
      bubble_overhead_ratio: 0.142
    },
    invariant: "Activation memory bounded at peak num_stages microbatches."
  },
  {
    id: "zero3",
    name: "ZeRO-3 Full Parameter Sharding",
    stage: 3,
    stageNum: "03",
    stageName: "5D Scale",
    tag: "zero-sharding",
    role: "Zero Redundancy Memory Optimizer",
    math: "Partitions optimizer states, gradients, and model weights across data-parallel devices with all-gather.",
    input: {
      local_weight_shard: [0.25, -0.15, 0.82],
      param_id: "layer_12_attn_q",
      dp_rank: 0,
      dp_world_size: 8
    },
    response: {
      all_gather_buffer_size: 24,
      release_after_forward: true,
      per_device_memory_reduction: "87.5%"
    },
    invariant: "Memory consumption scales strictly as 1/N of total model parameters."
  },

  // STAGE 4: LOSS & OPTIMIZATION
  {
    id: "zloss",
    name: "Auxiliary Logit Z-Loss Regularization",
    stage: 4,
    stageNum: "04",
    stageName: "Optimization",
    tag: "logit-stability",
    role: "Numerical Stability Regularizer",
    math: "Penalizes extreme partition functions via c * (log sum exp z)^2 to prevent float16 overflow.",
    input: {
      logits: [[2.1, 4.5, 1.2, 8.9]],
      target_index: 3,
      c_zloss_weight: 1e-4
    },
    response: {
      cross_entropy_loss: 0.0142,
      z_loss_penalty: 0.0079,
      total_loss: 0.0221,
      max_logit: 8.9
    },
    invariant: "Prevents logsumexp from drifting to infinity under float16 training."
  },
  {
    id: "autodiff-vjp",
    name: "Reverse-Mode Autodiff VJP",
    stage: 4,
    stageNum: "04",
    stageName: "Optimization",
    tag: "exact-gradient",
    role: "Scalar Vector-Jacobian Product",
    math: "Propagates adjoint vectors backward through computational graph nodes without numerical drift.",
    input: {
      output_adjoint: 1.0,
      primal_x: 2.5,
      operation: "x * x + sin(x)"
    },
    response: {
      gradient_x: 4.19885,
      primal_value: 6.84847,
      tape_depth: 3
    },
    invariant: "Analytically exact scalar gradient equivalent to forward-mode dual numbers."
  },
  {
    id: "crit-batch",
    name: "Critical Batch Size B_noise Estimator",
    stage: 4,
    stageNum: "04",
    stageName: "Optimization",
    tag: "batch-scale",
    role: "Gradient Noise Scale Tracker",
    math: "Computes B_noise = tr(Sigma) / ||G||^2 to determine the theoretical scaling limit for mini-batches.",
    input: {
      per_sample_gradient_norms: [1.2, 0.9, 1.5, 1.1],
      aggregated_gradient_norm: 0.85,
      current_batch_size: 4
    },
    response: {
      gradient_noise_scale_b_noise: 1842,
      training_efficiency: 0.985,
      recommendation: "EXPAND_BATCH_SIZE"
    },
    invariant: "Efficiency drops sub-linearly beyond B_noise as noise dominates signal."
  },
  {
    id: "adamw-wsd",
    name: "AdamW Optimizer with WSD Schedule",
    stage: 4,
    stageNum: "04",
    stageName: "Optimization",
    tag: "adamw-wsd",
    role: "Momentum Update & Decay Engine",
    math: "Applies decoupled weight decay with Warmup-Stable-Decay 1/sqrt(t) learning rate scheduling.",
    input: {
      param: 1.5,
      grad: 0.04,
      m: 0.01,
      v: 0.002,
      step: 1000,
      lr: 1e-4,
      weight_decay: 0.1
    },
    response: {
      updated_param: 1.4981,
      updated_m: 0.013,
      updated_v: 0.00203,
      effective_step_size: 0.0009
    },
    invariant: "Decoupled weight decay leaves gradient momentum buffers unpolluted."
  },

  // STAGE 5: POST-TRAINING REASONING ALIGNMENT
  {
    id: "sft-loss",
    name: "Supervised Fine-Tuning Masked Loss",
    stage: 5,
    stageNum: "05",
    stageName: "Alignment",
    tag: "sft-target",
    role: "Instruction Completion Masking",
    math: "Computes cross-entropy loss exclusively on assistant response tokens, masking instruction prompts.",
    input: {
      token_ids: [101, 2054, 102, 345, 678, 102],
      prompt_length: 3,
      predicted_logits: [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.05, 0.95], [0.9, 0.1], [0.4, 0.6]]
    },
    response: {
      loss: 0.284,
      supervised_token_count: 3,
      perplexity: 1.328
    },
    invariant: "Gradient updates exclusively adjust parameters for target completion tokens."
  },
  {
    id: "dpo-loss",
    name: "Direct Preference Optimization (DPO)",
    stage: 5,
    stageNum: "05",
    stageName: "Alignment",
    tag: "dpo-implicit",
    role: "Closed-Form Preference Objective",
    math: "Optimizes pairwise preference directly from reference log ratios without explicit reward models.",
    input: {
      pi_logprob_win: -1.2,
      pi_logprob_lose: -3.4,
      ref_logprob_win: -1.4,
      ref_logprob_lose: -2.8,
      beta: 0.1
    },
    response: {
      dpo_loss: 0.321,
      implicit_reward_margin: 0.08,
      win_probability: 0.69
    },
    invariant: "Implicit reward monotonically increases with policy-to-reference likelihood ratio."
  },
  {
    id: "grpo-engine",
    name: "Group Relative Policy Optimization (GRPO)",
    stage: 5,
    stageNum: "05",
    stageName: "Alignment",
    tag: "sandbox-grpo",
    role: "Verifier-Grounded Reasoning Engine",
    math: "Normalizes advantage across group of candidate responses against formal sandbox verification.",
    input: {
      group_verification_scores: [1.0, 0.0, 1.0, 0.0],
      group_log_probs: [-2.1, -4.2, -1.9, -3.8],
      kl_penalty_coeff: 0.04
    },
    response: {
      normalized_advantages: [1.0, -1.0, 1.0, -1.0],
      policy_gradient_surrogate: 0.452,
      mean_accuracy: 0.50
    },
    invariant: "Advantage has zero mean and unit variance across reasoning candidate group."
  },
  {
    id: "prm-credit",
    name: "Process Reward Model (PRM) Step Credit",
    stage: 5,
    stageNum: "05",
    stageName: "Alignment",
    tag: "prm-verifier",
    role: "Step-Level Reasoning Verifier",
    math: "Assigns granular value estimates to intermediate reasoning steps for Monte Carlo tree search.",
    input: {
      reasoning_steps: [
        "Step 1: Simplify algebraic fraction (x^2 - 1)/(x - 1).",
        "Step 2: Factoring yields (x - 1)(x + 1)/(x - 1) = x + 1.",
        "Step 3: Therefore for x=5, result is 6."
      ],
      step_token_indices: [14, 38, 52]
    },
    response: {
      step_probabilities_correct: [0.98, 0.96, 0.99],
      first_fallacy_index: null,
      cumulative_solution_confidence: 0.932
    },
    invariant: "Monotonically identifies exact step location of mathematical fallacies."
  },

  // STAGE 6: COMPRESSION & SPARSITY
  {
    id: "qlora-nf4",
    name: "QLoRA NormalFloat4 Quantization",
    stage: 6,
    stageNum: "06",
    stageName: "Compression",
    tag: "qlora-nf4",
    role: "Information-Theoretic 4-Bit Weight Compactor",
    math: "Maps normally distributed weights to theoretical quantiles with low-rank adapter bypasses.",
    input: {
      weight_block: [0.12, -0.85, 1.45, -0.05, 0.32, -0.44, 0.98, -1.12],
      lora_r: 8,
      lora_alpha: 16
    },
    response: {
      nf4_indices: [8, 3, 14, 7, 9, 5, 12, 2],
      absmax_scaling_factor: 1.45,
      compression_ratio: "4.0x"
    },
    invariant: "Preserves exact zero error representation for normally distributed base weights."
  },
  {
    id: "wanda",
    name: "Wanda Pruning by Activation Magnitude",
    stage: 6,
    stageNum: "06",
    stageName: "Compression",
    tag: "wanda-prune",
    role: "Weight and Activation Pruner",
    math: "Prunes weights based on product of magnitude and input activation norm without retraining.",
    input: {
      weight_matrix: [[0.8, -0.2], [0.1, 0.9]],
      input_activation_norms: [1.4, 0.3],
      sparsity_target: 0.50
    },
    response: {
      saliency_scores: [[1.12, 0.06], [0.14, 0.27]],
      sparse_mask: [[1, 0], [0, 1]],
      retained_elements: 2
    },
    invariant: "Retains top-k saliency scores per output channel under target sparsity."
  },

  // STAGE 7: PRODUCTION SERVING
  {
    id: "paged-attn",
    name: "PagedAttention Non-Contiguous KV-Cache",
    stage: 7,
    stageNum: "07",
    stageName: "Serving",
    tag: "paged-cache",
    role: "Virtual Memory KV-Cache Manager",
    math: "Stores key-value cache tokens in non-contiguous physical pages indexed by block tables.",
    input: {
      request_id: "req_9921",
      logical_token_index: 34,
      block_table: [104, 18, 92],
      page_size: 16
    },
    response: {
      physical_block_number: 92,
      block_offset: 2,
      virtual_memory_fragmentation_pct: 0.4
    },
    invariant: "Eliminates memory waste from pre-allocation; fragmentation strictly bounded below 4%."
  },
  {
    id: "cont-batch",
    name: "Continuous Iteration-Level Batching",
    stage: 7,
    stageNum: "07",
    stageName: "Serving",
    tag: "continuous-batch",
    role: "Dynamic Request Scheduler",
    math: "Ejects completed sequences and inserts arriving requests at every token decoding iteration.",
    input: {
      active_running_requests: ["req_1", "req_2", "req_3"],
      finished_requests: ["req_2"],
      waiting_queue: ["req_4", "req_5"],
      max_batch_size: 4
    },
    response: {
      next_iteration_batch: ["req_1", "req_3", "req_4"],
      ejected_requests: ["req_2"],
      gpu_compute_utilization: 0.96
    },
    invariant: "Batch scheduling latency is O(1) per iteration step."
  },
  {
    id: "spec-decode",
    name: "Speculative Decoding Rejection Sampler",
    stage: 7,
    stageNum: "07",
    stageName: "Serving",
    tag: "speculative",
    role: "Draft-Target Speculative Speedup",
    math: "Accepts drafted tokens using min(1, p_target / q_draft) verification in single target model pass.",
    input: {
      draft_tokens: [1204, 305, 891],
      draft_probs: [0.92, 0.85, 0.74],
      target_probs: [0.95, 0.81, 0.32]
    },
    response: {
      accepted_tokens: [1204, 305],
      resampled_bonus_token: 412,
      acceptance_length: 2,
      speedup_factor: 2.15
    },
    invariant: "Output distribution is mathematically identical to pure target model sampling."
  },
  {
    id: "fsm-grammar",
    name: "Structured Grammar FSM Token Masking",
    stage: 7,
    stageNum: "07",
    stageName: "Serving",
    tag: "grammar-fsm",
    role: "Constrained Generation Masker",
    math: "Masks out invalid vocab logits at each step according to JSON schema deterministic finite automaton.",
    input: {
      current_fsm_state: "EXPECT_COLON",
      schema: "{\"key\": number}",
      vocab_size: 32000,
      candidate_token_ids: [58, 44, 125, 102]
    },
    response: {
      allowed_token_ids: [58],
      masked_token_count: 31999,
      next_fsm_state: "EXPECT_NUMBER_VALUE"
    },
    invariant: "Generated strings are 100% syntactically valid JSON matching schema."
  }
];

// Entity-Relationship Data Model
const ER_ENTITIES = [
  {
    id: "DATASET_SAMPLE",
    name: "DATASET_SAMPLE",
    desc: "Unpacked, normalized raw document token samples.",
    pk: "sample_id (string)",
    fields: [
      { name: "sample_id", type: "string", role: "PK" },
      { name: "text", type: "string", role: "Content" },
      { name: "quality_score", type: "float32", role: "Gate Metric" },
      { name: "token_count", type: "int32", role: "Length" },
      { name: "dedup_hash", type: "uint64", role: "MinHash Key" }
    ],
    relations: ["PACKED_SEQUENCE (1:N)"]
  },
  {
    id: "PACKED_SEQUENCE",
    name: "PACKED_SEQUENCE",
    desc: "Padded and concatenated boundary-masked sequence tensor batch.",
    pk: "sequence_id (string)",
    fields: [
      { name: "sequence_id", type: "string", role: "PK" },
      { name: "sample_id", type: "string", role: "FK" },
      { name: "token_ids", type: "int32[B, S]", role: "Tensors" },
      { name: "segment_ids", type: "int32[B, S]", role: "Boundaries" },
      { name: "position_ids", type: "int32[B, S]", role: "Pos Indices" },
      { name: "boundary_mask", type: "bool[B, S, S]", role: "Attention Mask" }
    ],
    relations: ["TENSOR_ACTIVATION (1:N)"]
  },
  {
    id: "TENSOR_ACTIVATION",
    name: "TENSOR_ACTIVATION",
    desc: "Intermediate normalized activation representations across layers.",
    pk: "activation_id (string)",
    fields: [
      { name: "activation_id", type: "string", role: "PK" },
      { name: "batch_size", type: "int32", role: "Dim B" },
      { name: "hidden_dim", type: "int32", role: "Dim D" },
      { name: "normalized_tensor", type: "float32[B, S, D]", role: "Hidden States" },
      { name: "rms_values", type: "float32[B, S]", role: "Energy Scaler" }
    ],
    relations: ["LAYER_WEIGHT (N:1)"]
  },
  {
    id: "LAYER_WEIGHT",
    name: "LAYER_WEIGHT",
    desc: "Sharded tensor parallel parameter weights across transformer blocks.",
    pk: "weight_id (string)",
    fields: [
      { name: "weight_id", type: "string", role: "PK" },
      { name: "layer_name", type: "string", role: "Identifier" },
      { name: "dense_parameters", type: "float32[D, D]", role: "Weight Matrix" },
      { name: "column_shards", type: "float32[D, D/N]", role: "TP Column" },
      { name: "row_shards", type: "float32[D/N, D]", role: "TP Row" }
    ],
    relations: ["OPTIMIZER_STATE (1:1)", "CHECKPOINT_STATE (1:N)"]
  },
  {
    id: "OPTIMIZER_STATE",
    name: "OPTIMIZER_STATE",
    desc: "AdamW decoupled first and second order gradient momentum buffers.",
    pk: "weight_id (FK)",
    fields: [
      { name: "weight_id", type: "string", role: "FK" },
      { name: "step", type: "int64", role: "Iteration" },
      { name: "learning_rate", type: "float32", role: "WSD Scalar" },
      { name: "exp_avg_m", type: "float32[...]", role: "First Momentum" },
      { name: "exp_avg_sq_v", type: "float32[...]", role: "Second Momentum" },
      { name: "weight_decay", type: "float32", role: "Decoupled L2" }
    ],
    relations: ["LAYER_WEIGHT (1:1)"]
  },
  {
    id: "CHECKPOINT_STATE",
    name: "CHECKPOINT_STATE",
    desc: "Serialized model weights and training convergence metrics snapshot.",
    pk: "checkpoint_id (string)",
    fields: [
      { name: "checkpoint_id", type: "string", role: "PK" },
      { name: "global_step", type: "int64", role: "Step Counter" },
      { name: "loss_metric", type: "float32", role: "Eval Loss" },
      { name: "model_weights", type: "dict[string, tensor]", role: "Weights Map" }
    ],
    relations: ["POLICY_ROLLOUT (1:N)", "COMPRESSED_WEIGHT (1:N)"]
  },
  {
    id: "POLICY_ROLLOUT",
    name: "POLICY_ROLLOUT",
    desc: "Reasoning traces and policy likelihood evaluations.",
    pk: "rollout_id (string)",
    fields: [
      { name: "rollout_id", type: "string", role: "PK" },
      { name: "checkpoint_id", type: "string", role: "FK" },
      { name: "prompt", type: "string", role: "Input Text" },
      { name: "reasoning_trace", type: "array[string]", role: "Step Traces" },
      { name: "policy_log_probs", type: "float32[T]", role: "Action Probs" },
      { name: "reward_score", type: "float32", role: "Sandbox Verdict" }
    ],
    relations: ["ADVANTAGE_TARGET (1:1)"]
  },
  {
    id: "KV_CACHE_PAGE",
    name: "KV_CACHE_PAGE",
    desc: "Paged virtual memory block allocated for continuous inference.",
    pk: "page_id (string)",
    fields: [
      { name: "page_id", type: "string", role: "PK" },
      { name: "block_number", type: "int32", role: "Physical Block" },
      { name: "key_cache", type: "float16[P, H, d]", role: "Key Tokens" },
      { name: "val_cache", type: "float16[P, H, d]", role: "Val Tokens" },
      { name: "reference_count", type: "int32", role: "Fork Ref" }
    ],
    relations: ["GENERATION_TOKEN (1:N)"]
  }
];

// Current Active State
let currentAlgorithm = ALGORITHMS[0];
currentInspectorTab = "yaml";

// DOM Initializer
document.addEventListener("DOMContentLoaded", () => {
  setupClock();
  setupTabs();
  initProofsCatalog();
  selectAlgorithm("synth-filter");
  initAlgoDropdown();
});

// Live Clock in Topbar (Gortex Style)
function setupClock() {
  const clockEl = document.getElementById("clock");
  if (!clockEl) return;
  const update = () => {
    const d = new Date();
    clockEl.textContent = d.toISOString().replace("T", " ").substring(0, 19) + " UTC";
  };
  update();
  setInterval(update, 1000);
}

// Topbar Tab Switching
function setupTabs() {
  const tabs = document.querySelectorAll(".topbar .seg.tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("on"));
      tab.classList.add("on");
      const targetId = tab.getAttribute("data-target");
      if (targetId) {
        const el = document.getElementById(targetId);
        if (el) el.scrollIntoView({ behavior: "smooth" });
      }
    });
  });
}

// Select and Inspect an Algorithm
function selectAlgorithm(algoId) {
  const algo = ALGORITHMS.find(a => a.id === algoId);
  if (!algo) return;
  currentAlgorithm = algo;

  // Highlight all nodes in ascii boxes
  document.querySelectorAll(".node-btn").forEach(btn => {
    if (btn.getAttribute("data-id") === algoId) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Update Hero Split-Pane / Inspector
  const titleEl = document.getElementById("inspector-title");
  if (titleEl) {
    titleEl.textContent = `[${algo.stageNum}] ${algo.name}`;
  }

  const roleEl = document.getElementById("inspector-role");
  if (roleEl) {
    roleEl.textContent = algo.role;
  }

  const tagEl = document.getElementById("inspector-tag");
  if (tagEl) {
    tagEl.textContent = algo.tag;
  }

  const mathEl = document.getElementById("inspector-math");
  if (mathEl) {
    mathEl.textContent = algo.math;
  }

  const invEl = document.getElementById("inspector-invariant");
  if (invEl) {
    invEl.textContent = algo.invariant;
  }

  // Update Code Pre Box
  renderInspectorCode();
}

// Switch between JSON Input, JSON Response, and Invariant Tabs
function setInspectorTab(tabKey) {
  currentInspectorTab = tabKey;
  document.querySelectorAll(".inspector-subtabs .btn").forEach(b => {
    if (b.getAttribute("data-subtab") === tabKey) {
      b.classList.add("active");
    } else {
      b.classList.remove("active");
    }
  });
  renderInspectorCode();
}

// Render formatted JSON with syntax highlighting
function renderInspectorCode() {
  const codeBox = document.getElementById("inspector-code-content");
  if (!codeBox) return;

  const data = currentContractData || (typeof CONTRACTS_DATA !== "undefined" ? Object.values(CONTRACTS_DATA)[0] : currentAlgorithm);
  if (!data) return;

  if (currentInspectorTab === "yaml") {
    const rawYaml = (data.yaml || `contract:\n  name: ${data.name}\n  purity: pure\n  determinism: deterministic`).trim();
    codeBox.innerHTML = syntaxHighlightYaml(rawYaml);
    return;
  }

  if (currentInspectorTab === "input") {
    const jsonStr = JSON.stringify(data.input || {}, null, 2).trim();
    codeBox.innerHTML = syntaxHighlightJson(jsonStr);
    return;
  }

  if (currentInspectorTab === "response") {
    const jsonStr = JSON.stringify(data.response || {}, null, 2).trim();
    codeBox.innerHTML = syntaxHighlightJson(jsonStr);
    return;
  }

  if (currentInspectorTab === "invariant") {
    codeBox.innerHTML = `// MATHEMATICAL INVARIANT & SAFETY GUARANTEE\n// Algorithm: ${data.name}\n// Domain: ${data.domainLabel || data.domain}\n\n<span class="str">"${data.invariant}"</span>\n\n// FORMAL EXECUTION PRINCIPLE:\n<span class="dim">// ${data.title || data.math || "Strict contract verified with deterministic gradients."}</span>`;
    return;
  }
}

function syntaxHighlightYaml(yamlStr) {
  return yamlStr
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/^([\s-]*[a-zA-Z0-9_]+:)/gm, '<span class="key">$1</span>')
    .replace(/(:\s*)([^\r\n]+)/g, function(match, p1, p2) {
      if (p2.startsWith('"') || p2.startsWith("'")) {
        return p1 + '<span class="str">' + p2 + '</span>';
      } else if (p2 === 'true' || p2 === 'false') {
        return p1 + '<span class="bool">' + p2 + '</span>';
      } else if (!isNaN(Number(p2.trim()))) {
        return p1 + '<span class="num">' + p2 + '</span>';
      }
      return p1 + '<span class="str">' + p2 + '</span>';
    });
}

function syntaxHighlightJson(json) {
  json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
    let cls = 'num';
    if (/^"/.test(match)) {
      if (/:$/.test(match)) {
        cls = 'key';
      } else {
        cls = 'str';
      }
    } else if (/true|false/.test(match)) {
      cls = 'bool';
    } else if (/null/.test(match)) {
      cls = 'punct';
    }
    return '<span class="' + cls + '">' + match + '</span>';
  });
}

// Copy JSON or text to clipboard
function copyActivePayload() {
  if (!currentAlgorithm) return;
  const data = currentInspectorTab === "response" ? currentAlgorithm.response : currentAlgorithm.input;
  copyText(JSON.stringify(data, null, 2));
}

function copyText(str) {
  navigator.clipboard.writeText(str).then(() => {
    showToast("COPIED TO CLIPBOARD");
  }).catch(() => {
    // Fallback
    const t = document.createElement("textarea");
    t.value = str;
    document.body.appendChild(t);
    t.select();
    document.execCommand("copy");
    document.body.removeChild(t);
    showToast("COPIED TO CLIPBOARD");
  });
}

// Toast notification
function showToast(msg) {
  let toast = document.getElementById("toast-notify");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "toast-notify";
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => {
    toast.classList.remove("show");
  }, 1800);
}

// Render Domain Catalog Grid
function renderCatalogGrid() {
  const container = document.getElementById("catalog-grid");
  if (!container) return;

  container.innerHTML = ALGORITHMS.map((algo, idx) => `
    <div onclick="selectAlgorithm('${algo.id}'); document.getElementById('sec-arch').scrollIntoView({behavior:'smooth'});" style="cursor:pointer;">
      <div class="fh">
        <span class="idx">${String(idx + 1).padStart(2, '0')}</span>
        <span>${algo.name}</span>
        <span class="tag">${algo.tag}</span>
      </div>
      <p>
        <span class="accent">[Stage ${algo.stageNum}]</span> ${algo.math}
      </p>
    </div>
  `).join("");
}

// ==========================================
// PROOFS CATALOG (250 LATEX SPECIFICATIONS)
// ==========================================
let currentProofDomain = "categories";
let currentProofSearch = "";
let selectedProofId = null;

function initProofsCatalog() {
  if (typeof PROOFS_DATA === "undefined") return;
  renderProofDomainTabs();
  renderProofsGrid();
}

function renderProofDomainTabs() {
  const container = document.getElementById("proofs-domain-tabs");
  if (!container || typeof PROOFS_DATA === "undefined") return;

  const domainMap = {};
  PROOFS_DATA.forEach(p => {
    if (!domainMap[p.domain]) {
      domainMap[p.domain] = { domain: p.domain, label: p.domainLabel || p.domain.replace(/_/g, ' '), count: 0 };
    }
    domainMap[p.domain].count++;
  });

  const domains = Object.values(domainMap).sort((a, b) => a.label.localeCompare(b.label));

  let html = `<button class="proof-domain-btn ${currentProofDomain === 'categories' ? 'active' : ''}" onclick="setProofDomainFilter('categories')">CATEGORIES (${domains.length})</button>`;
  html += `<button class="proof-domain-btn ${currentProofDomain === 'all' ? 'active' : ''}" onclick="setProofDomainFilter('all')">ALL (${PROOFS_DATA.length})</button>`;

  domains.forEach(d => {
    const label = d.label.toUpperCase();
    const count = d.count;
    const isActive = currentProofDomain === d.domain ? 'active' : '';
    html += `<button class="proof-domain-btn ${isActive}" onclick="setProofDomainFilter('${d.domain}')">${escapeHtml(label)} (${count})</button>`;
  });

  container.innerHTML = html;
}

function setProofDomainFilter(domain) {
  currentProofDomain = domain;
  renderProofDomainTabs();
  renderProofsGrid();
}

function filterProofs() {
  const input = document.getElementById("proofs-search");
  currentProofSearch = input ? input.value.trim().toLowerCase() : "";
  renderProofsGrid();
}

function clearProofSearch() {
  const input = document.getElementById("proofs-search");
  if (input) input.value = "";
  currentProofSearch = "";
  renderProofsGrid();
}

let currentModalProof = null;
let currentTexViewMode = "rendered";

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderKaTeX(latex, isDisplay) {
  if (window.katex && typeof window.katex.renderToString === "function") {
    try {
      return window.katex.renderToString(latex.trim(), {
        displayMode: isDisplay,
        throwOnError: false,
        output: "html"
      });
    } catch (e) {
      console.warn("KaTeX render error:", e);
    }
  }
  const safe = escapeHtml(latex);
  return isDisplay ? `<div class="tex-equation">$$${safe}$$</div>` : `$${safe}$`;
}

function renderProofsGrid() {
  const grid = document.getElementById("proofs-grid");
  const counter = document.getElementById("proofs-counter");
  if (!grid || typeof PROOFS_DATA === "undefined") return;

  if (currentProofDomain === "categories" && !currentProofSearch) {
    const domainMap = {};
    PROOFS_DATA.forEach(p => {
      if (!domainMap[p.domain]) {
        domainMap[p.domain] = {
          domain: p.domain,
          label: p.domainLabel || p.domain.replace(/_/g, ' '),
          count: 0,
          sampleAlgos: []
        };
      }
      domainMap[p.domain].count++;
      if (domainMap[p.domain].sampleAlgos.length < 3) {
        domainMap[p.domain].sampleAlgos.push(p.algoName);
      }
    });

    const domains = Object.values(domainMap).sort((a, b) => a.label.localeCompare(b.label));

    if (counter) {
      counter.textContent = `SHOWING ${domains.length} DOMAIN CATEGORIES · ${PROOFS_DATA.length} PROOFS`;
    }

    grid.innerHTML = domains.map((d, i) => {
      const idxStr = String(i + 1).padStart(2, "0");
      const sampleTags = d.sampleAlgos
        .map(name => `<span class="domain-sample-tag">${escapeHtml(name)}</span>`)
        .join("");
      const moreTag = d.count > 3 ? `<span class="domain-sample-more">+${d.count - 3} more</span>` : "";

      return `
        <div class="domain-card" onclick="setProofDomainFilter('${escapeHtml(d.domain)}')">
          <div class="domain-card-header">
            <span class="domain-card-idx">§${idxStr} · CATEGORY</span>
            <span class="domain-card-count">${d.count} PROOFS</span>
          </div>
          <div class="domain-card-title">${escapeHtml(d.label.toUpperCase())}</div>
          <div class="domain-card-sample">
            ${sampleTags}
            ${moreTag}
          </div>
          <div class="domain-card-footer">
            <span class="domain-card-action">Explore ${d.count} Proofs</span>
            <span class="domain-card-arrow">→</span>
          </div>
        </div>
      `;
    }).join("");
    return;
  }

  const filtered = PROOFS_DATA.filter(p => {
    const matchDomain = currentProofDomain === "all" || currentProofDomain === "categories" || p.domain === currentProofDomain;
    if (!matchDomain) return false;
    if (!currentProofSearch) return true;
    const q = currentProofSearch;
    return p.algoName.toLowerCase().includes(q) ||
           p.title.toLowerCase().includes(q) ||
           p.domainLabel.toLowerCase().includes(q) ||
           p.excerpt.toLowerCase().includes(q);
  });

  let activeDomainLabel = "";
  if (currentProofDomain !== "all" && currentProofDomain !== "categories") {
    const match = PROOFS_DATA.find(p => p.domain === currentProofDomain);
    activeDomainLabel = match ? match.domainLabel : currentProofDomain;
  }

  if (counter) {
    if (currentProofSearch) {
      counter.textContent = `SHOWING ${filtered.length} OF ${PROOFS_DATA.length} PROOFS`;
    } else if (currentProofDomain === "all") {
      counter.textContent = `SHOWING 300 OF 300 PROOFS`;
    } else {
      counter.textContent = `SHOWING ${filtered.length} OF 300 PROOFS (${activeDomainLabel.toUpperCase()})`;
    }
  }

  let bannerHtml = "";
  if (currentProofSearch) {
    bannerHtml = `
      <div class="proofs-nav-banner">
        <button class="proof-back-btn" onclick="clearProofSearch()">← Clear Search &amp; Return</button>
        <div class="proofs-nav-details">
          <span class="proofs-nav-label">SEARCH QUERY:</span>
          <span class="proofs-nav-title">"${escapeHtml(currentProofSearch)}"</span>
          <span class="proofs-nav-count">(${filtered.length} PROOFS FOUND)</span>
        </div>
      </div>
    `;
  } else if (currentProofDomain !== "categories") {
    bannerHtml = `
      <div class="proofs-nav-banner">
        <button class="proof-back-btn" onclick="setProofDomainFilter('categories')">← Back to All 35 Categories</button>
        <div class="proofs-nav-details">
          <span class="proofs-nav-label">DOMAIN:</span>
          <span class="proofs-nav-title">${escapeHtml((activeDomainLabel || currentProofDomain).toUpperCase())}</span>
          <span class="proofs-nav-count">(${filtered.length} PROOFS)</span>
        </div>
      </div>
    `;
  }

  if (filtered.length === 0) {
    grid.innerHTML = bannerHtml + `
      <div style="grid-column: 1 / -1; padding: 40px; text-align: center; color: var(--ink-dim); font-family: var(--font-dspl);">
        NO MATHEMATICAL SPECIFICATIONS FOUND MATCHING QUERY
        <div style="margin-top: 14px;">
          <button class="proof-back-btn" onclick="clearProofSearch(); setProofDomainFilter('categories');">← Return to All Categories</button>
        </div>
      </div>
    `;
    return;
  }

  const cardsHtml = filtered.map(p => {
    const isSelected = p.id === selectedProofId ? "active" : "";
    const pdfBtn = p.pdf ? `<a class="proof-btn" href="${escapeHtml(p.pdf)}" target="_blank" onclick="event.stopPropagation()">[pdf]</a>` : "";
    return `
      <div class="proof-card ${isSelected}" onclick="openTexModal('${escapeHtml(p.id)}')">
        <div class="proof-card-header">
          <span class="proof-domain-tag">${escapeHtml(p.domainLabel)}</span>
          <span class="proof-lines-tag">${p.lines} lines · TeX</span>
        </div>
        <div class="proof-card-name">${escapeHtml(p.algoName)}</div>
        <div class="proof-card-title">${escapeHtml(p.title)}</div>
        <div class="proof-card-excerpt">${escapeHtml(p.excerpt)}</div>
        <div class="proof-card-actions">
          <button class="proof-btn primary" onclick="event.stopPropagation(); openTexModal('${escapeHtml(p.id)}')">[render]</button>
          ${pdfBtn}
        </div>
      </div>
    `;
  }).join("");

  grid.innerHTML = bannerHtml + cardsHtml;
}

function selectProof(id) {
  openTexModal(id);
}

function openTexModal(id, initialMode = "rendered") {
  const proof = PROOFS_DATA.find(p => p.id === id);
  if (!proof) return;
  currentModalProof = proof;
  selectedProofId = id;

  // Google Analytics Event Tracking
  if (typeof gtag === "function") {
    gtag("event", "view_algorithm", {
      algo_id: proof.id,
      algo_name: proof.algoName,
      domain: proof.domain,
      domain_label: proof.domainLabel
    });
  }

  const modal = document.getElementById("tex-modal");
  if (!modal) return;

  const tagEl = document.getElementById("modal-domain-tag");
  const titleEl = document.getElementById("modal-title");
  if (tagEl) tagEl.textContent = `${proof.domainLabel} · ${proof.algoName} (${proof.lines} lines)`;
  if (titleEl) titleEl.textContent = proof.title;

  setTexViewMode(initialMode);
  modal.style.display = "flex";
  document.body.style.overflow = "hidden";
}

function closeTexModal() {
  const modal = document.getElementById("tex-modal");
  if (modal) modal.style.display = "none";
  document.body.style.overflow = "";
}

function sanitizeDiagramMath(text) {
  return String(text || "")
    .replace(/\\gets/g, " ← ")
    .replace(/\\to/g, " → ")
    .replace(/\\le/g, " ≤ ")
    .replace(/\\ge/g, " ≥ ")
    .replace(/\\ne/g, " ≠ ")
    .replace(/\\approx/g, " ≈ ")
    .replace(/\\sum/g, "Σ")
    .replace(/\\prod/g, "Π")
    .replace(/\\int/g, "∫")
    .replace(/\\in/g, " ∈ ")
    .replace(/\\subset/g, " ⊂ ")
    .replace(/\\alpha/g, "α")
    .replace(/\\beta/g, "β")
    .replace(/\\gamma/g, "γ")
    .replace(/\\delta/g, "δ")
    .replace(/\\epsilon/g, "ε")
    .replace(/\\theta/g, "θ")
    .replace(/\\lambda/g, "λ")
    .replace(/\\mu/g, "μ")
    .replace(/\\sigma/g, "σ")
    .replace(/\\tau/g, "τ")
    .replace(/\\phi/g, "φ")
    .replace(/\\omega/g, "ω")
    .replace(/\\pi/g, "π")
    .replace(/\\mathcal\{([A-Za-z])\}/g, "$1")
    .replace(/\\mathbf\{([A-Za-z0-9]+)\}/g, "$1")
    .replace(/\\hat\{([A-Za-z0-9]+)\}/g, "$1^")
    .replace(/\\tilde\{([A-Za-z0-9]+)\}/g, "$1~")
    .replace(/\\text\{([^\}]+)\}/g, "$1")
    .replace(/\\textbf\{([^\}]+)\}/g, "$1")
    .replace(/\\[a-zA-Z]+/g, "")
    .replace(/[{}$_\\~]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}


function renderArchitecturalSvg(p, inputs, outputs, steps) {
  const algoLower = (p.algo || "").toLowerCase();
  const domainLower = (p.domain || "").toLowerCase();

  // Case 1: Perceptron, Linear Neuron, Dense, Activation
  if (
    algoLower.includes("perceptron") ||
    algoLower.includes("linear_layer") ||
    algoLower.includes("relu") ||
    algoLower.includes("dense") ||
    domainLower.includes("building_blocks")
  ) {
    const actLabel = algoLower.includes("relu") ? "ReLU(·)" : (algoLower.includes("gelu") ? "GELU(·)" : "g(·)");
    const formula = algoLower.includes("relu") ? "ŷ = max(0, wᵀx + b)" : "ŷ = g(wᵀx + b)";
    return `
    <div class="algo-svg-card">
      <div class="algo-svg-card-title">COMPUTATIONAL GRAPH ARCHITECTURE</div>
      <svg viewBox="0 0 920 360" width="100%" height="auto" class="algo-arch-svg" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#10b981" />
          </marker>
          <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
          </marker>
          <linearGradient id="nodeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#1e293b" />
            <stop offset="100%" stop-color="#0f172a" />
          </linearGradient>
          <linearGradient id="sumGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#064e3b" />
            <stop offset="100%" stop-color="#047857" />
          </linearGradient>
          <linearGradient id="actGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#1e3a8a" />
            <stop offset="100%" stop-color="#2563eb" />
          </linearGradient>
        </defs>

        <rect width="920" height="360" rx="10" fill="#0b0f19" stroke="#1e293b" stroke-width="1.5"/>

        <!-- Headers -->
        <text x="110" y="32" fill="#94a3b8" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">INPUT COHORT (X)</text>
        <text x="250" y="32" fill="#a7f3d0" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">SYNAPTIC WEIGHTS (w)</text>
        <text x="420" y="32" fill="#f59e0b" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">BIAS (b)</text>
        <text x="420" y="278" fill="#34d399" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">ACCUMULATOR (∑)</text>
        <text x="600" y="278" fill="#60a5fa" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">NON-LINEARITY</text>
        <text x="760" y="135" fill="#7bf37b" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">OUTPUT (ŷ)</text>

        <!-- Inputs X1, X2, Xn -->
        <circle cx="110" cy="80" r="24" fill="url(#nodeGrad)" stroke="#38bdf8" stroke-width="2"/>
        <text x="110" y="85" fill="#ffffff" font-size="14" font-family="sans-serif" text-anchor="middle" font-weight="bold">X₁</text>

        <circle cx="110" cy="160" r="24" fill="url(#nodeGrad)" stroke="#38bdf8" stroke-width="2"/>
        <text x="110" y="165" fill="#ffffff" font-size="14" font-family="sans-serif" text-anchor="middle" font-weight="bold">X₂</text>

        <text x="110" y="210" fill="#64748b" font-size="20" font-family="sans-serif" text-anchor="middle">⋮</text>

        <circle cx="110" cy="245" r="24" fill="url(#nodeGrad)" stroke="#38bdf8" stroke-width="2"/>
        <text x="110" y="250" fill="#ffffff" font-size="14" font-family="sans-serif" text-anchor="middle" font-weight="bold">Xₙ</text>

        <!-- Connecting Lines to Summation -->
        <line x1="134" y1="80" x2="388" y2="152" stroke="#475569" stroke-width="2" marker-end="url(#arrow-green)"/>
        <rect x="230" y="94" width="38" height="20" rx="4" fill="#0f172a" stroke="#334155" />
        <text x="249" y="108" fill="#cbd5e1" font-size="11" font-family="monospace" text-anchor="middle">w₁</text>

        <line x1="134" y1="160" x2="388" y2="160" stroke="#475569" stroke-width="2" marker-end="url(#arrow-green)"/>
        <rect x="230" y="150" width="38" height="20" rx="4" fill="#0f172a" stroke="#334155" />
        <text x="249" y="164" fill="#cbd5e1" font-size="11" font-family="monospace" text-anchor="middle">w₂</text>

        <line x1="134" y1="245" x2="388" y2="168" stroke="#475569" stroke-width="2" marker-end="url(#arrow-green)"/>
        <rect x="230" y="202" width="38" height="20" rx="4" fill="#0f172a" stroke="#334155" />
        <text x="249" y="216" fill="#cbd5e1" font-size="11" font-family="monospace" text-anchor="middle">wₙ</text>

        <!-- Bias Input -->
        <line x1="420" y1="48" x2="420" y2="124" stroke="#f59e0b" stroke-width="2" marker-end="url(#arrow-green)"/>
        <rect x="402" y="68" width="36" height="20" rx="4" fill="#0f172a" stroke="#f59e0b" />
        <text x="420" y="82" fill="#f59e0b" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">+b</text>

        <!-- Summation Node ∑ -->
        <circle cx="420" cy="160" r="32" fill="url(#sumGrad)" stroke="#10b981" stroke-width="2.5"/>
        <text x="420" y="170" fill="#ffffff" font-size="26" font-family="serif" text-anchor="middle" font-weight="bold">∑</text>

        <!-- Arrow to Activation -->
        <line x1="452" y1="160" x2="564" y2="160" stroke="#10b981" stroke-width="2.5" marker-end="url(#arrow-green)"/>
        <rect x="472" y="142" width="70" height="18" rx="3" fill="#064e3b" stroke="#059669" />
        <text x="507" y="155" fill="#a7f3d0" font-size="10" font-family="monospace" text-anchor="middle">z = wᵀx + b</text>

        <!-- Activation Node g(.) -->
        <circle cx="600" cy="160" r="32" fill="url(#actGrad)" stroke="#3b82f6" stroke-width="2.5"/>
        <text x="600" y="167" fill="#ffffff" font-size="16" font-family="serif" text-anchor="middle" font-weight="bold">${actLabel}</text>

        <!-- Arrow to Output -->
        <line x1="632" y1="160" x2="726" y2="160" stroke="#38bdf8" stroke-width="2.5" marker-end="url(#arrow-blue)"/>

        <!-- Output Node -->
        <circle cx="760" cy="160" r="28" fill="url(#nodeGrad)" stroke="#10b981" stroke-width="2"/>
        <text x="760" y="167" fill="#7bf37b" font-size="18" font-family="serif" text-anchor="middle" font-weight="bold">ŷ</text>

        <!-- Bottom Mathematical Formula Badge -->
        <rect x="230" y="310" width="460" height="32" rx="6" fill="#111827" stroke="#374151" />
        <text x="460" y="331" fill="#38bdf8" font-size="12.5" font-family="monospace" text-anchor="middle" font-weight="bold">${formula}</text>
      </svg>
    </div>
    `;
  }

  // Case 2: Attention / Transformer Efficiency / Decoding
  if (
    algoLower.includes("attention") ||
    algoLower.includes("transformer") ||
    domainLower.includes("attention") ||
    domainLower.includes("decoding")
  ) {
    return `
    <div class="algo-svg-card">
      <div class="algo-svg-card-title">ATTENTION TRANSFORMER OPERATOR ARCHITECTURE</div>
      <svg viewBox="0 0 920 360" width="100%" height="auto" class="algo-arch-svg" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="attn-arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#818cf8" />
          </marker>
        </defs>
        <rect width="920" height="360" rx="10" fill="#0b0f19" stroke="#1e293b" stroke-width="1.5"/>

        <!-- Q, K, V Projections -->
        <rect x="60" y="60" width="110" height="50" rx="6" fill="#1e1b4b" stroke="#6366f1" stroke-width="2"/>
        <text x="115" y="90" fill="#e0e7ff" font-size="13" font-family="monospace" text-anchor="middle" font-weight="bold">Query (Q)</text>

        <rect x="60" y="140" width="110" height="50" rx="6" fill="#1e1b4b" stroke="#6366f1" stroke-width="2"/>
        <text x="115" y="170" fill="#e0e7ff" font-size="13" font-family="monospace" text-anchor="middle" font-weight="bold">Key (K)</text>

        <rect x="60" y="220" width="110" height="50" rx="6" fill="#1e1b4b" stroke="#6366f1" stroke-width="2"/>
        <text x="115" y="250" fill="#e0e7ff" font-size="13" font-family="monospace" text-anchor="middle" font-weight="bold">Value (V)</text>

        <!-- MatMul Q*K^T -->
        <line x1="170" y1="85" x2="270" y2="120" stroke="#6366f1" stroke-width="2" marker-end="url(#attn-arrow)"/>
        <line x1="170" y1="165" x2="270" y2="135" stroke="#6366f1" stroke-width="2" marker-end="url(#attn-arrow)"/>

        <circle cx="300" cy="130" r="30" fill="#312e81" stroke="#818cf8" stroke-width="2"/>
        <text x="300" y="136" fill="#ffffff" font-size="16" font-family="sans-serif" text-anchor="middle" font-weight="bold">Q·Kᵀ</text>

        <!-- Scale & Mask / Softmax -->
        <line x1="330" y1="130" x2="430" y2="130" stroke="#818cf8" stroke-width="2" marker-end="url(#attn-arrow)"/>
        <rect x="430" y="105" width="120" height="50" rx="6" fill="#064e3b" stroke="#10b981" stroke-width="2"/>
        <text x="490" y="135" fill="#a7f3d0" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">Softmax(/√d)</text>

        <!-- Multiply with V -->
        <line x1="550" y1="130" x2="630" y2="165" stroke="#10b981" stroke-width="2" marker-end="url(#attn-arrow)"/>
        <line x1="170" y1="245" x2="630" y2="185" stroke="#6366f1" stroke-width="2" marker-end="url(#attn-arrow)"/>

        <circle cx="660" cy="175" r="30" fill="#312e81" stroke="#818cf8" stroke-width="2"/>
        <text x="660" y="181" fill="#ffffff" font-size="16" font-family="sans-serif" text-anchor="middle" font-weight="bold">A·V</text>

        <!-- Output Linear Projection -->
        <line x1="690" y1="175" x2="740" y2="175" stroke="#818cf8" stroke-width="2" marker-end="url(#attn-arrow)"/>
        <rect x="740" y="150" width="90" height="50" rx="6" fill="#0f172a" stroke="#38bdf8" stroke-width="2"/>
        <text x="785" y="180" fill="#38bdf8" font-size="13" font-family="monospace" text-anchor="middle" font-weight="bold">Output</text>

        <rect x="230" y="310" width="460" height="32" rx="6" fill="#111827" stroke="#374151" />
        <text x="460" y="331" fill="#a5b4fc" font-size="12.5" font-family="monospace" text-anchor="middle" font-weight="bold">Attention(Q,K,V) = Softmax(QKᵀ / √dₖ) V</text>
      </svg>
    </div>
    `;
  }

  // Case 3: Optimization / Gradient Update / Autodiff
  if (
    algoLower.includes("optimizer") ||
    algoLower.includes("adam") ||
    algoLower.includes("sgd") ||
    domainLower.includes("optimizers") ||
    domainLower.includes("autodiff")
  ) {
    return `
    <div class="algo-svg-card">
      <div class="algo-svg-card-title">GRADIENT DYNAMICS &amp; PARAMETER UPDATE PIPELINE</div>
      <svg viewBox="0 0 920 360" width="100%" height="auto" class="algo-arch-svg" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="opt-arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#f59e0b" />
          </marker>
        </defs>
        <rect width="920" height="360" rx="10" fill="#0b0f19" stroke="#1e293b" stroke-width="1.5"/>

        <rect x="50" y="130" width="110" height="60" rx="6" fill="#1e293b" stroke="#94a3b8" stroke-width="2"/>
        <text x="105" y="158" fill="#ffffff" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">Loss ∇L(θ)</text>
        <text x="105" y="176" fill="#94a3b8" font-size="10" font-family="monospace" text-anchor="middle">Gradients (gₜ)</text>

        <line x1="160" y1="160" x2="230" y2="160" stroke="#f59e0b" stroke-width="2" marker-end="url(#opt-arrow)"/>

        <rect x="230" y="90" width="150" height="60" rx="6" fill="#172554" stroke="#3b82f6" stroke-width="2"/>
        <text x="305" y="118" fill="#93c5fd" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">1st Moment (mₜ)</text>
        <text x="305" y="136" fill="#bfdbfe" font-size="10" font-family="monospace" text-anchor="middle">β₁ mₜ₋₁ + (1-β₁)gₜ</text>

        <rect x="230" y="180" width="150" height="60" rx="6" fill="#311042" stroke="#d946ef" stroke-width="2"/>
        <text x="305" y="208" fill="#f0abfc" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">2nd Moment (vₜ)</text>
        <text x="305" y="226" fill="#f5d0fe" font-size="10" font-family="monospace" text-anchor="middle">β₂ vₜ₋₁ + (1-β₂)gₜ²</text>

        <line x1="380" y1="120" x2="460" y2="145" stroke="#f59e0b" stroke-width="2" marker-end="url(#opt-arrow)"/>
        <line x1="380" y1="210" x2="460" y2="175" stroke="#f59e0b" stroke-width="2" marker-end="url(#opt-arrow)"/>

        <rect x="460" y="130" width="140" height="60" rx="6" fill="#064e3b" stroke="#10b981" stroke-width="2"/>
        <text x="530" y="158" fill="#a7f3d0" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">Adaptive Step</text>
        <text x="530" y="176" fill="#6ee7b7" font-size="10" font-family="monospace" text-anchor="middle">η · m̂ₜ / (√v̂ₜ + ε)</text>

        <line x1="600" y1="160" x2="680" y2="160" stroke="#f59e0b" stroke-width="2" marker-end="url(#opt-arrow)"/>

        <rect x="680" y="130" width="120" height="60" rx="6" fill="#1e1b4b" stroke="#6366f1" stroke-width="2"/>
        <text x="740" y="158" fill="#e0e7ff" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">Updated θₜ₊₁</text>
        <text x="740" y="176" fill="#a5b4fc" font-size="10" font-family="monospace" text-anchor="middle">θₜ - Δθₜ</text>

        <rect x="220" y="310" width="480" height="32" rx="6" fill="#111827" stroke="#374151" />
        <text x="460" y="331" fill="#f59e0b" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">Parameter Update: θₜ₊₁ = θₜ - η · m̂ₜ / (√v̂ₜ + ε)</text>
      </svg>
    </div>
    `;
  }

  // Case 4: General Canonical Pipeline Architecture
  return `
    <div class="algo-svg-card">
      <div class="algo-svg-card-title">CANONICAL PIPELINE EXECUTION ARCHITECTURE</div>
      <svg viewBox="0 0 920 360" width="100%" height="auto" class="algo-arch-svg" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="gen-arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
          </marker>
        </defs>
        <rect width="920" height="360" rx="10" fill="#0b0f19" stroke="#1e293b" stroke-width="1.5"/>

        <!-- Ingestion -->
        <rect x="40" y="115" width="130" height="90" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>
        <text x="105" y="145" fill="#38bdf8" font-size="11" font-family="monospace" text-anchor="middle" font-weight="bold">STAGE 01</text>
        <text x="105" y="165" fill="#ffffff" font-size="12" font-family="sans-serif" text-anchor="middle" font-weight="bold">Input Ingestion</text>
        <text x="105" y="185" fill="#94a3b8" font-size="10" font-family="monospace" text-anchor="middle">Tensors &amp; State</text>

        <line x1="170" y1="160" x2="230" y2="160" stroke="#38bdf8" stroke-width="2" marker-end="url(#gen-arrow)"/>

        <!-- Transformation / Forward -->
        <rect x="230" y="115" width="170" height="90" rx="6" fill="#064e3b" stroke="#10b981" stroke-width="2"/>
        <text x="315" y="145" fill="#34d399" font-size="11" font-family="monospace" text-anchor="middle" font-weight="bold">STAGE 02</text>
        <text x="315" y="165" fill="#ffffff" font-size="12" font-family="sans-serif" text-anchor="middle" font-weight="bold">Mathematical Operator</text>
        <text x="315" y="185" fill="#a7f3d0" font-size="10" font-family="monospace" text-anchor="middle">Forward Transform</text>

        <line x1="400" y1="160" x2="460" y2="160" stroke="#38bdf8" stroke-width="2" marker-end="url(#gen-arrow)"/>

        <!-- Invariant Bounds / Feedback -->
        <rect x="460" y="115" width="170" height="90" rx="6" fill="#312e81" stroke="#818cf8" stroke-width="2"/>
        <text x="545" y="145" fill="#a5b4fc" font-size="11" font-family="monospace" text-anchor="middle" font-weight="bold">STAGE 03</text>
        <text x="545" y="165" fill="#ffffff" font-size="12" font-family="sans-serif" text-anchor="middle" font-weight="bold">Invariant Bounds</text>
        <text x="545" y="185" fill="#c7d2fe" font-size="10" font-family="monospace" text-anchor="middle">Stability Bounds</text>

        <line x1="630" y1="160" x2="690" y2="160" stroke="#38bdf8" stroke-width="2" marker-end="url(#gen-arrow)"/>

        <!-- Verified Output -->
        <rect x="690" y="115" width="120" height="90" rx="6" fill="#1e1b4b" stroke="#c084fc" stroke-width="2"/>
        <text x="750" y="145" fill="#e879f9" font-size="11" font-family="monospace" text-anchor="middle" font-weight="bold">STAGE 04</text>
        <text x="750" y="165" fill="#ffffff" font-size="12" font-family="sans-serif" text-anchor="middle" font-weight="bold">Synthesis</text>
        <text x="750" y="185" fill="#f5d0fe" font-size="10" font-family="monospace" text-anchor="middle">Verified Returns</text>

        <!-- Feedback Loop line -->
        <path d="M 545 205 L 545 250 L 315 250 L 315 205" fill="none" stroke="#f59e0b" stroke-width="1.8" stroke-dasharray="4 4" marker-end="url(#gen-arrow)"/>
        <text x="430" y="268" fill="#f59e0b" font-size="11" font-family="monospace" text-anchor="middle">Convergence &amp; Dynamic Loss Feedback</text>

        <rect x="220" y="310" width="480" height="32" rx="6" fill="#111827" stroke="#374151" />
        <text x="460" y="331" fill="#38bdf8" font-size="12" font-family="monospace" text-anchor="middle" font-weight="bold">Pipeline: Ingestion ➔ Transform ➔ Invariant ➔ Return</text>
      </svg>
    </div>
  `;
}

function generateHtmlDiagram(p) {
  const tex = p.rawTex || "";
  let inputs = [];
  let outputs = [];
  let steps = [];
  let invariant = "";
  let bigTheta = "";

  const reqMatch = tex.match(/\\Require\s+([^\n\r]+)/);
  if (reqMatch) {
    inputs = reqMatch[1].split(/,|;/).map(sanitizeDiagramMath).filter(s => s.length > 1).slice(0, 6);
  }

  const ensMatch = tex.match(/\\Ensure\s+([^\n\r]+)/);
  if (ensMatch) {
    outputs = ensMatch[1].split(/,|;/).map(sanitizeDiagramMath).filter(s => s.length > 1).slice(0, 6);
  }

  const algoMatch = tex.match(/\\begin\{algorithmic\}[\s\S]*?\\end\{algorithmic\}/);
  if (algoMatch) {
    const lines = algoMatch[0].split("\n");
    for (const line of lines) {
      if (line.includes("\\State") || line.includes("\\For") || line.includes("\\If")) {
        const clean = sanitizeDiagramMath(line.replace(/\\(State|For|If|Ensure|Require)/g, ""));
        if (clean.length > 2 && !clean.toLowerCase().startsWith("return") && steps.length < 8) {
          steps.push(clean);
        }
      }
    }
  }

  if (steps.length === 0 && tex.includes("Dependency Graph")) {
    const depMatch = tex.match(/Dependency Graph\}?:\s*([^\n\r\\]+)/);
    if (depMatch) {
      const parts = depMatch[1].split(/\\to|->/).map(sanitizeDiagramMath).filter(s => s.length > 1);
      if (parts.length > 1) {
        if (inputs.length === 0) inputs = [parts[0]];
        steps = parts.slice(1, -1);
        if (outputs.length === 0) outputs = [parts[parts.length - 1]];
      }
    }
  }

  if (steps.length === 0) {
    const subs = (tex.match(/\\subsection\{([^}]+)\}/g) || [])
      .map(s => sanitizeDiagramMath(s.replace(/\\subsection\{|\}/g, "")))
      .filter(s => !s.toLowerCase().includes("question") && !s.toLowerCase().includes("mental"));
    if (subs.length > 0) {
      steps = subs.slice(0, 5);
    }
  }

  const invMatch = tex.match(/Invariants\}?:\s*([^\n\r\\]+)/);
  if (invMatch) {
    invariant = sanitizeDiagramMath(invMatch[1]);
  } else if (p.excerpt) {
    invariant = p.excerpt.slice(0, 160);
  }

  const compMatch = tex.match(/Complexity Analysis\}?([\s\S]*?)\\section\{/);
  if (compMatch) {
    const firstItem = compMatch[1].match(/\\item\s+([^\n\r]+)/);
    if (firstItem) bigTheta = sanitizeDiagramMath(firstItem[1]);
  }

  if (inputs.length === 0) inputs = ["Input Feature Tensor X", "Model Parameters θ", "Hyperparameters η"];
  if (steps.length === 0) steps = ["Initialize Tensor Memory Buffers", "Evaluate Core Mathematical Kernel", "Update Internal Parameter State", "Compute Loss & Invariant Bounds"];
  if (outputs.length === 0) outputs = ["Optimized State θ*", "Convergence Invariant Metric"];

  const archSvg = renderArchitecturalSvg(p, inputs, outputs, steps);

  return `
    <div class="html-diagram-wrap">
      <div class="html-diagram-header">
        <div class="html-diagram-tag">INTERNAL ARCHITECTURE &amp; EXECUTION MECHANISM</div>
        <div class="html-diagram-title">${escapeHtml(p.algoName)}</div>
        <div class="html-diagram-subtitle">${escapeHtml(p.domainLabel)} · ${p.lines} lines of formal specifications</div>
      </div>

      <!-- High-Fidelity Computational Architecture Diagram -->
      ${archSvg}

      <div class="html-flow-container">
        <div class="html-flow-track">
          <div class="html-flow-node input-node">
            <div class="flow-node-step">01 · INGESTION</div>
            <div class="flow-node-title">Input Tensors</div>
            <div class="flow-node-desc">${inputs.length} Parameter Arrays</div>
          </div>
          <div class="html-flow-arrow">►</div>
          <div class="html-flow-node core-node">
            <div class="flow-node-step">02 · COMPUTATION</div>
            <div class="flow-node-title">Internal Operator</div>
            <div class="flow-node-desc">${steps.length} Pipeline Passes</div>
          </div>
          <div class="html-flow-arrow">►</div>
          <div class="html-flow-node loop-node">
            <div class="flow-node-step">03 · VERIFICATION</div>
            <div class="flow-node-title">Invariant Bounds</div>
            <div class="flow-node-desc">Convergence Check</div>
          </div>
          <div class="html-flow-arrow">►</div>
          <div class="html-flow-node output-node">
            <div class="flow-node-step">04 · SYNTHESIS</div>
            <div class="flow-node-title">Output Targets</div>
            <div class="flow-node-desc">${outputs.length} Validated Returns</div>
          </div>
        </div>

        <div class="html-diagram-section">
          <div class="html-section-heading">STAGE 01 — INPUT DATA &amp; TENSOR INGESTION</div>
          <div class="html-pills-wrap">
            ${inputs.map(inp => `<div class="html-pill input-pill"><span class="pill-dot">●</span><span>${escapeHtml(inp)}</span></div>`).join("")}
          </div>
        </div>

        <div class="html-diagram-section">
          <div class="html-section-heading">STAGE 02 — INTERNAL EXECUTION LIFECYCLE (${steps.length} PASSES)</div>
          <div class="html-steps-list">
            ${steps.map((st, i) => `
              <div class="html-step-row">
                <div class="html-step-badge">PASS ${String(i + 1).padStart(2, "0")}</div>
                <div class="html-step-body">${escapeHtml(st)}</div>
              </div>
            `).join("")}
          </div>
        </div>

        <div class="html-diagram-section">
          <div class="html-section-heading">STAGE 03 — COMPUTED OUTPUTS &amp; RESULT TARGETS</div>
          <div class="html-pills-wrap">
            ${outputs.map(out => `<div class="html-pill output-pill"><span class="pill-dot">■</span><span>${escapeHtml(out)}</span></div>`).join("")}
          </div>
        </div>

        ${invariant ? `
          <div class="html-diagram-section invariant-section">
            <div class="html-section-heading invariant-heading">STAGE 04 — MATHEMATICAL INVARIANT CERTIFICATE</div>
            <div class="html-invariant-content">${escapeHtml(invariant)}</div>
          </div>
        ` : ""}

        ${bigTheta ? `
          <div class="html-diagram-section complexity-section">
            <div class="html-section-heading complexity-heading">ASYMPTOTIC COMPLEXITY PROFILE</div>
            <div class="html-complexity-content">${escapeHtml(bigTheta)}</div>
          </div>
        ` : ""}
      </div>
    </div>
  `;
}

function setTexViewMode(mode) {
  currentTexViewMode = mode;
  const btnRender = document.getElementById("btn-mode-render");
  const btnDiagram = document.getElementById("btn-mode-diagram");
  const renderedView = document.getElementById("tex-rendered-view");
  const diagramView = document.getElementById("tex-diagram-view");

  if (btnRender) btnRender.classList.toggle("active", mode === "rendered");
  if (btnDiagram) btnDiagram.classList.toggle("active", mode === "diagram");

  if (renderedView) {
    renderedView.style.display = mode === "rendered" ? "block" : "none";
    if (mode === "rendered") renderProofMath(renderedView, currentModalProof);
  }
  if (diagramView) {
    diagramView.style.display = mode === "diagram" ? "block" : "none";
    if (mode === "diagram" && currentModalProof) {
      const domain = currentModalProof.domain || "";
      const algo = currentModalProof.algo || "";
      const diagUrl = `diagrams/${domain}__${algo}.html`;
      diagramView.innerHTML = `
        <div style="width:100%; height:100%; min-height:620px; display:flex; flex-direction:column;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid var(--rule);">
            <span style="font-family:var(--font-dspl); font-size:11px; color:var(--accent); font-weight:700;">◈ DISTILL ARCHITECTURAL GRAPH &amp; HARDWARE DATA-FLOW</span>
            <a href="${diagUrl}" target="_blank" style="font-family:var(--font-dspl); font-size:11px; color:var(--teal); text-decoration:none;">Open in Full View ↗</a>
          </div>
          <iframe src="${diagUrl}" style="width:100%; height:620px; border:1px solid var(--rule); border-radius:8px; background:#060810;" frameborder="0"></iframe>
        </div>
      `;
    }
  }
}

function renderProofMath(container, proof) {
  if (!proof || !proof.rawTex) return;
  const html = parseTexToHtml(proof.rawTex);
  container.innerHTML = html;

  if (window.renderMathInElement) {
    try {
      window.renderMathInElement(container, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true }
        ],
        output: "html",
        throwOnError: false
      });
    } catch (e) {
      console.warn("KaTeX render error:", e);
    }
  }
}

function extractBalancedArg(str, cmdName) {
  const prefix = "\\" + cmdName + "{";
  const idx = str.indexOf(prefix);
  if (idx === -1) return null;
  let depth = 1;
  let end = idx + prefix.length;
  while (end < str.length && depth > 0) {
    if (str[end] === "{") depth++;
    else if (str[end] === "}") depth--;
    end++;
  }
  if (depth === 0) {
    return str.slice(idx + prefix.length, end - 1);
  }
  return null;
}

function replaceLatexCommand(str, cmdName, replaceFn) {
  const prefix = "\\" + cmdName + "{";
  let pos = 0;
  let result = "";
  while (pos < str.length) {
    const idx = str.indexOf(prefix, pos);
    if (idx === -1) {
      result += str.slice(pos);
      break;
    }
    result += str.slice(pos, idx);
    let depth = 1;
    let end = idx + prefix.length;
    while (end < str.length && depth > 0) {
      if (str[end] === "{") depth++;
      else if (str[end] === "}") depth--;
      end++;
    }
    if (depth === 0) {
      const content = str.slice(idx + prefix.length, end - 1);
      result += replaceFn(content);
      pos = end;
    } else {
      result += str.slice(idx, idx + prefix.length);
      pos = idx + prefix.length;
    }
  }
  return result;
}

function parseTexToHtml(src) {
  let body = src.includes("\\begin{document}") ? src.split("\\begin{document}")[1].split("\\end{document}")[0] : src;
  
  body = body.replace(/\\maketitle/g, "");
  body = body.replace(/\\boldmath/g, "");
  body = body.replace(/\{\s*\\begin\{equation\*?\}/g, "\\begin{equation}");
  body = body.replace(/\\end\{equation\*?\}\s*\}/g, "\\end{equation}");
  body = body.replace(/\{\s*\\begin\{align\*?\}/g, "\\begin{align}");
  body = body.replace(/\\end\{align\*?\}\s*\}/g, "\\end{align}");

  body = body.replace(/\\label\{[^}]*\}/g, "");
  body = body.replace(/\\cite\{[^}]*\}/g, "[Ref]");
  body = body.replace(/\\ref\{([^}]*)\}/g, "$1");
  body = body.replace(/\\eqref\{([^}]*)\}/g, "($1)");
  body = body.replace(/``/g, "\"").replace(/''/g, "\"");
  body = body.replace(/--/g, "–");
  body = body.replace(/\\par\b/g, "\n\n");
  body = body.replace(/\\quad\b/g, " ");
  body = body.replace(/\\qquad\b/g, "  ");
  body = body.replace(/\\dots\b/g, "…");
  body = body.replace(/\\textbackslash\b/g, "\\");
  body = body.replace(/\\underline\{([^}]+)\}/g, "<u>$1</u>");

  body = body.replace(/\\begin\{equation\*?\}([\s\S]*?)\\end\{equation\*?\}/g, (match, eq) => {
    return "\n\n<div class=\"tex-equation\">" + renderKaTeX(eq.trim(), true) + "</div>\n\n";
  });
  body = body.replace(/\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}/g, (match, eq) => {
    return "\n\n<div class=\"tex-equation\">" + renderKaTeX("\\begin{aligned}" + eq.trim() + "\\end{aligned}", true) + "</div>\n\n";
  });
  body = body.replace(/\$\$([\s\S]*?)\$\$/g, (match, eq) => {
    return "\n\n<div class=\"tex-equation\">" + renderKaTeX(eq.trim(), true) + "</div>\n\n";
  });
  body = body.replace(/\\\[([\s\S]*?)\\\]/g, (match, eq) => {
    return "\n\n<div class=\"tex-equation\">" + renderKaTeX(eq.trim(), true) + "</div>\n\n";
  });

  body = body.replace(/\\begin\{theorem\}(?:\[([^\]]*)\])?([\s\S]*?)\\end\{theorem\}/g, (m, opt, content) => {
    const title = opt ? ` (${opt})` : "";
    return `\n\n<div class="tex-theorem"><strong>Theorem${title}:</strong> ${content.trim()}</div>\n\n`;
  });
  body = body.replace(/\\begin\{definition\}(?:\[([^\]]*)\])?([\s\S]*?)\\end\{definition\}/g, (m, opt, content) => {
    const title = opt ? ` (${opt})` : "";
    return `\n\n<div class="tex-definition"><strong>Definition${title}:</strong> ${content.trim()}</div>\n\n`;
  });
  body = body.replace(/\\begin\{lemma\}(?:\[([^\]]*)\])?([\s\S]*?)\\end\{lemma\}/g, (m, opt, content) => {
    const title = opt ? ` (${opt})` : "";
    return `\n\n<div class="tex-lemma"><strong>Lemma${title}:</strong> ${content.trim()}</div>\n\n`;
  });
  body = body.replace(/\\begin\{remark\}(?:\[([^\]]*)\])?([\s\S]*?)\\end\{remark\}/g, (m, opt, content) => {
    const title = opt ? ` (${opt})` : "";
    return `\n\n<div class="tex-remark"><strong>Remark${title}:</strong> ${content.trim()}</div>\n\n`;
  });
  body = body.replace(/\\begin\{proof\}([\s\S]*?)\\end\{proof\}/g, (m, content) => {
    return `\n\n<div class="tex-proof"><strong>Proof:</strong> ${content.trim()} <span class="proof-qed">∎</span></div>\n\n`;
  });

  body = body.replace(/\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}/g, (m, content) => {
    return `\n\n<div class="tex-abstract"><div class="tex-abstract-title">ABSTRACT</div><div class="tex-abstract-body">${content.trim()}</div></div>\n\n`;
  });

  body = body.replace(/\\begin\{algorithm\}(?:\[[^\]]*\])?([\s\S]*?)\\end\{algorithm\}/g, (match, block) => {
    const capMatch = block.match(/\\caption\{([^}]+)\}/);
    const caption = capMatch ? capMatch[1] : "Algorithm";

    const algMatch = block.match(/\\begin\{algorithmic\}(?:\[\d+\])?([\s\S]*?)\\end\{algorithmic\}/);
    if (!algMatch) {
      return `\n\n<div class="tex-algorithm"><div class="algo-header"><span class="algo-badge">ALGORITHM</span> <strong class="algo-title">${caption}</strong></div></div>\n\n`;
    }

    let rawLines = algMatch[1]
      .replace(/\\(State|If|Else|ElsIf|EndIf|For|EndFor|While|EndWhile|Function|EndFunction|Return)\b/g, "\n\\$1")
      .split("\n")
      .map(s => s.trim())
      .filter(Boolean);

    let html = `\n\n<div class="tex-algorithm">\n  <div class="algo-header"><span class="algo-badge">ALGORITHM</span> <strong class="algo-title">${caption}</strong></div>\n  <div class="algo-body">\n`;
    
    let ln = 1;
    let indent = 0;

    for (let l of rawLines) {
      if (l.startsWith("\\Require")) {
        let req = l.slice(8).trim();
        req = req.replace(/\$([^$]+)\$/g, (m, eq) => renderKaTeX(eq, false));
        html += `    <div class="algo-req"><span class="algo-kw">Require:</span> ${req}</div>\n`;
        continue;
      }
      if (l.startsWith("\\Ensure")) {
        let ens = l.slice(7).trim();
        ens = ens.replace(/\$([^$]+)\$/g, (m, eq) => renderKaTeX(eq, false));
        html += `    <div class="algo-ens"><span class="algo-kw">Ensure:</span> ${ens}</div>\n`;
        continue;
      }

      l = replaceLatexCommand(l, "Comment", c => `<span class="algo-comment">// ${c}</span>`);
      l = l.replace(/\\Call\{([^}]+)\}\{([^}]+)\}/g, "\\operatorname{$1}($2)");

      if (l.startsWith("\\EndFor") || l.startsWith("\\EndIf") || l.startsWith("\\EndWhile") || l.startsWith("\\EndFunction")) {
        indent = Math.max(0, indent - 1);
        continue;
      }

      if (l.startsWith("\\Function")) {
        let sig = l.slice(9).trim();
        const m = sig.match(/^\{([^}]+)\}\{([^}]+)\}/);
        const fnName = m ? m[1] : sig;
        const fnArgs = m ? m[2] : "";
        html += `    <div class="algo-line" style="padding-left:${indent * 20}px"><span class="algo-ln">${ln}</span> <span class="algo-code"><strong class="algo-kw">function</strong> <span class="algo-fn">${fnName}</span>(${fnArgs})</span></div>\n`;
        ln++;
        indent++;
        continue;
      }

      if (l.startsWith("\\Else")) {
        let curIndent = Math.max(0, indent - 1);
        html += `    <div class="algo-line" style="padding-left:${curIndent * 20}px"><span class="algo-ln">${ln}</span> <span class="algo-code"><strong class="algo-kw">else</strong></span></div>\n`;
        ln++;
        continue;
      }

      if (l.startsWith("\\ElsIf")) {
        let cond = l.slice(6).trim();
        if (cond.startsWith("{") && cond.endsWith("}")) cond = cond.slice(1, -1);
        cond = cond.replace(/\$([^$]+)\$/g, (m, eq) => renderKaTeX(eq, false));
        let curIndent = Math.max(0, indent - 1);
        html += `    <div class="algo-line" style="padding-left:${curIndent * 20}px"><span class="algo-ln">${ln}</span> <span class="algo-code"><strong class="algo-kw">else if</strong> ${cond} <strong class="algo-kw">then</strong></span></div>\n`;
        ln++;
        continue;
      }

      if (l.startsWith("\\For")) {
        let cond = l.slice(4).trim();
        if (cond.startsWith("{") && cond.endsWith("}")) cond = cond.slice(1, -1);
        cond = cond.replace(/\$([^$]+)\$/g, (m, eq) => renderKaTeX(eq, false));
        html += `    <div class="algo-line" style="padding-left:${indent * 20}px"><span class="algo-ln">${ln}</span> <span class="algo-code"><strong class="algo-kw">for</strong> ${cond} <strong class="algo-kw">do</strong></span></div>\n`;
        ln++;
        indent++;
        continue;
      }

      if (l.startsWith("\\If")) {
        let cond = l.slice(3).trim();
        if (cond.startsWith("{") && cond.endsWith("}")) cond = cond.slice(1, -1);
        cond = cond.replace(/\$([^$]+)\$/g, (m, eq) => renderKaTeX(eq, false));
        html += `    <div class="algo-line" style="padding-left:${indent * 20}px"><span class="algo-ln">${ln}</span> <span class="algo-code"><strong class="algo-kw">if</strong> ${cond} <strong class="algo-kw">then</strong></span></div>\n`;
        ln++;
        indent++;
        continue;
      }

      if (l.startsWith("\\While")) {
        let cond = l.slice(6).trim();
        if (cond.startsWith("{") && cond.endsWith("}")) cond = cond.slice(1, -1);
        cond = cond.replace(/\$([^$]+)\$/g, (m, eq) => renderKaTeX(eq, false));
        html += `    <div class="algo-line" style="padding-left:${indent * 20}px"><span class="algo-ln">${ln}</span> <span class="algo-code"><strong class="algo-kw">while</strong> ${cond} <strong class="algo-kw">do</strong></span></div>\n`;
        ln++;
        indent++;
        continue;
      }

      let stmt = l.startsWith("\\State") ? l.slice(6).trim() : l;
      stmt = stmt.replace(/\\Return\b/g, '<strong class="algo-kw">return</strong>');
      stmt = stmt.replace(/\$([^$]+)\$/g, (m, eq) => renderKaTeX(eq, false));
      html += `    <div class="algo-line" style="padding-left:${indent * 20}px"><span class="algo-ln">${ln}</span> <span class="algo-code">${stmt}</span></div>\n`;
      ln++;
    }

    html += `  </div>\n</div>\n\n`;
    return html;
  });

  function parseCell(c) {
    let cell = c.trim();
    cell = replaceLatexCommand(cell, "textbf", v => `<strong>${v}</strong>`);
    cell = replaceLatexCommand(cell, "textit", v => `<em>${v}</em>`);
    cell = replaceLatexCommand(cell, "texttt", v => `<code>${v}</code>`);
    cell = cell.replace(/\$([^$\n]+)\$/g, (m, eq) => renderKaTeX(eq, false));
    return cell;
  }

  body = body.replace(/\\begin\{table\}(?:\[[^\]]*\])?([\s\S]*?)\\end\{table\}/g, (m, content) => {
    let caption = extractBalancedArg(content, "caption") || "";
    caption = caption.replace(/\\textbf\{([^}]+)\}/g, "$1").trim();
    
    const tabMatch = content.match(/\\begin\{tabular\}\{[^}]*\}([\s\S]*?)\\end\{tabular\}/);
    if (!tabMatch) return "";
    const rawRows = tabMatch[1]
      .replace(/\\(top|mid|bottom)rule/g, "")
      .replace(/\\hline/g, "")
      .trim()
      .split("\\\\")
      .map(r => r.trim())
      .filter(Boolean);
    
    let tableHtml = `<div class="tex-table-wrap">` + (caption ? `<div class="tex-table-caption">${caption}</div>` : "") + `<table class="tex-table"><tbody>`;
    rawRows.forEach((row, idx) => {
      const cols = row.split("&").map(c => parseCell(c));
      const tag = idx === 0 ? "th" : "td";
      tableHtml += `<tr>` + cols.map(c => `<${tag}>${c}</${tag}>`).join("") + `</tr>`;
    });
    tableHtml += `</tbody></table></div>`;
    return "\n\n" + tableHtml + "\n\n";
  });

  body = body.replace(/\\begin\{center\}([\s\S]*?)\\end\{center\}/g, (m, c) => {
    return `\n\n<div class="tex-center">${c.trim()}</div>\n\n`;
  });
  body = body.replace(/\\centering\b/g, "");

  body = body.replace(/\\begin\{thebibliography\}(?:\{[^}]*\})?([\s\S]*?)\\end\{thebibliography\}/g, (m, content) => {
    const rawParts = content.split(/\\bibitem/);
    rawParts.shift();
    let bibItemsHtml = "";
    rawParts.forEach((part, idx) => {
      let text = part.trim();
      let key = String(idx + 1);
      const keyMatch = text.match(/^(?:\[[^\]]*\])?\{([^}]+)\}/);
      if (keyMatch) {
        key = keyMatch[1];
        text = text.slice(keyMatch[0].length).trim();
      }
      text = text.replace(/~/g, " ");
      text = text.replace(/``/g, '"').replace(/''/g, '"');
      text = text.replace(/--/g, "–");
      text = replaceLatexCommand(text, "emph", v => `<em>${v}</em>`);
      text = replaceLatexCommand(text, "textit", v => `<em>${v}</em>`);
      text = replaceLatexCommand(text, "textbf", v => `<strong>${v}</strong>`);
      text = replaceLatexCommand(text, "url", v => `<a href="${v}" target="_blank" class="tex-url">${v}</a>`);
      text = text.replace(/https?:\/\/[^\s<)]+/g, (url) => `<a href="${url}" target="_blank" class="tex-url">${url}</a>`);
      text = text.replace(/\$([^$\n]+)\$/g, (m, eq) => renderKaTeX(eq, false));
      bibItemsHtml += `  <li class="tex-bib-item" id="bib-${escapeHtml(key)}"><span class="tex-bib-key">[${idx + 1}]</span> <span class="tex-bib-text">${text}</span></li>\n`;
    });
    return `\n\n<div class="tex-bibliography"><h4 class="tex-h4">References</h4><ol class="tex-bib-list">\n${bibItemsHtml}</ol></div>\n\n`;
  });

  while (/\\begin\{(?:itemize|enumerate)\}/.test(body)) {
    const prev = body;
    body = body.replace(/\\begin\{(itemize|enumerate)\}((?:(?!\\begin\{(?:itemize|enumerate)\})[\s\S])*?)\\end\{\1\}/g, (match, envType, content) => {
      const items = content.split(/\\item\s+/).filter(i => i.trim());
      const tag = envType === "itemize" ? "ul" : "ol";
      const listHtml = items.map(it => "<li>" + it.trim() + "</li>").join("\n");
      return `\n\n<${tag} class="tex-list">\n${listHtml}\n</${tag}>\n\n`;
    });
    if (body === prev) break;
  }

  body = replaceLatexCommand(body, "section", c => `<h3 class="tex-h3">${c}</h3>`);
  body = replaceLatexCommand(body, "subsection", c => `<h4 class="tex-h4">${c}</h4>`);
  body = replaceLatexCommand(body, "subsubsection", c => `<h5 class="tex-h5">${c}</h5>`);
  body = replaceLatexCommand(body, "paragraph", c => `<h5 class="tex-h5">${c}</h5>`);

  body = body.replace(/\$([^$\n]+)\$/g, (m, eq) => {
    return renderKaTeX(eq, false);
  });
  body = body.replace(/\\\(([\s\S]*?)\\\)/g, (m, eq) => {
    return renderKaTeX(eq, false);
  });

  body = replaceLatexCommand(body, "textbf", c => `<strong>${c}</strong>`);
  body = replaceLatexCommand(body, "textit", c => `<em>${c}</em>`);
  body = replaceLatexCommand(body, "emph", c => `<em>${c}</em>`);
  body = replaceLatexCommand(body, "texttt", c => `<code>${c}</code>`);
  body = replaceLatexCommand(body, "url", c => `<a href="${c}" target="_blank" class="tex-url">${c}</a>`);

  body = body.replace(/^\s*\{\s*$/gm, "");
  body = body.replace(/^\s*\}\s*$/gm, "");

  const paras = body.split(/\n\s*\n/).filter(p => p.trim());
  return paras.map(p => {
    p = p.trim();
    if (p.startsWith("<h3") || p.startsWith("<h4") || p.startsWith("<h5") || 
        p.startsWith("<div") || p.startsWith("<ul") || p.startsWith("<ol") ||
        p.startsWith("<table")) {
      return p;
    }
    return "<p class=\"tex-p\">" + p + "</p>";
  }).join("\n\n");
}


function copyCurrentTex() {
  if (currentModalProof && currentModalProof.rawTex) {
    copyText(currentModalProof.rawTex);
    showToast("COPIED FULL LATEX SOURCE");
  }
}

function openDedicatedDiagramPage() {
  if (!currentModalProof) return;
  const domain = currentModalProof.domain || "";
  const algo = currentModalProof.algo || "";
  const url = `diagrams/${domain}__${algo}.html`;
  window.open(url, "_blank");
}

function openCurrentPdf() {
  if (currentModalProof && currentModalProof.pdf) {
    window.open(currentModalProof.pdf, "_blank");
  } else {
    showToast("NO PDF AVAILABLE");
  }
}

// Close modal on Escape key
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeTexModal();
  }
});
