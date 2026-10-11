# Neural Network Foundation & Frontier Engine (`algos/nn`)

[![Algorithms](https://img.shields.io/badge/Algorithms-Core%20Catalog-success.svg)](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn)
[![Domains](https://img.shields.io/badge/Domains-29%20Architectural%20Slices-blueviolet.svg)](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-0%20External%20(Pure%20Python)-brightgreen.svg)](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn)
[![Contracts](https://img.shields.io/badge/Contracts-26--Key%20YAML%20Strict-blue.svg)](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn)
[![LaTeX & PDF](https://img.shields.io/badge/Math%20PDFs-Formal%20Proofs%20Compiled-orange.svg)](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn)
[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-100%25%20Passing-brightgreen.svg)](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/tests/unit/algos)

> **Complete, Deterministic Scalar Neural Network Engine (Core Component Catalog)**  
> *Every mathematical operator, transformer micro-architecture, distributed 5D parallel scaling primitive, post-training alignment recipe, PEFT adapter, and inference optimization algorithm implemented from first principles with zero external dependencies (`torch`, `numpy` free).*

---

## 1. System Architecture: How All Components Combine Into One

The algorithms in this repository are not isolated routines; they form an interconnected, production-grade computational graph spanning the entire life cycle of frontier foundation models and Large Language Models (LLMs).

Below is the complete architectural specification across:
1. **Master End-to-End System Blueprint** (Unified Lifecycle Data Flow)
2. **Control Plane Architecture** (High-Level Cluster Orchestration & Low-Level Per-Step Adaptive Controllers)
3. **Data Plane & Distributed Messaging Topology** (5D Interconnect Fabrics & Intra-Block Tensor Pipelines)
4. **Post-Training Alignment & Verifiable Reasoning Engine** (Dual-Track Synthesis Loop)
5. **Production Inference Serving & Constrained Generation Engine**

---

### 1.1 Master End-to-End System Blueprint

The diagram below illustrates the end-to-end journey of building, scaling, aligning, compressing, and serving a frontier LLM using the components in this repository:

```mermaid
flowchart TD
    classDef data fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef arch fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef scale fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;
    classDef opt fill:#1a3300,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;
    classDef align fill:#4a154b,stroke:#f5c2e7,stroke-width:2px,color:#cdd6f4;
    classDef peft fill:#2e3440,stroke:#88c0d0,stroke-width:2px,color:#cdd6f4;
    classDef serve fill:#204040,stroke:#94e2d5,stroke-width:2px,color:#cdd6f4;

    %% -------------------------------------------------------------------------
    %% PHASE 1: DATA CURATION & PREPARATION
    %% -------------------------------------------------------------------------
    subgraph P1 ["Phase 1: Data Curation, Filtering & Sequence Preparation"]
        RawCorpora["Raw Pre-Training Web, Code & Math Corpora"]:::data
        SynthGen["Synthetic Candidate Stream"]:::data

        SynthGen --> FilterQual["Synthetic Quality Gate & Token Length Bounds"]:::data
        FilterQual --> FilterDedup["Incremental Jaccard Deduplication Hash Pool"]:::data
        RawCorpora --> DoReMiProxy["DoReMi Proxy vs Reference Excess Loss Engine"]:::data
        FilterDedup --> DoReMiProxy

        DoReMiProxy --> SimplexWeights["Dynamic Simplex Mixture Weights alpha_d"]:::data
        SimplexWeights --> SeqPack["Sequence Packing with Best-Fit Decreasing"]:::data
        SeqPack --> DocMask["Causal 2D Block-Diagonal Document Masking"]:::data
    end

    %% -------------------------------------------------------------------------
    %% PHASE 2: TRANSFORMER MICRO-ARCHITECTURE
    %% -------------------------------------------------------------------------
    subgraph P2 ["Phase 2: Transformer Block Micro-Architecture & Dynamics"]
        DocMask --> InTokens["Packed Token Sequence [Batch, SeqLen]"]:::arch
        InTokens --> EmbLookup["Embedding Lookup & High-Dimensional Projections"]:::arch
        EmbLookup --> PreNorm["Pre-Attention RMSNorm Layer Normalization"]:::arch

        PreNorm --> QKVProj["Linear Q, K, V Projections"]:::arch
        QKVProj --> RoPE["Rotary Position Embeddings (RoPE)"]:::arch
        QKVProj --> YaRN["Context Extension (YaRN & LongRoPE)"]:::arch

        RoPE --> AttnEngine["FlashAttention Causal Tiled Computation"]:::arch
        YaRN --> AttnEngine
        AttnEngine --> AttnSinks["Attention Sinks & Streaming Context Preservation"]:::arch

        AttnSinks --> OutProj["Attention Output Projection Matrix"]:::arch
        OutProj --> ResDrop1["Residual Addition + DropPath Stochastic Depth"]:::arch

        ResDrop1 --> PostNorm["Pre-FFN RMSNorm Normalization"]:::arch
        PostNorm --> MoERouter["Top-K Softmax Mixture-of-Experts Router"]:::arch

        MoERouter --> SwiGLU["SwiGLU Gated Feedforward Expert Networks"]:::arch
        SwiGLU --> ResDrop2["Residual Addition + Highway Connections"]:::arch
    end

    %% -------------------------------------------------------------------------
    %% PHASE 3: 5D DISTRIBUTED TRAINING AT SCALE
    %% -------------------------------------------------------------------------
    subgraph P3 ["Phase 3: 5D Distributed Cluster Scaling & Hardware Execution"]
        ResDrop2 --> MegatronTP["Tensor Parallelism (Column & Row Linear All-Reduce)"]:::scale
        MegatronTP --> RingCP["Context Parallelism (Ring Attention Token Exchange)"]:::scale
        RingCP --> PipePP["Pipeline Parallelism (1F1B Interleaved Schedule)"]:::scale
        PipePP --> ExpertEP["Expert Parallelism (All-to-All Token Dispatch)"]:::scale

        ExpertEP --> ZeRO3["ZeRO-3 / FSDP Fully Sharded Parameters & Optimizers"]:::scale
        ZeRO3 --> FloatPrecision["Mixed Precision (BF16 & FP8 Tensor Cores)"]:::scale
        FloatPrecision --> Offload["Activation Checkpointing & Host CPU Offloading"]:::scale
    end

    %% -------------------------------------------------------------------------
    %% PHASE 4: OPTIMIZATION, LOSS & RECOVERY
    %% -------------------------------------------------------------------------
    subgraph P4 ["Phase 4: Objective Loss, Autodiff & Gradient Dynamics"]
        Offload --> CrossEntropy["Cross-Entropy Loss with Label Smoothing"]:::opt
        CrossEntropy --> ZLoss["Logit Stabilization Z-Loss Regularization"]:::opt

        ZLoss --> AutodiffVJP["Reverse-Mode Autodiff Vector-Jacobian Products"]:::opt
        AutodiffVJP --> GradAccum["Gradient Accumulation Micro-Batch Buffer"]:::opt
        GradAccum --> NoiseScale["Critical Batch Size & Gradient Noise Scale Monitor"]:::opt

        NoiseScale --> AdamW["Adaptive Optimizers (AdamW, Adafactor, Lion, Muon)"]:::opt
        AdamW --> MuP["Maximal Update Parametrization (muP Width Transfer)"]:::opt
        MuP --> WSD["Warmup-Stable-Decay (WSD) Learning Rate Schedule"]:::opt

        WSD --> SpikeDetect["Loss Spike Detection & Health Diagnostics"]:::opt
        SpikeDetect --> ElasticDCP["Distributed Checkpointing & Elastic Resharding"]:::opt
    end

    %% -------------------------------------------------------------------------
    %% PHASE 5: POST-TRAINING ALIGNMENT & REASONING
    %% -------------------------------------------------------------------------
    subgraph P5 ["Phase 5: Post-Training Alignment, Reasoning & Self-Play"]
        ElasticDCP --> SFT["Supervised Fine-Tuning (SFT)"]:::align
        
        SFT --> TrackPrefs["Human Preference Alignment Track"]:::align
        SFT --> TrackReason["Verifiable Mathematical Reasoning Track"]:::align

        TrackPrefs --> RewardModel["Bradley-Terry Pairwise Reward Modeling"]:::align
        RewardModel --> RLHF["RLHF PPO Policy Gradient with KL Divergence Penalty"]:::align
        TrackPrefs --> DPO["Direct Alignment: DPO, SimPO & KTO"]:::align

        TrackReason --> GRPO["Group Relative Policy Optimization (GRPO)"]:::align
        TrackReason --> RLVR["Reinforcement Learning with Verifiable Rewards (RLVR)"]:::align
        RLVR --> PRM["Process Reward Models (Step-Level Credit PRM)"]:::align
        PRM --> ConstAI["Constitutional AI Self-Critique & Best-of-N Rejection"]:::align
    end

    %% -------------------------------------------------------------------------
    %% PHASE 6: PEFT, MERGING, DISTILLATION & PRUNING
    %% -------------------------------------------------------------------------
    subgraph P6 ["Phase 6: Parameter-Efficiency, Compression & Hardware Pruning"]
        ConstAI --> LoRA["LoRA, QLoRA NF4 & DoRA Low-Rank Adapters"]:::peft
        DPO --> LoRA
        RLHF --> LoRA

        LoRA --> ModelMerge["TIES & DARE Multi-Task Model Merging"]:::peft
        ModelMerge --> Distill["Knowledge Distillation (Soft Targets & On-Policy)"]:::peft
        Distill --> Continual["Continual Learning (EWC & Experience Replay)"]:::peft

        Continual --> Wanda["Wanda One-Shot Activation-Weighted Pruning"]:::peft
        Wanda --> Sparse24["Semi-Structured 2:4 Hardware Sparse Tensor Cores"]:::peft
    end

    %% -------------------------------------------------------------------------
    %% PHASE 7: ULTRA-FAST INFERENCE DECODING
    %% -------------------------------------------------------------------------
    subgraph P7 ["Phase 7: Production Inference Serving & Constrained Generation"]
        Sparse24 --> PagedKV["PagedAttention Virtual Memory KV Cache Pool"]:::serve
        PagedKV --> FlashDecode["FlashDecoding Parallel Split-K Attention Reduction"]:::serve
        FlashDecode --> Speculative["Speculative Draft-Verify Accelerated Sampling"]:::serve
        Speculative --> GrammarEngine["Constrained CFG & JSON Grammar Finite State Machine"]:::serve
        GrammarEngine --> Contrastive["Contrastive Search & Min-P Temperature Sampling"]:::serve
        Contrastive --> FinalTokens["Deterministic, Aligned, Verifiable Generation"]:::serve
    end
```

---

### 1.1.1 Phase 1 Detailed Architecture: Data Ingestion, Mixture Optimization & Packing

```mermaid
flowchart TD
    classDef source fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef filter fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef mixture fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;
    classDef pack fill:#1a3300,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;

    subgraph CurationTrack ["Data Quality & Deduplication"]
        RawCorpus["Raw Web & Synthetic Corpus"]:::source --> FilterAlgo["Synthetic Data Quality Filtering<br/>• Threshold: quality >= 0.70<br/>• Bounds: 10 <= tokens <= 1000"]:::filter
        FilterAlgo --> DedupAlgo["N-Gram Jaccard Deduplication<br/>• Discards overlap > 0.80"]:::filter
    end

    subgraph MixtureTrack ["DoReMi Dynamic Mixture Optimization"]
        DedupAlgo --> DomainSplits["Domain Partitioning (Code, Math, Books, Web)"]:::source
        DomainSplits --> ProxyLoss["Proxy Model Domain Losses"]:::mixture
        DomainSplits --> RefLoss["Reference Baseline Losses"]:::mixture
        ProxyLoss & RefLoss --> ExcessLoss["Compute Excess Loss: Delta_d = L_proxy - L_ref"]:::mixture
        ExcessLoss --> MirrorDescent["DoReMi Exponentiated Gradient Mirror Descent<br/>• Updates domain sampling weights on simplex"]:::mixture
    end

    subgraph PackingTrack ["Sequence Packing & Mask Synthesis"]
        MirrorDescent --> WeightedSampler["Dynamic Domain Data Sampler"]:::mixture
        WeightedSampler --> DocSequences["Variable-Length Document Sequences"]:::source
        DocSequences --> BinPack["Sequence Packing (First-Fit Decreasing)<br/>• Capacity: max_sequence_length"]:::pack
        BinPack --> SegmentIDs["Generate Segment IDs & Monotonic Position IDs"]:::pack
        BinPack --> MaskGen["2D Block-Diagonal Causal Attention Mask Generator<br/>• Zero cross-document attention leakage"]:::pack
    end

    SegmentIDs & MaskGen --> DataLoaderOut["Distributed DataLoader Batch to Embedding Layer"]:::pack
```

---

### 1.1.2 Phase 2 Detailed Architecture: Transformer Block Micro-Architecture & Activation Dynamics

```mermaid
flowchart TD
    classDef act fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef norm fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef attn fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;
    classDef ffn fill:#1a3300,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;

    InputTokens["Packed Token Batch [Batch, SeqLen]"]:::act --> Emb["Vocabulary Embedding Lookup"]:::act
    Emb --> PreAttnNorm["Pre-Attention RMSNorm (Unit RMS Scaling)"]:::norm

    subgraph AttentionBlock ["Attention Projection & Context Dynamics"]
        PreAttnNorm --> QKV_GEMM["Q, K, V Linear Projections"]:::attn
        QKV_GEMM --> RoPE_Kernel["Rotary Position Embeddings (RoPE)"]:::attn
        QKV_GEMM --> YaRN_Kernel["YaRN / LongRoPE Context Window Extension"]:::attn
        RoPE_Kernel & YaRN_Kernel --> FlashKernel["FlashAttention Tiled SRAM Softmax Engine"]:::attn
        FlashKernel --> Sinks["Attention Sinks & Streaming Context Retainers"]:::attn
        Sinks --> OutProj["Attention Output Projection Matrix W_o"]:::attn
    end

    OutProj --> Skip1["Residual Addition + DropPath Stochastic Depth"]:::act
    Emb -.-> Skip1
    Skip1 --> PreFFNNorm["Pre-FFN RMSNorm Normalization"]:::norm

    subgraph ExpertFFNBlock ["Mixture-of-Experts & Feed-Forward Block"]
        PreFFNNorm --> MoERouter["Top-K Softmax Gating Router"]:::ffn
        MoERouter --> DispatchTokens["Token Dispatch across Specialized Experts"]:::ffn
        DispatchTokens --> SwiGLU_Experts["SwiGLU Expert Projections: (W_gate * x) * SiLU(W_up * x)"]:::ffn
        SwiGLU_Experts --> CombineTokens["Weighted Expert Token Aggregation"]:::ffn
        CombineTokens --> DownProj["Row Down-Projection W_down"]:::ffn
    end

    DownProj --> Skip2["Residual Addition + Highway Connection"]:::act
    Skip1 -.-> Skip2
    Skip2 --> BlockOut["Output Hidden Activations to Next Layer"]:::act
```

---

### 1.1.3 Phase 3 Detailed Architecture: 5D Distributed Cluster Scaling & Hardware Execution Mesh

```mermaid
flowchart TD
    classDef nvlink fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef ib fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef pipe fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;
    classDef zero fill:#1a3300,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;

    subgraph IntraNode ["1. Intra-Node NVLink Mesh: Tensor Parallelism (TP)"]
        ColGEMM["Column-Parallel GEMM: Sharded across 8 GPUs"]:::nvlink --> GeLUAct["GeLU / SwiGLU Local Activation"]:::nvlink
        GeLUAct --> RowGEMM["Row-Parallel GEMM: Sharded across 8 GPUs"]:::nvlink
        RowGEMM --> AllReduceTP["NVLink All-Reduce Sum Collective"]:::nvlink
    end

    subgraph ContextRing ["2. Inter-Node InfiniBand Ring: Context Parallelism (CP)"]
        AllReduceTP --> RingChunks["Split Sequence S into N_cp Chunks"]:::ib
        RingChunks --> RingShift["P2P Ring Send/Recv Key-Value Activations"]:::ib
        RingShift --> LocalAttn["Local Block-Causal Attention Computation"]:::ib
    end

    subgraph PipelineRack ["3. Inter-Rack Pipeline Parallelism (PP)"]
        LocalAttn --> PP1F1B["1F1B Interleaved Pipeline Schedule"]:::pipe
        PP1F1B --> StageForward["P2P Activation Forwarding to Next Stage"]:::pipe
        PP1F1B --> StageBackward["P2P Gradient Backpropagation to Prior Stage"]:::pipe
    end

    subgraph MemoryMesh ["4. Memory Optimization: ZeRO-3 & Precision"]
        StageForward & StageBackward --> ZeROSharding["ZeRO-3 / FSDP Sharding: Weights, Gradients, Optimizers"]:::zero
        ZeROSharding --> FP8Exec["FP8 E4M3 Forward / FP8 E5M2 Backward Tensor Cores"]:::zero
        FP8Exec --> ActOffload["Selective Activation Checkpointing & CPU Offloading"]:::zero
    end
```

---

### 1.1.4 Phase 4 Detailed Architecture: Objective Loss Optimization, Autodiff & Recovery Engine

```mermaid
flowchart TD
    classDef loss fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef grad fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef opt fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;
    classDef watchdog fill:#1a3300,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;

    ModelLogits["Model Output Logits [Batch, SeqLen, Vocab]"]:::loss --> CELoss["Cross-Entropy Loss with Label Smoothing"]:::loss
    ModelLogits --> ZLossCalc["Logit Stabilization Z-Loss Regularization"]:::loss
    CELoss & ZLossCalc --> TotalLoss["Total Optimization Objective"]:::loss

    subgraph AutodiffEngine ["Reverse-Mode Autodiff & Gradient Processing"]
        TotalLoss --> VJP["Reverse-Mode Autodiff Vector-Jacobian Products"]:::grad
        VJP --> Recomp["Activation Checkpointing Recomputation"]:::grad
        Recomp --> MicroBuf["Gradient Accumulation Micro-Batch Buffer"]:::grad
        MicroBuf --> GradSync["ZeRO-2 Reduce-Scatter Gradient Accumulation"]:::grad
    end

    subgraph DynamicsWatchdog ["Scale Dynamics & Anomaly Detection"]
        GradSync --> NoiseEstimator["Critical Batch Size Noise Scale Estimator: tr(Sigma) / norm(G)^2"]:::watchdog
        NoiseEstimator --> BatchTuner["Dynamic Micro-Batch & Step Size Tuner"]:::watchdog
        GradSync --> SpikeMonitor["Loss Spike & Gradient Norm Watchdog"]:::watchdog
        SpikeMonitor -->|"If Divergence Detected"| Rollback["Elastic Rollback to Prior Checkpoint"]:::watchdog
    end

    subgraph OptimizationStep ["Adaptive Parameter Updates"]
        BatchTuner --> WSD_Sched["Warmup-Stable-Decay (WSD) Learning Rate Schedule"]:::opt
        WSD_Sched --> MuP_Scale["Maximal Update Parametrization (muP) Scaling"]:::opt
        MuP_Scale --> AdamW_Step["AdamW Decoupled Weight Decay Optimizer"]:::opt
        AdamW_Step --> Checkpoint["Distributed Elastic Checkpointing & Persistence"]:::opt
    end
```

---

### 1.1.5 Phase 5 Detailed Architecture: Post-Training Dual-Track Alignment & Verifiable Reasoning Loop

```mermaid
flowchart TD
    classDef base fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef pref fill:#4a154b,stroke:#f5c2e7,stroke-width:2px,color:#cdd6f4;
    classDef reason fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;
    classDef merge fill:#204040,stroke:#94e2d5,stroke-width:2px,color:#cdd6f4;

    PretrainedBase["Pre-Trained Foundation Weights"]:::base --> SFT["Supervised Fine-Tuning (SFT) Core"]:::base

    subgraph PreferenceTrack ["Track A: Human Preference Alignment"]
        SFT --> PairData["Winning vs Losing Pairwise Comparisons"]:::pref
        PairData --> BT_Model["Bradley-Terry Reward Model Training"]:::pref
        BT_Model --> RLHF_PPO["RLHF PPO Policy Optimization + Adaptive KL Penalty"]:::pref
        PairData --> DPO_Algo["Direct Preference Optimization (DPO)"]:::pref
        PairData --> SimPO_KTO["SimPO Length Normalization & KTO Prospect Loss"]:::pref
    end

    subgraph ReasoningTrack ["Track B: Verifiable Mathematical & Code Reasoning"]
        SFT --> MathPrompts["Complex Reasoning Prompts (MATH, Code, Logic)"]:::reason
        MathPrompts --> RolloutSampler["Group Policy Rollouts: Sample G Outputs per Prompt"]:::reason
        RolloutSampler --> SandboxEnv["Deterministic Verification Sandbox (Python / Sympy / Tests)"]:::reason
        SandboxEnv --> GRPO_Algo["Group Relative Policy Optimization (Critic-Free Advantages)"]:::reason
        SandboxEnv --> RLVR_Algo["Reinforcement Learning with Verifiable Rewards (RLVR)"]:::reason
        RolloutSampler --> PRM_Model["Process Reward Models (Step-Level Credit PRM)"]:::reason
        PRM_Model --> ConstitutionalAI["Constitutional AI Self-Critique & Best-of-N Rejection"]:::reason
    end

    RLHF_PPO & DPO_Algo & SimPO_KTO --> PrefCheckpoint["Aligned Conversational Checkpoint"]:::pref
    GRPO_Algo & RLVR_Algo & ConstitutionalAI --> ReasonCheckpoint["Aligned Deep Reasoning Checkpoint"]:::reason

    PrefCheckpoint & ReasonCheckpoint --> MergeEngine["TIES & DARE Multi-Task Sign-Concordant Checkpoint Merging"]:::merge
    MergeEngine --> UnifiedFrontier["Unified Frontier Aligned Reasoning Model"]:::merge
```

---

### 1.1.6 Phase 6 Detailed Architecture: Parameter-Efficiency, Compression & Hardware Sparsification

```mermaid
flowchart TD
    classDef peft fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef distill fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef prune fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;
    classDef deploy fill:#1a3300,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;

    AlignedWeights["Aligned Frontier Checkpoint Weights"]:::peft --> LoRA_Family["Low-Rank Adapters: LoRA, QLoRA NF4, DoRA"]:::peft
    LoRA_Family --> TIES_Merge["TIES & DARE Multi-Task Weight Consolidation"]:::peft

    subgraph DistillContinual ["Knowledge Distillation & Continual Retention"]
        TIES_Merge --> TeacherStudent["Knowledge Distillation (Soft Targets & On-Policy Rollouts)"]:::distill
        TeacherStudent --> EWC_Consolidation["Elastic Weight Consolidation (EWC Fisher Information Matrix)"]:::distill
        EWC_Consolidation --> RetainedWeights["Memory-Preserved Compact Student Weights"]:::distill
    end

    subgraph PruneSparse ["One-Shot Pruning & Hardware Sparsity"]
        RetainedWeights --> WandaPruning["Wanda One-Shot Activation-Weighted Weight Pruning"]:::prune
        WandaPruning --> Sparse24_Tensors["Semi-Structured 2:4 Sparse Tensor Cores Projector<br/>• Exactly 2 non-zeros per 4 contiguous elements<br/>• 50% physical weight reduction"]:::prune
    end

    Sparse24_Tensors --> ProductionWeights["Hardware-Optimized Compact Weights Ready for Serving"]:::deploy
```

---

### 1.1.7 Phase 7 Detailed Architecture: Ultra-Fast Production Serving & Constrained Generation

```mermaid
flowchart TD
    classDef req fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef paged fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef decode fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;
    classDef stream fill:#1a3300,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;

    ClientRequests["Client Streaming Prompt Ingestion"]:::req --> ContinuousBatching["Continuous Dynamic Batching Scheduler"]:::req

    subgraph VirtualMemoryKV ["PagedAttention Virtual Memory Management"]
        ContinuousBatching --> BlockTable["Virtual Memory Block Table Manager"]:::paged
        BlockTable --> NonContigHBM["Physical GPU HBM KV Cache (Non-Contiguous Pages)"]:::paged
        NonContigHBM --> FlashDecode["FlashDecoding Parallel Split-K Attention Reduction"]:::paged
    end

    subgraph SpeculativeEngine ["Speculative Draft & Verification"]
        FlashDecode --> DraftModel["Small Speculative Draft Model (Generates K Candidates)"]:::decode
        DraftModel --> VerifyMain["Main Model Parallel Verification of K Tokens"]:::decode
        VerifyMain --> VerifiedTokens["Accepted Verified Token Sequence"]:::decode
    end

    subgraph ConstrainedSampling ["Grammar Enforcement & Deterministic Sampling"]
        VerifiedTokens --> GrammarFSM["Constrained Grammar FSM (JSON Schema / CFG Regex Masking)"]:::decode
        GrammarFSM --> SamplerModes["Sampling Selection: Contrastive Search, Min-P, Greedy Argmax"]:::decode
    end

    SamplerModes --> OutputSocket["Streaming Token Socket Delivery to Client Application"]:::stream
```

---

### 1.2 Control Plane Architecture: High-Level & Low-Level

The **Control Plane** orchestrates cluster topology assignment, elastic state synchronization, fault tolerance, and per-iteration adaptive feedback loops.

#### 1.2.1 High-Level Control Plane & Cluster Orchestration

```mermaid
flowchart TD
    classDef cplane fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef worker fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef storage fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;

    subgraph MasterPlane ["Global Cluster Control Plane"]
        Orchestrator["Cluster Orchestration & Job Scheduler"]:::cplane
        TopologyPlanner["5D Parallel Geometry Planner (TP x CP x PP x DP x EP)"]:::cplane
        DataCoordinator["Data Stream Coordinator & Domain Weight Broker"]:::cplane
        FaultManager["Health Monitor, Heartbeat & Fault Recovery Daemon"]:::cplane
        ManifestManager["Distributed Checkpoint Manifest Coordinator"]:::cplane
    end

    Orchestrator -->|Deploys Topology Config| TopologyPlanner
    Orchestrator -->|Dispatches Token Batches| DataCoordinator
    FaultManager -->|Monitors Node Heartbeats| Orchestrator
    ManifestManager -->|Registers Checkpoint Shards| Orchestrator

    subgraph ClusterMesh ["Distributed GPU Worker Ranks"]
        Node1["Rank 0..7 (Node 1 - TP Group 0)"]:::worker
        Node2["Rank 8..15 (Node 2 - TP Group 1)"]:::worker
        NodeK["Rank K..K+7 (Node N - PP Stage M)"]:::worker
    end

    subgraph StorageFabric ["Parallel Storage & State Store"]
        SharedStorage["High-Performance Parallel File System (Lustre / S3)"]:::storage
        ProxyCache["DoReMi Proxy Loss Log Store"]:::storage
    end

    TopologyPlanner -->|Assigns Process Groups| Node1
    TopologyPlanner -->|Assigns Process Groups| Node2
    TopologyPlanner -->|Assigns Process Groups| NodeK

    DataCoordinator -->|Streams Domain Mixture alpha_d| Node1
    DataCoordinator -->|Streams Domain Mixture alpha_d| Node2

    Node1 -->|Heartbeats & Loss Metrics| FaultManager
    Node2 -->|Heartbeats & Loss Metrics| FaultManager
    NodeK -->|Heartbeats & Loss Metrics| FaultManager

    Node1 -->|Persists Local State Shards| SharedStorage
    Node2 -->|Persists Local State Shards| SharedStorage
    NodeK -->|Persists Local State Shards| SharedStorage

    SharedStorage -->|Consolidates Shard Manifests| ManifestManager
    Node1 -->|Emits Proxy Losses L_proxy,d| ProxyCache
    ProxyCache -->|Updates Domain Weights| DataCoordinator
```

#### 1.2.2 Low-Level Control Plane: Per-Iteration Dynamic Adaptive Feedback Loops

Inside every single optimization step, five low-level control loops dynamically tune execution parameters based on real-time mathematical telemetry:

```mermaid
flowchart TD
    classDef telemetry fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef controller fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef action fill:#1a3300,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;
    classDef alert fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;

    StepExec["Step t Execution (Forward + Backward Pass)"]:::telemetry

    %% Control Loop 1: Noise Scale
    StepExec -->|"Per-Sample Gradient Covariance"| Telemetry1["Covariance and Gradient Norm Telemetry"]:::telemetry
    Telemetry1 --> CtlNoise["Critical Batch Noise Scale Controller"]:::controller
    CtlNoise -->|"Computes B_noise = tr(Sigma) / norm(G)^2"| ActBatch["Adapts Micro-Batch Size & Gradient Accumulation Count"]:::action

    %% Control Loop 2: Loss Spike Watchdog
    StepExec -->|"Step Loss L_t and Gradient Norm norm(g_t)"| Telemetry2["Loss & Gradient Norm Telemetry"]:::telemetry
    Telemetry2 --> CtlSpike["Loss Spike & Divergence Watchdog"]:::controller
    CtlSpike -->|"If L_t > MovingAvg + 4*StdDev"| AlertRollback["Trigger Rollback to Checkpoint t-100"]:::alert
    CtlSpike -->|"If Normal"| ActLR["Apply Warmup-Stable-Decay Learning Rate"]:::action
    AlertRollback -->|"Attenuate Learning Rate by 50%"| ActLR

    %% Control Loop 3: FP8 & Mixed-Precision Scaler
    StepExec -->|"Tensor Max Absolute Value amax"| Telemetry3["FP8 Quantization Exponent Telemetry"]:::telemetry
    Telemetry3 --> CtlFP8["Dynamic Delayed Loss & Tensor Scaler"]:::controller
    CtlFP8 -->|"Detects Inf/NaN Overflow"| ActSkip["Skip Optimizer Step & Downscale Scale Factor S"]:::alert
    CtlFP8 -->|"Stable Dynamic Range"| ActScale["Upscale S for Maximum FP8 Mantissa Utilization"]:::action

    %% Control Loop 4: DoReMi Domain Mixture
    StepExec -->|"Domain Cross-Entropy Losses"| Telemetry4["Proxy vs Reference Domain Losses"]:::telemetry
    Telemetry4 --> CtlDoReMi["DoReMi Simplex Mirror Descent Controller"]:::controller
    CtlDoReMi -->|"Exponentiated Excess Loss Update"| ActMixture["Adjust Next Epoch Domain Sampling Probabilities"]:::action

    %% Actions feed into next step
    ActBatch --> NextStep["Configure Step t+1 Parameters"]:::telemetry
    ActLR --> NextStep
    ActScale --> NextStep
    ActMixture --> NextStep
```

---

### 1.3 Data Plane & Distributed Messaging Topology: High-Level & Low-Level

The **Data Plane** handles high-throughput numerical tensor computation, collective networking operations, and memory hierarchy staging across GPUs.

#### 1.3.1 High-Level 5D Distributed Messaging Fabric & Collective Topologies

Frontier models partition tensor representations across five orthogonal communication planes:

```mermaid
flowchart TD
    classDef nvlink fill:#1a3300,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;
    classDef ib_ring fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef ib_pipe fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;
    classDef switch_all fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef zero_mesh fill:#204040,stroke:#94e2d5,stroke-width:2px,color:#cdd6f4;

    subgraph Plane1 ["1. Tensor Parallelism (TP) — Intra-Node NVLink Mesh (900 GB/s)"]
        TP1["GPU 0 (Col W_qkv)"]:::nvlink <-->|All-Reduce Sum / All-Gather| TP2["GPU 1 (Col W_qkv)"]:::nvlink
        TP2 <-->|All-Reduce Sum / All-Gather| TP3["GPU 2 (Col W_qkv)"]:::nvlink
        TP3 <-->|All-Reduce Sum / All-Gather| TP4["GPU 3 (Col W_qkv)"]:::nvlink
    end

    subgraph Plane2 ["2. Context Parallelism (CP) — Intra-Pod InfiniBand Ring (400 Gbps)"]
        CP1["GPU Rank i (Chunk 0..S/4)"]:::ib_ring -->|Ring Send/Recv K, V| CP2["GPU Rank i+1 (Chunk S/4..S/2)"]:::ib_ring
        CP2 -->|Ring Send/Recv K, V| CP3["GPU Rank i+2 (Chunk S/2..3S/4)"]:::ib_ring
        CP3 -->|Ring Send/Recv K, V| CP4["GPU Rank i+3 (Chunk 3S/4..S)"]:::ib_ring
        CP4 -->|Ring Send/Recv K, V| CP1
    end

    subgraph Plane3 ["3. Pipeline Parallelism (PP) — Inter-Rack P2P InfiniBand Pipelines"]
        Stage0["Pipeline Stage 0 (Layers 1..16)"]:::ib_pipe -->|P2P Forward Activations| Stage1["Pipeline Stage 1 (Layers 17..32)"]:::ib_pipe
        Stage1 -->|P2P Forward Activations| Stage2["Pipeline Stage 2 (Layers 33..48)"]:::ib_pipe
        Stage2 -->|P2P Backward Gradients| Stage1
        Stage1 -->|P2P Backward Gradients| Stage0
    end

    subgraph Plane4 ["4. Expert Parallelism (EP) — Cluster Switched Fabric All-to-All"]
        RankE0["GPU Worker 0 (Tokens)"]:::switch_all <-->|All-to-All Token Dispatch & Combine| ExpertNode0["Expert Host 0 (Specialized MLP 0..3)"]:::switch_all
        RankE1["GPU Worker 1 (Tokens)"]:::switch_all <-->|All-to-All Token Dispatch & Combine| ExpertNode1["Expert Host 1 (Specialized MLP 4..7)"]:::switch_all
    end

    subgraph Plane5 ["5. Data Parallelism & ZeRO-3 / FSDP — Memory Partition Mesh"]
        ShardedOpt["Optimizer States Partitioned (ZeRO-1)"]:::zero_mesh
        ShardedGrad["Gradients Reduce-Scattered (ZeRO-2)"]:::zero_mesh
        ShardedParam["Parameters All-Gathered On-Demand (ZeRO-3)"]:::zero_mesh
        ShardedOpt --> ShardedGrad --> ShardedParam
    end
```

#### 1.3.2 How Data Moves Through a Transformer Layer (Simple Explanation)

A single Transformer layer does two main jobs:
1. **The Forward Pass (Making Predictions)**: Words come in as numbers, the model cleans them up, adds word positions, lets words compare and pay attention to each other, looks up facts in feed-forward layers, and produces new, smarter representations.
2. **The Backward Pass (Learning from Errors)**: The model checks how far off its prediction was, sends error signals (gradients) backward through each step, and updates its weights using AdamW so it makes fewer mistakes next time.

```mermaid
flowchart TD
    classDef input fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef step fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef network fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;
    classDef learn fill:#1a3300,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;

    subgraph ForwardPass ["1. Forward Pass (Processing Words into Smarter Meanings)"]
        WordInputs["Input Word Representations"]:::input --> CleanNumbers["1. Normalize Values with RMSNorm<br/>(Keeps numbers stable so they don't explode)"]:::step
        CleanNumbers --> AddPositions["2. Add Word Order with RoPE<br/>(Tells the model which word came first, second, etc.)"]:::step
        AddPositions --> AttentionCompare["3. Multi-Head Attention<br/>(Lets words look at and compare each other)"]:::step
        AttentionCompare --> CombineAttn["4. Combine Attention Output<br/>(Puts all head discoveries back together)"]:::step
        CombineAttn --> AddSkip1["5. Add Skip Connection<br/>(Adds input back so the model never forgets original context)"]:::step
        AddSkip1 --> CleanNumbers2["6. Second Normalization (RMSNorm)<br/>(Prepares numbers for feed-forward knowledge lookup)"]:::step
        CleanNumbers2 --> ExpertLookup["7. Feed-Forward / Expert Networks (MLP / MoE)<br/>(Looks up facts and deeper linguistic concepts)"]:::step
        ExpertLookup --> AddSkip2["8. Second Skip Connection<br/>(Adds previous layer output back for stable gradient flow)"]:::step
        AddSkip2 --> WordOutputs["Output Representations Ready for Next Layer"]:::input
    end

    subgraph BackwardPass ["2. Backward Pass (Learning from Mistakes)"]
        LossError["Calculate Error at Output<br/>(How wrong was the model's guess?)"]:::learn --> SendBackErrors["Send Gradients Backward<br/>(Reverse-mode autodiff calculates each step's blame)"]:::learn
        SendBackErrors --> SyncAcrossGPUs["Synchronize Gradients Across GPUs<br/>(Share what every GPU learned across the cluster)"]:::network
        SyncAcrossGPUs --> UpdateWeights["Update Weights with AdamW<br/>(Nudge parameters in the right direction)"]:::learn
    end

    WordOutputs -.->|"Feeds Loss Function"| LossError
```

---

### 1.4 Post-Training Alignment & Verifiable Reasoning Loop

Post-training splits into two parallel pipelines: **Human Preference Alignment** and **Verifiable Mathematical/Code Reasoning**, which synthesize into unified frontier checkpoints:

```mermaid
flowchart TD
    classDef base fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef pref fill:#4a154b,stroke:#f5c2e7,stroke-width:2px,color:#cdd6f4;
    classDef reason fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;
    classDef merge fill:#204040,stroke:#94e2d5,stroke-width:2px,color:#cdd6f4;

    PretrainedBase["Pre-Trained Foundation Base Model"]:::base --> SFT["Supervised Fine-Tuning (SFT) on Instruction Pairs"]:::base

    %% Track 1: Human Preference
    subgraph Track1 ["Track A: Human Preference Alignment (Safety & Style)"]
        SFT --> PairData["Pairwise Preference Datasets: Winning vs Losing Completions"]:::pref
        PairData --> BT["Bradley-Terry Pairwise Reward Modeling"]:::pref
        BT --> PPO["RLHF PPO Policy Optimization with Adaptive KL Divergence Penalty"]:::pref

        PairData --> DirectAlign["Reference-Free Direct Alignment"]:::pref
        DirectAlign --> DPO_Algo["Direct Preference Optimization (DPO)"]:::pref
        DirectAlign --> SimPO_KTO["SimPO Length-Normalized Margin & KTO Prospect Theory"]:::pref
    end

    %% Track 2: Verifiable Reasoning
    subgraph Track2 ["Track B: Verifiable Reasoning & Self-Play (Math, Code & Logic)"]
        SFT --> MathPrompts["Complex Reasoning Prompts (GSM8K, MATH, HumanEval)"]:::reason
        MathPrompts --> GroupRollouts["Group Policy Rollouts: Sample G Outputs per Prompt"]:::reason
        GroupRollouts --> GroundTruthEnv["Deterministic Execution Sandbox (Python / Sympy / Unit Tests)"]:::reason

        GroundTruthEnv --> GRPO_Algo["Group Relative Policy Optimization (GRPO Critic-Free Advantages)"]:::reason
        GroundTruthEnv --> RLVR_Algo["Reinforcement Learning with Verifiable Rewards (RLVR)"]:::reason

        GroupRollouts --> StepPRM["Process Reward Models (Step-Level Credit PRM)"]:::reason
        StepPRM --> BoN["Best-of-N Monte Carlo Tree Rejection Sampling"]:::reason
        BoN --> RLAIF["Constitutional AI Self-Critique & Automated Revision"]:::reason
    end

    %% Merging & Deployment
    PPO --> AlignedPolicyA["Safety & Conversational Aligned Checkpoint"]:::pref
    DPO_Algo --> AlignedPolicyA
    SimPO_KTO --> AlignedPolicyA

    GRPO_Algo --> ReasonPolicyB["Deep Reasoning & Code Mathematical Checkpoint"]:::reason
    RLVR_Algo --> ReasonPolicyB
    RLAIF --> ReasonPolicyB

    AlignedPolicyA --> MergeEngine["TIES & DARE Multi-Task Parameter Merging (Sign Concordance)"]:::merge
    ReasonPolicyB --> MergeEngine

    MergeEngine --> MergedModel["Unified Frontier Aligned Reasoning Model"]:::merge
    MergedModel --> LoRA_Adapt["LoRA, QLoRA NF4 & DoRA Low-Rank Adaptation"]:::merge
    LoRA_Adapt --> Wanda_Prune["Wanda One-Shot Activation-Weighted Pruning"]:::merge
    Wanda_Prune --> Sparse24_HW["Semi-Structured 2:4 Hardware Tensor Cores (2x Faster GEMMs)"]:::merge
```

---

### 1.5 Production Inference Serving & Constrained Generation Data Plane

The diagram below maps the inference pipeline from continuous request ingestion to token streaming:

```mermaid
flowchart LR
    classDef req fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef mem fill:#1a3300,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;
    classDef kernel fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef sample fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;

    ClientPrompt["Incoming Client Generation Prompt"]:::req --> ContinuousBatcher["Continuous Dynamic Batching Scheduler"]:::req
    ContinuousBatcher --> BlockManager["PagedAttention Virtual Memory Block Table Manager"]:::mem

    BlockManager --> PhysicalBlocks["Physical GPU HBM KV Cache Blocks (Non-Contiguous Pages)"]:::mem
    PhysicalBlocks --> FlashDecoding["FlashDecoding Split-K Parallel Attention Reduction"]:::kernel

    FlashDecoding --> SpecDraft["Speculative Decoding: Small Draft Model Generates K Tokens"]:::kernel
    SpecDraft --> DraftVerify["Main Model Verifies K Candidate Draft Tokens in Parallel"]:::kernel

    DraftVerify --> GrammarFSM["Constrained Grammar Finite State Machine (JSON / CFG Enforcer)"]:::sample
    GrammarFSM --> Sampler["Contrastive Search, Min-P & Temperature Sampling"]:::sample
    Sampler --> TokenStream["Streaming Next-Token Output to Client Socket"]:::req
```

---

## 2. End-to-End Computational Walkthrough: Token to Generation

Every stage of a production LLM's life cycle is directly mapped to concrete algorithms in this directory:

### Step 1: Pre-Training Data Preparation & Ingestion
1. **Synthetic Data Quality Filtering and Deduplication** (`training_at_scale/synthetic_data_generation_filtering`): Validates incoming candidate documents against quality scores, token length boundaries, and incremental Jaccard similarity to reject hallucinations and prevent model collapse.
2. **DoReMi Dynamic Domain Reweighting** (`training_at_scale/data_mixture_optimization_doremi`): Employs a small proxy model against a reference baseline to calculate excess loss $\Delta_d = L_{\text{proxy}, d} - L_{\text{ref}, d}$, updating domain mixture weights $\boldsymbol{\alpha}$ via exponentiated gradient mirror descent to avoid noisy domain starvation.
3. **Sequence Packing with Causal Document Masking** (`training_at_scale/sequence_packing_doc_masking`): Packs multiple variable-length documents into fixed-length sequences (e.g., 4096 or 8192 tokens) using cumulative segment index offsets to construct 2D block-diagonal causal masks, eliminating cross-document token attention leakage and padding bubble waste.

### Step 2: Transformer Layer Construction & Forward Propagation
1. **Token & Position Embedding**: Inputs pass through `building_blocks/embedding_lookup`, augmented by rotary position embeddings (`position_encodings/rotary_position_embeddings_rope`) or linear biases (`position_encodings/alibi_attention_linear_biases`).
2. **Pre-Layer Normalization**: Evaluated through `normalization/rmsnorm` or `normalization/layer_normalization`, stabilizing activations across extreme depths without mean computation.
3. **Efficient Multi-Head Attention**: Token representations are processed by `transformer_efficiency/flash_attention` or `transformer_efficiency/grouped_query_attention`, with persistent KV cache management via `transformer_efficiency/kv_cache`.
4. **MoE Routing & Feed-Forward**: For sparse models, `mixture_of_experts/moe_top_k_gating` routes tokens across expert feed-forward networks with auxiliary load balancing (`mixture_of_experts/moe_load_balancing`) and gated linear activations (`building_blocks/gated_linear_units`).
5. **Stability & Sink Protection**: Softmax attention utilizes `scaling_stability/attention_sinks_streaming` to preserve initial sequence tokens, and `scaling_stability/logit_stabilization_z_loss` penalizes large log-sum-exp partitions.

### Step 3: Massive Distributed Execution (3D/5D Parallelism)
1. **Tensor Parallelism** (`training_at_scale/tensor_parallelism_megatron`): Shards column-parallel projections ($W_q, W_k, W_v, W_{\text{gate}}$) and row-parallel projections ($W_o, W_{\text{down}}$) across GPU intra-node NVLink meshes via All-Reduce synchronization.
2. **Sequence & Context Parallelism** (`training_at_scale/context_parallelism_ring_attention`): Splits multi-hundred-thousand token sequences across GPUs in a ring topology, overlapping causal block computation with peer-to-peer communication.
3. **Pipeline Parallelism** (`training_at_scale/pipeline_parallelism_1f1b`): Schedules micro-batches in a 1-Forward-1-Backward pipeline across multi-node stages, minimizing memory bubbles.
4. **Memory Sharding & Offloading** (`training_at_scale/zero_fsdp_sharded_training`; `offloading_activation_parameter`): Eliminates memory redundancy across data-parallel ranks by partitioning optimizer states (ZeRO-1), gradients (ZeRO-2), and model parameters (ZeRO-3/FSDP), offloading inactive layers to host CPU RAM.
5. **Numerical Precision** (`training_at_scale/mixed_precision_training`; `fp8_training`): Runs forward/backward passes in FP16/BF16 with dynamic loss scaling, or FP8 E4M3/E5M2 tensor cores with per-tensor delayed scaling.

### Step 4: Gradient Calculation, Optimization & Resumption
1. **Loss Evaluation**: Target predictions compute `losses/cross_entropy_nll` with label smoothing (`losses/label_smoothing`) and z-loss logit regularization.
2. **Autodiff & Backpropagation**: Gradients flow via `autodiff/reverse_mode_automatic_differentiation` with activation recomputation via `autodiff/gradient_checkpointing`.
3. **Noise Scale & Critical Batch Tuning**: Evaluates `training_at_scale/critical_batch_size_noise_scale` using gradient covariance trace $\operatorname{tr}(\Sigma) / \|G\|^2$ to maintain optimal compute efficiency.
4. **Parameter Updates**: Weights update through `optimizers/adamw_optimizer`, `adafactor_optimizer`, or `lion_optimizer` under Warmup-Stable-Decay schedules (`schedules_init/wsd_one_cycle_schedules`) and Maximal Update Parametrization (`schedules_init/maximal_update_param`).
5. **Spike Recovery & Resumption**: Detects gradient anomalies and restores healthy weights via `training_at_scale/loss_spike_detection_recovery` and persists elastic shards via `training_at_scale/distributed_checkpointing_resumption`.

### Step 5: Post-Training, Alignment, Reasoning & Self-Play
1. **Supervised Fine-Tuning**: Trains conversational capabilities on instruction pairs via `llm_alignment_post_training/supervised_fine_tuning`.
2. **Preference Optimization**: Learns human alignment through Bradley-Terry reward models (`reward_modeling_bradley_terry`) and RLHF PPO with KL penalties (`rlhf_ppo_kl_penalty`), or direct reference-free alignment via DPO (`direct_preference_optimization_dpo`) and SimPO/KTO (`dpo_variants_simpo_kto`).
3. **Reasoning & Verifiable Self-Play**: Elicits mathematical and algorithmic reasoning using Group Relative Policy Optimization (`group_relative_policy_optimization_grpo`) without a critic, Reinforcement Learning with Verifiable Rewards (`rl_verifiable_rewards_rlvr`), Process Reward Models (`process_reward_models_prm`), and Constitutional AI critiques (`rlaif_constitution_critique`).

### Step 6: Parameter-Efficient Adaptation, Merging & Pruning
1. **Low-Rank Adaptation**: Fine-tunes task adapters with minimal memory using LoRA ($W + \frac{\alpha}{r} B A$) (`peft_model_merging/lora_low_rank_adaptation`), QLoRA NF4 quantization (`qlora_nf4_quantization`), and Weight-Decomposed LoRA (`dora_weight_decomposed_lora`).
2. **Model Merging**: Blends distinct fine-tuned expert models without retraining via TIES and DARE (`peft_model_merging/model_merging_ties_dare`).
3. **Distillation & Continual Retention**: Transfers knowledge to small edge models (`distillation_continual_learning/knowledge_distillation_soft_targets`) while preventing catastrophic forgetting using Elastic Weight Consolidation (`elastic_weight_consolidation_ewc`).
4. **Structured & Semi-Structured Hardware Pruning**: Accelerates inference through one-shot activation-weighted pruning (`pruning_sparsity/structured_oneshot_pruning_llms` / Wanda) and projects weights to hardware 2:4 Sparse Tensor Cores (`pruning_sparsity/semistructured_2to4_sparsity`) for a deterministic $2\times$ GEMM speedup.

### Step 7: Fast Inference Generation & Grammar Decoding
1. **Paged Memory Allocation**: Manages inference memory pools using `transformer_efficiency/paged_attention`.
2. **Speculative & Flash Decoding**: Accelerates auto-regressive decoding via small draft models (`decoding/speculative_decoding`) and parallelized attention reductions (`transformer_efficiency/flash_decoding`).
3. **Constrained Generation**: Enforces strict JSON / CFG grammars via `decoding/constrained_grammar_decoding` and diversified output via `decoding/contrastive_search_decoding`.

---

## 3. Exhaustive Architectural Domain Catalog

| # | Domain Directory | Algos | Focus & Capabilities | Representative Algorithms |
|---|---|:---:|---|---|
| **1** | [`building_blocks`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/building_blocks) | 12 | Fundamental neural layers, activations, and residual skips | Linear Layer, GELU, SwiGLU, Residual Connections, Embedding Lookup |
| **2** | [`autodiff`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/autodiff) | 8 | Automatic differentiation, gradients, and memory recomputation | Reverse-Mode AD, Vector-Jacobian Products, Gradient Checkpointing |
| **3** | [`losses`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/losses) | 10 | Supervised, ranking, and alignment loss objectives | Cross-Entropy, Binary Cross-Entropy, Focal Loss, Triplet Loss, CTC |
| **4** | [`optimizers`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/optimizers) | 12 | First- and second-order stochastic gradient optimization | SGD+Momentum, Adam, AdamW, Adafactor, Lion, LARS/LAMB, Muon |
| **5** | [`schedules_init`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/schedules_init) | 8 | Parameter initialization and dynamic learning rate schedules | Cosine Warmup, WSD, He/Kaiming Init, Xavier, $\mu$P Parametrization |
| **6** | [`normalization`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/normalization) | 7 | Activation normalization and manifold centering | LayerNorm, RMSNorm, BatchNorm, GroupNorm, Weight Normalization |
| **7** | [`regularization_augmentation`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/regularization_augmentation) | 9 | Overfitting prevention, stochastic depth, and data augmentation | Dropout, DropPath, Mixup/CutMix, Stochastic Depth, Weight Decay |
| **8** | [`convolutional`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/convolutional) | 17 | Spatial feature extraction, conv mechanics, and vision backbones | 2D Convolution, Depthwise Separable Conv, ResNet Block, ConvNeXt |
| **9** | [`detection`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/detection) | 6 | Object localization, bounding boxes, and bipartite matching | NMS, IoU Box Losses, DETR Hungarian Bipartite Matching, FCOS |
| **10** | [`recurrent_sequence`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/recurrent_sequence) | 11 | Temporal processing, gated recurrent cells, and sequence translation | LSTM, GRU, Bidirectional RNN, CTC Decoder, Bahdanau Attention |
| **11** | [`attention_objectives`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/attention_objectives) | 5 | Foundational attention mechanics and pre-training objectives | Scaled Dot-Product, Multi-Head Attention, Causal LM, Masked LM |
| **12** | [`position_encodings`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/position_encodings) | 6 | Spatial and sequential token coordinate representations | Sinusoidal PE, Learned PE, RoPE, ALiBi, LongRoPE / YaRN Context |
| **13** | [`transformer_efficiency`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/transformer_efficiency) | 13 | Hardware-aligned attention and sub-quadratic state space models | FlashAttention, GQA, PagedAttention, Mamba SSM, S4, MLA, RetNet |
| **14** | [`mixture_of_experts`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/mixture_of_experts) | 3 | Conditional computation, dynamic routing, and capacity control | Top-K Softmax Gating, Auxiliary Load Balancing, Expert Choice Routing |
| **15** | [`decoding`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/decoding) | 10 | Generation strategies, speculative acceleration, and grammar guards | Greedy, Top-P, Min-P, Speculative Decoding, Contrastive, Grammar |
| **16** | [`scaling_stability`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/scaling_stability) | 3 | Compute-optimal scaling laws and extreme logit stabilization | Chinchilla Scaling, Attention Sinks Streaming, Logit Z-Loss |
| **17** | [`generative_autoencoders_gans`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/generative_autoencoders_gans) | 10 | Unsupervised generative modeling and latent density learning | Variational Autoencoders (VAE), VQ-VAE, PixelCNN, WGAN-GP |
| **18** | [`diffusion_flow_models`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/diffusion_flow_models) | 11 | Score-based generative models, continuous ODEs, and adapters | DDPM, DDIM, Classifier-Free Guidance, Flow Matching, ControlNet |
| **19** | [`multimodal_memory`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/multimodal_memory) | 10 | Cross-modal alignment, perceptual resampling, and multi-token loss | CLIP Contrastive Learning, Cross-Attention, Perceiver, Multi-Token Loss |
| **20** | [`self_supervised_learning`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/self_supervised_learning) | 5 | Visual and sequence representations without human labels | SimCLR, DINO Self-Distillation, Masked Autoencoders (MAE), I-JEPA |
| **21** | [`reinforcement_learning`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/reinforcement_learning) | 5 | Classic value-based and policy gradient agent decision methods | REINFORCE, DQN, Double/Dueling DQN, Actor-Critic GAE, PPO |
| **22** | [`timeseries_recommendation`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/timeseries_recommendation) | 7 | Forecasting, ranking architectures, and feature interactions | DLRM, Deep & Cross V2, DIN Attention, N-BEATS, PatchTST, SASRec |
| **23** | [`specialized_scientific_meta`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/specialized_scientific_meta) | 12 | Physics-informed networks, differentiable solvers, and meta-learning | PINNs, Neural ODEs, FNO, 3D Gaussian Splatting, NeRF, MAML, DARTS |
| **24** | [`advanced_reinforcement_learning`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/advanced_reinforcement_learning) | 6 | Continuous control, model-based simulation, and planning searches | Soft Actor-Critic (SAC), TD3, Dreamer World Models, AlphaZero MCTS |
| **25** | [`llm_alignment_post_training`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/llm_alignment_post_training) | 11 | Human preference alignment, reasoning self-play, and verifiers | SFT, Bradley-Terry, RLHF PPO, DPO, SimPO/KTO, GRPO, RLVR, PRM |
| **26** | [`peft_model_merging`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/peft_model_merging) | 7 | Low-rank adaptation, parameter quantization, and weight merging | LoRA, QLoRA NF4, DoRA, Bottleneck Adapters, $(IA)^3$, TIES/DARE |
| **27** | [`distillation_continual_learning`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/distillation_continual_learning) | 4 | Teacher-student knowledge transfer and forgetting prevention | Soft Distillation, On-Policy Distillation, EWC Fisher Matrix, Replay |
| **28** | [`training_at_scale`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/training_at_scale) | 19 | Distributed 5D parallel pipelines, fault recovery, and privacy | Megatron TP, Ring CP, 1F1B PP, ZeRO-3 FSDP, FP8, DoReMi, FedAvg, DP-SGD |
| **29** | [`pruning_sparsity`](file:///home/btpl-lap-22/live/llm-obs-infra/policies/policy-orchestrator/src/features/code_engine/algos/nn/pruning_sparsity) | 3 | Weight compression, lottery ticket discovery, and tensor sparsity | Lottery Ticket IMP, Wanda One-Shot Pruning, Semi-Structured 2:4 |

---

## 4. Contract Specification & Design Philosophy

Every algorithm implementation adheres to strict architectural guarantees:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ALGO IMPLEMENTATION FILE                        │
│                                                                        │
│  1. Zero External Runtime Imports                                      │
│     • Strictly no `torch`, `numpy`, or `scipy`                         │
│     • Deterministic, portable scalar Python execution                  │
│                                                                        │
│  2. Declarative 26-Key YAML Contract Docstring                         │
│     • algo_id, name, version, category, capability_tags                │
│     • Formal JSON schema for inputs and outputs                        │
│     • Precise Big-Theta time and space complexity                      │
│     • Preconditions, postconditions, and invariant certificates        │
│     • Compatible adapter interfaces and cross-algo links               │
│                                                                        │
│  3. Pure Class Method Signature: @classmethod def forward(...)         │
│     • Strict runtime precondition validation with informative errors   │
│     • Deterministic numerical return dictionary                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 4.1 Master End-to-End Inter-Component Data Transmission Diagram

The diagram below maps the complete end-to-end algorithmic data transmission architecture in Mermaid, covering data curation, micro-architecture, 5D scaling, loss optimization, post-training reasoning alignment, hardware compression, and high-throughput serving:

```mermaid
flowchart TD
    classDef curation fill:#1e3a5f,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef arch fill:#2d1b4e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef scale fill:#3d1f00,stroke:#fe8019,stroke-width:2px,color:#cdd6f4;
    classDef opt fill:#1a3300,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;
    classDef align fill:#4a154b,stroke:#f5c2e7,stroke-width:2px,color:#cdd6f4;
    classDef peft fill:#204040,stroke:#94e2d5,stroke-width:2px,color:#cdd6f4;
    classDef serve fill:#2b2c3b,stroke:#f9e2af,stroke-width:2px,color:#cdd6f4;
    classDef decode fill:#3a203a,stroke:#eba0ac,stroke-width:2px,color:#cdd6f4;

    %% -------------------------------------------------------------------------
    %% 1. DATA CURATION & DYNAMIC MIXTURE INGESTION
    %% -------------------------------------------------------------------------
    subgraph S1 ["1. Data Curation & Dynamic Mixture Ingestion"]
        Curation["<strong>Synthetic Data Quality Filtering and Deduplication</strong><br/>📥 <b>JSON Input:</b> <code>{ candidate_samples, min_quality_score, jaccard_threshold }</code><br/>📤 <b>JSON Response:</b> <code>{ accepted_samples, rejection_breakdown, acceptance_ratio }</code>"]:::curation
        DoReMi["<strong>DoReMi Dynamic Domain Reweighting</strong><br/>📥 <b>JSON Input:</b> <code>{ domain_losses_proxy, domain_losses_reference, current_weights }</code><br/>📤 <b>JSON Response:</b> <code>{ updated_domain_weights, excess_losses, worst_domain_idx }</code>"]:::curation
        Packing["<strong>Sequence Packing with Causal Document Masking</strong><br/>📥 <b>JSON Input:</b> <code>{ document_token_sequences, max_sequence_length, padding_token_id }</code><br/>📤 <b>JSON Response:</b> <code>{ packed_sequences, segment_ids, position_ids, boundary_masks }</code>"]:::curation

        Curation -->|"transmits JSON: { accepted_samples, acceptance_ratio }"| DoReMi
        DoReMi -->|"transmits JSON: { updated_domain_weights }"| Packing
    end

    %% -------------------------------------------------------------------------
    %% 2. TRANSFORMER BLOCK MICRO-ARCHITECTURE & ACTIVATION DYNAMICS
    %% -------------------------------------------------------------------------
    subgraph S2 ["2. Transformer Layer Activations & Attention Dynamics"]
        Emb["<strong>Embedding Lookup & High-Dimensional Projection</strong><br/>📥 <b>JSON Input:</b> <code>{ token_ids, embedding_table }</code><br/>📤 <b>JSON Response:</b> <code>{ embedded_representations }</code>"]:::arch
        RMSNorm["<strong>Root Mean Square Normalization (RMSNorm)</strong><br/>📥 <b>JSON Input:</b> <code>{ input_tensor, gamma, eps }</code><br/>📤 <b>JSON Response:</b> <code>{ normalized_tensor, rms_values }</code>"]:::arch
        RoPE["<strong>Rotary Position Embeddings (RoPE) & YaRN</strong><br/>📥 <b>JSON Input:</b> <code>{ vectors, positions, base }</code><br/>📤 <b>JSON Response:</b> <code>{ rotated_vectors, dim }</code>"]:::arch
        MHA["<strong>Multi-Head Attention & Attention Sinks</strong><br/>📥 <b>JSON Input:</b> <code>{ query, key, value, num_heads, mask }</code><br/>📤 <b>JSON Response:</b> <code>{ output, head_outputs, head_dim }</code>"]:::arch
        MoERoute["<strong>Top-K Mixture-of-Experts Router</strong><br/>📥 <b>JSON Input:</b> <code>{ input_activations, routing_weights }</code><br/>📤 <b>JSON Response:</b> <code>{ expert_dispatch_indices, routing_probabilities }</code>"]:::arch
        SwiGLU["<strong>SwiGLU Gated Feedforward Networks</strong><br/>📥 <b>JSON Input:</b> <code>{ gate_tensor, up_tensor, down_tensor }</code><br/>📤 <b>JSON Response:</b> <code>{ expert_output_activations }</code>"]:::arch
        Residual["<strong>Residual Addition & Highway Connections</strong><br/>📥 <b>JSON Input:</b> <code>{ identity_tensor, transformed_tensor }</code><br/>📤 <b>JSON Response:</b> <code>{ output_hidden_states }</code>"]:::arch

        Packing -->|"transmits JSON: { packed_sequences }"| Emb
        Packing -.->|"transmits JSON: { position_ids }"| RoPE
        Packing -.->|"transmits JSON: { boundary_masks }"| MHA
        Emb -->|"transmits JSON: { embedded_representations }"| RMSNorm
        RMSNorm -->|"transmits JSON: { normalized_tensor }"| RoPE
        RoPE -->|"transmits JSON: { rotated_vectors }"| MHA
        MHA -->|"transmits JSON: { output }"| MoERoute
        MoERoute -->|"transmits JSON: { expert_dispatch_indices }"| SwiGLU
        SwiGLU -->|"transmits JSON: { expert_output_activations }"| Residual
    end

    %% -------------------------------------------------------------------------
    %% 3. 5D DISTRIBUTED HARDWARE PARALLEL EXECUTION
    %% -------------------------------------------------------------------------
    subgraph S3 ["3. 5D Distributed Hardware Parallel Execution"]
        TP["<strong>Megatron Tensor Parallelism (Column/Row GEMM)</strong><br/>📥 <b>JSON Input:</b> <code>{ input_vector, column_shards, row_shards }</code><br/>📤 <b>JSON Response:</b> <code>{ output_vector, shard_hidden_activations, tp_degree }</code>"]:::scale
        CP["<strong>Context Parallelism (Ring Attention)</strong><br/>📥 <b>JSON Input:</b> <code>{ local_q_chunks, local_k_chunks, local_v_chunks }</code><br/>📤 <b>JSON Response:</b> <code>{ context_attended_hidden_states }</code>"]:::scale
        PP["<strong>Pipeline Parallelism (1F1B Micro-Batches)</strong><br/>📥 <b>JSON Input:</b> <code>{ microbatch_activations, stage_id }</code><br/>📤 <b>JSON Response:</b> <code>{ forward_activations, backward_gradients }</code>"]:::scale
        ZeRO["<strong>ZeRO-3 / FSDP Memory Sharding</strong><br/>📥 <b>JSON Input:</b> <code>{ sharded_weights, sharded_gradients, sharded_optimizer_states }</code><br/>📤 <b>JSON Response:</b> <code>{ all_gathered_weights, partitioned_updates }</code>"]:::scale
        FP8["<strong>FP8 Mixed-Precision Tensor Core Scaling</strong><br/>📥 <b>JSON Input:</b> <code>{ fp32_tensors, dynamic_scale_factor_S }</code><br/>📤 <b>JSON Response:</b> <code>{ fp8_e4m3_forward, fp8_e5m2_backward }</code>"]:::scale

        Residual -->|"transmits JSON: { output_hidden_states }"| TP
        TP -->|"transmits JSON: { output_vector }"| CP
        CP -->|"transmits JSON: { context_attended_hidden_states }"| PP
        PP -->|"transmits JSON: { forward_activations }"| ZeRO
        ZeRO -->|"transmits JSON: { all_gathered_weights }"| FP8
    end

    %% -------------------------------------------------------------------------
    %% 4. LOSS COMPUTATION, AUTODIFF & GRADIENT OPTIMIZATION
    %% -------------------------------------------------------------------------
    subgraph S4 ["4. Loss Computation, Autodiff & Gradient Optimization"]
        Loss["<strong>Cross-Entropy Loss with Label Smoothing & Z-Loss</strong><br/>📥 <b>JSON Input:</b> <code>{ logits, targets, label_smoothing, z_loss_tau }</code><br/>📤 <b>JSON Response:</b> <code>{ cross_entropy_loss, z_loss_term, total_objective }</code>"]:::opt
        VJP["<strong>Reverse-Mode Autodiff Vector-Jacobian Products</strong><br/>📥 <b>JSON Input:</b> <code>{ objective_loss, saved_activation_graph }</code><br/>📤 <b>JSON Response:</b> <code>{ parameter_gradients, activation_adjoints }</code>"]:::opt
        NoiseScale["<strong>Critical Batch Size Noise Scale Estimator</strong><br/>📥 <b>JSON Input:</b> <code>{ per_sample_gradients, current_step, total_steps }</code><br/>📤 <b>JSON Response:</b> <code>{ recommended_batch_size, estimated_noise_scale_b_noise }</code>"]:::opt
        AdamW["<strong>AdamW Decoupled Weight Decay Optimizer</strong><br/>📥 <b>JSON Input:</b> <code>{ parameters, gradients, exp_avg, exp_avg_sq, lr, weight_decay }</code><br/>📤 <b>JSON Response:</b> <code>{ updated_parameters, updated_exp_avg, updated_exp_avg_sq }</code>"]:::opt
        WSD["<strong>Warmup-Stable-Decay Learning Rate Scheduler</strong><br/>📥 <b>JSON Input:</b> <code>{ step, total_steps, warmup_ratio, decay_ratio }</code><br/>📤 <b>JSON Response:</b> <code>{ scheduled_learning_rate }</code>"]:::opt
        Watchdog["<strong>Loss Spike Divergence Watchdog</strong><br/>📥 <b>JSON Input:</b> <code>{ current_step_loss, moving_average, moving_std }</code><br/>📤 <b>JSON Response:</b> <code>{ is_anomaly_detected, rollback_trigger }</code>"]:::opt

        FP8 -->|"transmits JSON: { forward_activations }"| Loss
        Loss -->|"transmits JSON: { total_objective }"| VJP
        VJP -->|"transmits JSON: { parameter_gradients }"| NoiseScale
        NoiseScale -->|"transmits JSON: { recommended_batch_size }"| AdamW
        WSD -->|"transmits JSON: { scheduled_learning_rate }"| AdamW
        VJP -->|"transmits JSON: { per_sample_gradients }"| Watchdog
        AdamW -->|"transmits JSON: { updated_parameters }"| CheckpointSave["<strong>Elastic Distributed Checkpoint State</strong>"]:::opt
    end

    %% -------------------------------------------------------------------------
    %% 5. POST-TRAINING DUAL-TRACK ALIGNMENT & REASONING ENGINE
    %% -------------------------------------------------------------------------
    subgraph S5 ["5. Post-Training Dual-Track Alignment & Reasoning Engine"]
        SFT["<strong>Supervised Fine-Tuning (SFT) Core</strong><br/>📥 <b>JSON Input:</b> <code>{ base_weights, instruction_demonstrations }</code><br/>📤 <b>JSON Response:</b> <code>{ instruction_tuned_weights }</code>"]:::align
        
        BT["<strong>Bradley-Terry Reward Modeling</strong><br/>📥 <b>JSON Input:</b> <code>{ pairwise_comparisons, reward_network_weights }</code><br/>📤 <b>JSON Response:</b> <code>{ winning_scalar_rewards, losing_scalar_rewards }</code>"]:::align
        RLHF["<strong>RLHF PPO with Adaptive KL Penalty</strong><br/>📥 <b>JSON Input:</b> <code>{ policy_rollouts, reward_scores, reference_log_probs }</code><br/>📤 <b>JSON Response:</b> <code>{ ppo_policy_gradient_loss, kl_penalty_metric }</code>"]:::align
        DPO["<strong>Direct Preference Optimization & SimPO/KTO</strong><br/>📥 <b>JSON Input:</b> <code>{ winning_log_probs, losing_log_probs, ref_log_probs }</code><br/>📤 <b>JSON Response:</b> <code>{ dpo_loss, implicit_reward_margins }</code>"]:::align

        Rollouts["<strong>Group Policy Rollout Sampler</strong><br/>📥 <b>JSON Input:</b> <code>{ reasoning_prompts, rollout_group_size_G }</code><br/>📤 <b>JSON Response:</b> <code>{ candidate_reasoning_traces }</code>"]:::align
        Sandbox["<strong>Deterministic Execution Sandbox (Python / Sympy / Unit Tests)</strong><br/>📥 <b>JSON Input:</b> <code>{ candidate_reasoning_traces, test_assertions }</code><br/>📤 <b>JSON Response:</b> <code>{ binary_outcome_rewards, execution_logs }</code>"]:::align
        GRPO["<strong>Group Relative Policy Optimization (GRPO)</strong><br/>📥 <b>JSON Input:</b> <code>{ group_rewards, group_policy_log_probs, group_ref_log_probs }</code><br/>📤 <b>JSON Response:</b> <code>{ group_advantages, grpo_loss, approx_kl }</code>"]:::align
        PRM["<strong>Process Reward Models (Step-Level Credit PRM)</strong><br/>📥 <b>JSON Input:</b> <code>{ step_decomposed_solutions, step_verifier }</code><br/>📤 <b>JSON Response:</b> <code>{ step_credit_scores, best_of_n_selection }</code>"]:::align
        Merge["<strong>TIES & DARE Multi-Task Model Merging</strong><br/>📥 <b>JSON Input:</b> <code>{ preference_model_weights, reasoning_model_weights }</code><br/>📤 <b>JSON Response:</b> <code>{ unified_frontier_aligned_weights }</code>"]:::align

        CheckpointSave -->|"transmits JSON: { updated_parameters }"| SFT
        SFT -->|"transmits JSON: { instruction_tuned_weights }"| BT
        BT -->|"transmits JSON: { winning_scalar_rewards }"| RLHF
        SFT -->|"transmits JSON: { instruction_tuned_weights }"| DPO
        SFT -->|"transmits JSON: { instruction_tuned_weights }"| Rollouts
        Rollouts -->|"transmits JSON: { candidate_reasoning_traces }"| Sandbox
        Sandbox -->|"transmits JSON: { binary_outcome_rewards }"| GRPO
        Rollouts -->|"transmits JSON: { candidate_reasoning_traces }"| PRM
        RLHF -->|"transmits JSON: { preference_checkpoint }"| Merge
        DPO -->|"transmits JSON: { preference_checkpoint }"| Merge
        GRPO -->|"transmits JSON: { reasoning_checkpoint }"| Merge
        PRM -->|"transmits JSON: { reasoning_checkpoint }"| Merge
    end

    %% -------------------------------------------------------------------------
    %% 6. PARAMETER-EFFICIENCY, COMPRESSION & HARDWARE SPARSITY
    %% -------------------------------------------------------------------------
    subgraph S6 ["6. Parameter-Efficiency, Compression & Hardware Sparsity"]
        LoRA["<strong>LoRA, QLoRA NF4 & DoRA Low-Rank Adaptation</strong><br/>📥 <b>JSON Input:</b> <code>{ frozen_dense_weights, adapter_matrices_A_B }</code><br/>📤 <b>JSON Response:</b> <code>{ low_rank_adapted_weights }</code>"]:::peft
        Distill["<strong>Knowledge Distillation (Soft Targets & On-Policy)</strong><br/>📥 <b>JSON Input:</b> <code>{ teacher_logits, student_logits, temperature }</code><br/>📤 <b>JSON Response:</b> <code>{ distillation_kl_loss, student_parameter_updates }</code>"]:::peft
        EWC["<strong>Elastic Weight Consolidation (EWC)</strong><br/>📥 <b>JSON Input:</b> <code>{ previous_task_weights, fisher_information_matrix }</code><br/>📤 <b>JSON Response:</b> <code>{ continual_retention_loss }</code>"]:::peft
        Wanda["<strong>Wanda One-Shot Activation-Weighted Pruning</strong><br/>📥 <b>JSON Input:</b> <code>{ weight_matrix, activation_norm_vector }</code><br/>📤 <b>JSON Response:</b> <code>{ structured_sparse_weight_matrix }</code>"]:::peft
        Sparse24["<strong>Semi-Structured 2:4 Sparse Tensor Cores</strong><br/>📥 <b>JSON Input:</b> <code>{ dense_vector, importance_metric }</code><br/>📤 <b>JSON Response:</b> <code>{ sparse_vector, compressed_values, group_2bit_indices }</code>"]:::peft

        Merge -->|"transmits JSON: { unified_frontier_aligned_weights }"| LoRA
        LoRA -->|"transmits JSON: { low_rank_adapted_weights }"| Distill
        Distill -->|"transmits JSON: { student_parameter_updates }"| EWC
        EWC -->|"transmits JSON: { memory_retained_weights }"| Wanda
        Wanda -->|"transmits JSON: { structured_sparse_weight_matrix }"| Sparse24
    end

    %% -------------------------------------------------------------------------
    %% 7. PRODUCTION SERVING, PAGEDATTENTION & GRAMMAR DECODING
    %% -------------------------------------------------------------------------
    subgraph S7 ["7. Production Serving, PagedAttention & Grammar Decoding"]
        Continuous["<strong>Continuous Dynamic Batching Scheduler</strong><br/>📥 <b>JSON Input:</b> <code>{ incoming_requests, max_batch_slots }</code><br/>📤 <b>JSON Response:</b> <code>{ active_generation_batch }</code>"]:::serve
        PagedKV["<strong>PagedAttention Virtual Memory Block Manager</strong><br/>📥 <b>JSON Input:</b> <code>{ block_table, logical_kv_cache, physical_pages }</code><br/>📤 <b>JSON Response:</b> <code>{ paged_attention_context }</code>"]:::serve
        FlashDecode["<strong>FlashDecoding Parallel Split-K Attention</strong><br/>📥 <b>JSON Input:</b> <code>{ split_k_queries, paged_keys, paged_values }</code><br/>📤 <b>JSON Response:</b> <code>{ reduced_attention_activations }</code>"]:::serve
        Speculative["<strong>Speculative Draft-Verify Decoding</strong><br/>📥 <b>JSON Input:</b> <code>{ draft_tokens_K, target_model_verification_logits }</code><br/>📤 <b>JSON Response:</b> <code>{ accepted_tokens, rollback_token_count }</code>"]:::decode
        Grammar["<strong>Constrained Grammar FSM (JSON / CFG Masking)</strong><br/>📥 <b>JSON Input:</b> <code>{ raw_next_token_logits, schema_grammar_fsm }</code><br/>📤 <b>JSON Response:</b> <code>{ grammar_constrained_logits }</code>"]:::serve
        Sampling["<strong>Greedy Temperature Sampling</strong><br/>📥 <b>JSON Input:</b> <code>{ logits, temperature }</code><br/>📤 <b>JSON Response:</b> <code>{ selected_token, probabilities, is_greedy }</code>"]:::serve

        Sparse24 -->|"transmits JSON: { compressed_values, group_2bit_indices }"| PagedKV
        Continuous -->|"transmits JSON: { active_generation_batch }"| PagedKV
        PagedKV -->|"transmits JSON: { paged_attention_context }"| FlashDecode
        FlashDecode -->|"transmits JSON: { reduced_attention_activations }"| Speculative
        Speculative -->|"transmits JSON: { accepted_tokens }"| Grammar
        Grammar -->|"transmits JSON: { grammar_constrained_logits }"| Sampling
        Sampling -->|"transmits JSON: { selected_token } to client streaming socket"| ClientOut["<strong>Streamed Token Output Stream</strong>"]:::serve
    end
```

---

### 4.2 Inter-Component Data Contract Transition Matrix

The table below details the exact input fields, output fields, data types, and consumer targets across the end-to-end pipeline:

| Pipeline Phase | Algorithm Component | Upstream Caller / Source | Exact Input Payload Fields & Types | Exact Output Payload Fields & Types | Downstream Consumer / Target | Invariant Certificate |
|---|---|---|---|---|---|---|
| **Data Curation** | `Synthetic Data Quality Filtering and Deduplication` | Synthetic Generation Worker Pool | • `candidate_samples`: `List[Dict]`<br>• `min_quality_score`: `float`<br>• `min_token_length`: `int`<br>• `max_token_length`: `int`<br>• `jaccard_dedup_threshold`: `float` | • `accepted_samples`: `List[Dict]`<br>• `rejection_breakdown`: `Dict[str, int]`<br>• `acceptance_ratio`: `float`<br>• `total_candidates_processed`: `int` | `Sequence Packing` & Tokenizer Ingestion | Only candidate samples satisfying quality, length bounds, and sub-threshold Jaccard similarity are accepted. |
| **Data Mixture** | `DoReMi Dynamic Domain Reweighting` | Pre-Training Scheduler / Proxy Evaluator | • `domain_losses_proxy`: `List[float]`<br>• `domain_losses_reference`: `List[float]`<br>• `current_domain_weights`: `List[float]`<br>• `step_size_eta`: `float`<br>• `smoothing_epsilon`: `float` | • `updated_domain_weights`: `List[float]`<br>• `excess_losses`: `List[float]`<br>• `worst_case_domain_idx`: `int`<br>• `max_excess_loss`: `float`<br>• `num_domains`: `int` | Distributed DataLoader Sampler | Domain weights are updated via exponentiated excess loss and normalized to a valid simplex summing to 1. |
| **Sequence Packing** | `Sequence Packing with Causal Document Masking` | Tokenized Corpus Reader | • `document_token_sequences`: `List[List[int]]`<br>• `max_sequence_length`: `int`<br>• `padding_token_id`: `int` | • `packed_sequences`: `List[List[int]]`<br>• `segment_ids`: `List[List[int]]`<br>• `position_ids`: `List[List[int]]`<br>• `document_boundary_masks`: `List[List[List[int]]]`<br>• `packing_efficiency`: `float`<br>• `num_packed_sequences`: `int` | Embedding Lookup & Attention Engine | Sequence packing preserves intra-document token causality while cross-document attention is strictly zeroed. |
| **Normalization** | `Root Mean Square Normalization (RMSNorm)` | Layer Input / Residual Stream | • `input_tensor`: `List[List[float]]`<br>• `gamma`: `List[float]`<br>• `eps`: `float` | • `normalized_tensor`: `List[List[float]]`<br>• `rms_values`: `List[float]` | Linear QKV GEMM Projections | Normalized activations maintain unit root-mean-square magnitude scaled by gamma weights. |
| **Position Encoding** | `Rotary Position Embeddings (RoPE)` | Q & K Projection GEMMs | • `vectors`: `List[List[float]]`<br>• `positions`: `List[int]`<br>• `base`: `float` | • `rotated_vectors`: `List[List[float]]`<br>• `dim`: `int` | FlashAttention Causal Kernel | Rotary transformation preserves inner-product norm while embedding relative positional distance. |
| **Attention** | `Multi-Head Attention` | Q, K, V Rotated Tensors | • `query`: `List[List[float]]`<br>• `key`: `List[List[float]]`<br>• `value`: `List[List[float]]`<br>• `num_heads`: `int`<br>• `mask`: `List[List[float]]` (optional) | • `output`: `List[List[float]]`<br>• `head_outputs`: `List[List[List[float]]]`<br>• `head_dim`: `int` | Output Projection Matrix $W_o$ | Multi-head attention outputs convex combinations of value vectors weighted by softmax scores. |
| **Tensor Parallelism** | `Megatron Tensor Parallelism (Column/Row GEMM)` | Intra-Node Attention / MLP Block | • `input_vector`: `List[float]`<br>• `column_shards`: `List[List[List[float]]]`<br>• `row_shards`: `List[List[List[float]]]` | • `output_vector`: `List[float]`<br>• `shard_hidden_activations`: `List[List[float]]`<br>• `tp_degree`: `int`<br>• `communication_volume_elements`: `int` | Residual Stream Addition | Column-parallel GEMM followed by row-parallel GEMM and All-Reduce reproduces exact dense projection. |
| **Batch Noise Scaling** | `Critical Batch Size Noise Scale Estimator` | Gradient Accumulator Engine | • `per_sample_gradients`: `List[List[float]]`<br>• `current_step`: `int`<br>• `total_steps`: `int`<br>• `min_batch_size`: `int`<br>• `max_batch_size`: `int` | • `estimated_noise_scale_b_noise`: `float`<br>• `gradient_trace_covariance`: `float`<br>• `mean_gradient_squared_norm`: `float`<br>• `recommended_batch_size`: `int`<br>• `compute_efficiency`: `float` | Cluster Job Dynamic Batch Tuner | Critical batch size $B_{\text{noise}}$ is estimated from the ratio of gradient covariance trace to squared mean norm. |
| **Optimization** | `AdamW Decoupled Weight Decay Optimizer` | Autodiff VJP Backward Engine | • `parameters`: `List[float]`<br>• `gradients`: `List[float]`<br>• `exp_avg`: `List[float]`<br>• `exp_avg_sq`: `List[float]`<br>• `step`: `int`<br>• `lr`: `float`<br>• `beta1`: `float`<br>• `beta2`: `float`<br>• `eps`: `float`<br>• `weight_decay`: `float` | • `updated_parameters`: `List[float]`<br>• `updated_exp_avg`: `List[float]`<br>• `updated_exp_avg_sq`: `List[float]` | Model Weight Parameter Buffer | AdamW decouples $L_2$ weight decay from first and second gradient moment moving averages. |
| **Alignment (Reasoning)** | `Group Relative Policy Optimization (GRPO)` | Candidate Rollout Sampler & Ground Truth Verifier | • `group_rewards`: `List[float]`<br>• `group_policy_log_probs`: `List[List[float]]`<br>• `group_old_log_probs`: `List[List[float]]`<br>• `group_ref_log_probs`: `List[List[float]]`<br>• `clip_epsilon`: `float`<br>• `kl_coeff`: `float` | • `group_advantages`: `List[float]`<br>• `mean_group_reward`: `float`<br>• `std_group_reward`: `float`<br>• `grpo_loss`: `float`<br>• `approx_kl`: `float` | Autodiff Policy Gradient Engine | Group advantages are normalized to zero-mean and unit-variance across group rollouts without a critic. |
| **Hardware Sparsity** | `Semi-Structured 2:4 Sparse Tensor Cores` | Trained Dense Weight Matrix | • `dense_vector`: `List[float]`<br>• `importance_metric`: `str` | • `sparse_vector`: `List[float]`<br>• `compressed_values`: `List[float]`<br>• `group_2bit_indices`: `List[List[int]]`<br>• `compression_ratio`: `float`<br>• `block_count`: `int` | NVIDIA Sparse Tensor Core GEMM | Exactly 2 out of every 4 contiguous elements are zeroed, achieving 50% storage compression. |
| **Serving Decoding** | `Greedy Temperature Sampling` | Logit Projection Layer | • `logits`: `List[float]`<br>• `temperature`: `float` | • `selected_token`: `int`<br>• `probabilities`: `List[float]`<br>• `is_greedy`: `bool` | Client Streaming Output Socket | Temperature-scaled softmax distribution produces deterministic argmax at $T=0$. |

---

### 4.3 End-to-End Entity-Relationship Architecture & Concrete JSON Schema Contracts

The Entity-Relationship Diagram below maps the data entities, primary/foreign keys, and relational cardinality across the entire computational pipeline. Each entity is accompanied by its concrete, production JSON schema:

```mermaid
erDiagram
    DATASET_SAMPLE ||--o{ PACKED_SEQUENCE : packs_into
    PACKED_SEQUENCE ||--|{ TENSOR_ACTIVATION : embeds_to
    TENSOR_ACTIVATION }|--|| LAYER_WEIGHT : multiplies_with
    LAYER_WEIGHT ||--|| OPTIMIZER_STATE : updated_by
    LAYER_WEIGHT ||--o{ CHECKPOINT_STATE : saved_to
    CHECKPOINT_STATE ||--o{ POLICY_ROLLOUT : generates
    POLICY_ROLLOUT ||--|| ADVANTAGE_TARGET : evaluates_into
    CHECKPOINT_STATE ||--o{ COMPRESSED_WEIGHT : pruned_to
    COMPRESSED_WEIGHT ||--|{ KV_CACHE_PAGE : serves_with
    KV_CACHE_PAGE ||--o{ GENERATION_TOKEN : emits

    DATASET_SAMPLE {
        string sample_id PK
        string text
        float quality_score
        int token_count
        float dedup_hash
    }
    PACKED_SEQUENCE {
        string sequence_id PK
        string sample_id FK
        array token_ids
        array segment_ids
        array position_ids
        array boundary_mask
    }
    TENSOR_ACTIVATION {
        string activation_id PK
        int batch_size
        int hidden_dim
        array normalized_tensor
        array rms_values
    }
    LAYER_WEIGHT {
        string weight_id PK
        string layer_name
        array dense_parameters
        array column_shards
        array row_shards
    }
    OPTIMIZER_STATE {
        string weight_id FK
        int step
        float learning_rate
        array exp_avg_m
        array exp_avg_sq_v
        float weight_decay
    }
    CHECKPOINT_STATE {
        string checkpoint_id PK
        int global_step
        float loss_metric
        array model_weights
    }
    POLICY_ROLLOUT {
        string rollout_id PK
        string checkpoint_id FK
        string prompt
        array reasoning_trace
        array policy_log_probs
        float reward_score
    }
    ADVANTAGE_TARGET {
        string rollout_id FK
        float group_advantage
        float ppo_grpo_loss
        float kl_divergence
    }
    COMPRESSED_WEIGHT {
        string weight_id FK
        array sparse_2_4_values
        array group_2bit_indices
        float compression_ratio
    }
    KV_CACHE_PAGE {
        string page_id PK
        int logical_seq_id
        int physical_page_ptr
        int block_capacity
    }
    GENERATION_TOKEN {
        int token_id PK
        string page_id FK
        float probability
        boolean is_greedy
    }
```

#### Entity JSON Data Schemas

##### 1. `DATASET_SAMPLE` (Synthetic Data Quality & Filtering Entity)
```json
{
  "": "https://json-schema.org/draft/2020-12/schema",
  "title": "DATASET_SAMPLE",
  "type": "object",
  "required": ["sample_id", "text", "quality_score", "token_count", "dedup_hash"],
  "properties": {
    "sample_id": { "type": "string", "description": "Unique UUID identifier for candidate sample" },
    "text": { "type": "string", "description": "Raw string text content" },
    "quality_score": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Synthetic generation heuristic or classifier quality score" },
    "token_count": { "type": "integer", "minimum": 1, "description": "Token count under target tokenizer" },
    "dedup_hash": { "type": "number", "description": "MinHash / SimHash 64-bit fingerprint float" }
  }
}
```

##### 2. `PACKED_SEQUENCE` (Causal Document Boundary Packing Entity)
```json
{
  "": "https://json-schema.org/draft/2020-12/schema",
  "title": "PACKED_SEQUENCE",
  "type": "object",
  "required": ["sequence_id", "sample_id", "token_ids", "segment_ids", "position_ids", "boundary_mask"],
  "properties": {
    "sequence_id": { "type": "string", "description": "Unique packed sequence batch ID" },
    "sample_id": { "type": "string", "description": "Foreign key reference to originating dataset sample" },
    "token_ids": { "type": "array", "items": { "type": "integer" }, "description": "Concatenated integer token IDs up to max sequence length" },
    "segment_ids": { "type": "array", "items": { "type": "integer" }, "description": "Document segment membership index per token" },
    "position_ids": { "type": "array", "items": { "type": "integer" }, "description": "Resetting per-document 0-indexed position coordinates" },
    "boundary_mask": { "type": "array", "items": { "type": "array", "items": { "type": "integer" } }, "description": "Causal block-diagonal attention mask matrix" }
  }
}
```

##### 3. `TENSOR_ACTIVATION` (Micro-Architecture Activation State Entity)
```json
{
  "": "https://json-schema.org/draft/2020-12/schema",
  "title": "TENSOR_ACTIVATION",
  "type": "object",
  "required": ["activation_id", "batch_size", "hidden_dim", "normalized_tensor", "rms_values"],
  "properties": {
    "activation_id": { "type": "string", "description": "Unique activation buffer snapshot identifier" },
    "batch_size": { "type": "integer", "minimum": 1 },
    "hidden_dim": { "type": "integer", "minimum": 1 },
    "normalized_tensor": { "type": "array", "items": { "type": "array", "items": { "type": "number" } }, "description": "Unit-scale activations post-RMSNorm/RoPE" },
    "rms_values": { "type": "array", "items": { "type": "number" }, "description": "Per-row root-mean-square norm scalers" }
  }
}
```

##### 4. `LAYER_WEIGHT` (5D Distributed Model Parameters Entity)
```json
{
  "": "https://json-schema.org/draft/2020-12/schema",
  "title": "LAYER_WEIGHT",
  "type": "object",
  "required": ["weight_id", "layer_name", "dense_parameters", "column_shards", "row_shards"],
  "properties": {
    "weight_id": { "type": "string", "description": "Unique parameter tensor key" },
    "layer_name": { "type": "string", "description": "Dot-delimited module path e.g. transformer.layers.14.attention.q_proj" },
    "dense_parameters": { "type": "array", "items": { "type": "number" }, "description": "Unsharded flattened model parameter weights" },
    "column_shards": { "type": "array", "items": { "type": "array", "items": { "type": "number" } }, "description": "TP column-partitioned GEMM parameter matrix" },
    "row_shards": { "type": "array", "items": { "type": "array", "items": { "type": "number" } }, "description": "TP row-partitioned GEMM parameter matrix" }
  }
}
```

##### 5. `OPTIMIZER_STATE` (AdamW Decoupled Optimization State Entity)
```json
{
  "": "https://json-schema.org/draft/2020-12/schema",
  "title": "OPTIMIZER_STATE",
  "type": "object",
  "required": ["weight_id", "step", "learning_rate", "exp_avg_m", "exp_avg_sq_v", "weight_decay"],
  "properties": {
    "weight_id": { "type": "string", "description": "Foreign key reference to LAYER_WEIGHT" },
    "step": { "type": "integer", "minimum": 1, "description": "Current optimization step iteration" },
    "learning_rate": { "type": "number", "minimum": 0.0, "description": "Current effective learning rate from WSD schedule" },
    "exp_avg_m": { "type": "array", "items": { "type": "number" }, "description": "First-moment moving average buffer vector" },
    "exp_avg_sq_v": { "type": "array", "items": { "type": "number" }, "description": "Second-moment moving average buffer vector" },
    "weight_decay": { "type": "number", "minimum": 0.0, "description": "Decoupled weight decay coefficient" }
  }
}
```

##### 6. `POLICY_ROLLOUT` (Post-Training Reasoning Trajectory Entity)
```json
{
  "": "https://json-schema.org/draft/2020-12/schema",
  "title": "POLICY_ROLLOUT",
  "type": "object",
  "required": ["rollout_id", "checkpoint_id", "prompt", "reasoning_trace", "policy_log_probs", "reward_score"],
  "properties": {
    "rollout_id": { "type": "string", "description": "Unique trajectory rollout identifier" },
    "checkpoint_id": { "type": "string", "description": "Foreign key reference to CHECKPOINT_STATE" },
    "prompt": { "type": "string", "description": "Input instruction prompt" },
    "reasoning_trace": { "type": "array", "items": { "type": "string" }, "description": "Step-by-step thinking tokens or sub-clauses" },
    "policy_log_probs": { "type": "array", "items": { "type": "number" }, "description": "Per-token log probabilities under active policy" },
    "reward_score": { "type": "number", "description": "Scalar reward from verifier sandbox or reward model" }
  }
}
```

##### 7. `COMPRESSED_WEIGHT` (Sparse & Quantized Hardware Representation Entity)
```json
{
  "": "https://json-schema.org/draft/2020-12/schema",
  "title": "COMPRESSED_WEIGHT",
  "type": "object",
  "required": ["weight_id", "sparse_2_4_values", "group_2bit_indices", "compression_ratio"],
  "properties": {
    "weight_id": { "type": "string", "description": "Foreign key reference to LAYER_WEIGHT" },
    "sparse_2_4_values": { "type": "array", "items": { "type": "number" }, "description": "Non-zero values compressed to 50% density" },
    "group_2bit_indices": { "type": "array", "items": { "type": "array", "items": { "type": "integer" } }, "description": "2-bit position indices for hardware Sparse Tensor Cores" },
    "compression_ratio": { "type": "number", "default": 0.5 }
  }
}
```

##### 8. `KV_CACHE_PAGE` & `GENERATION_TOKEN` (Paged Serving & Output Entity)
```json
{
  "": "https://json-schema.org/draft/2020-12/schema",
  "title": "KV_CACHE_PAGE",
  "type": "object",
  "required": ["page_id", "logical_seq_id", "physical_page_ptr", "block_capacity"],
  "properties": {
    "page_id": { "type": "string", "description": "Physical GPU memory page index" },
    "logical_seq_id": { "type": "integer", "description": "Active request sequence slot" },
    "physical_page_ptr": { "type": "integer", "description": "Memory address pointer to pinned page table" },
    "block_capacity": { "type": "integer", "default": 16, "description": "Tokens accommodated per physical cache block" }
  }
}
```

```json
{
  "": "https://json-schema.org/draft/2020-12/schema",
  "title": "GENERATION_TOKEN",
  "type": "object",
  "required": ["token_id", "page_id", "probability", "is_greedy"],
  "properties": {
    "token_id": { "type": "integer", "minimum": 0, "description": "Sampled output vocabulary token ID" },
    "page_id": { "type": "string", "description": "Foreign key reference to KV_CACHE_PAGE" },
    "probability": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Softmax probability of selected token" },
    "is_greedy": { "type": "boolean", "description": "True if selected via deterministic argmax (T=0)" }
  }
}
```

