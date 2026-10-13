"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: REST API V1 UNIFIED FACADE ROUTER
================================================================================

1. OVERVIEW & OBJECTIVE:
   Unified facade aggregator composing all modular role-based routers for REST API V1:
   - System & Domain Router (System health, RAG, AI agents, audit, graph, catalog)
   - Search Algorithms Router (ALGO-SRCH-01..15)
   - Observability Algorithms Router (ALGO-OBS-16..21)
   - Update Algorithms Router (ALGO-UPD-22..24)
   - Base Vector Algorithms Router (ALGO-VEC-01..09)
   - Graph Algorithms Router (ALGO-GRAPH-01..09)
   - Vector Filter Algorithms Router (ALGO-VEC-FLTR-80..84)
   - Vector Search Algorithms Router (ALGO-VEC-SRCH-51..110)
   - Vector Transform Algorithms Router (ALGO-VEC-TRFM-01..50)

2. ZERO-INLINE-COMMENT DOCTRINE & COMPLIANCE:
   Decomposes the monolithic router into modular role-specific sub-routers while
   maintaining 100% backward compatibility, preserving all public DTOs, endpoint
   handlers, and dependency injection factories.
================================================================================
"""

from fastapi import APIRouter
from .dependencies import get_services, get_code_engine_service, get_orchestrator_services

router = APIRouter(prefix="/api/v1")

from .routers.system_router import router as system_router
router.include_router(system_router)

from .routers.search_router import router as search_router
router.include_router(search_router)

from .routers.observability_router import router as observability_router
router.include_router(observability_router)

from .routers.update_router import router as update_router
router.include_router(update_router)

from .routers.vector_router import router as vector_router
router.include_router(vector_router)

from .routers.graph_router import router as graph_router
router.include_router(graph_router)

from .routers.vector_filter_router import router as vector_filter_router
router.include_router(vector_filter_router)

from .routers.vector_search_router import router as vector_search_router
router.include_router(vector_search_router)

from .routers.vector_transform_router import router as vector_transform_router
router.include_router(vector_transform_router)

from .routers.vector_update_router import router as vector_update_router
router.include_router(vector_update_router)

from .routers.vector_observability_router import router as vector_observability_router
router.include_router(vector_observability_router)

from .routers.graph_representation_traversal_router import router as graph_representation_traversal_router
router.include_router(graph_representation_traversal_router)

from .routers.graph_connectivity_flows_router import router as graph_connectivity_flows_router
router.include_router(graph_connectivity_flows_router)

from .routers.graph_centrality_dense_router import router as graph_centrality_dense_router
router.include_router(graph_centrality_dense_router)

from .routers.graph_communities_spectral_router import router as graph_communities_spectral_router
router.include_router(graph_communities_spectral_router)

from .routers.graph_dynamic_streaming_router import router as graph_dynamic_streaming_router
router.include_router(graph_dynamic_streaming_router)

from .routers.graph_embeddings_gnn_router import router as graph_embeddings_gnn_router
router.include_router(graph_embeddings_gnn_router)

from .routers.graph_systems_router import router as graph_systems_router
router.include_router(graph_systems_router)

from .routers.knowledge_graph_modeling_storage_router import router as knowledge_graph_modeling_storage_router
router.include_router(knowledge_graph_modeling_storage_router)

from .routers.knowledge_graph_query_reasoning_router import router as knowledge_graph_query_reasoning_router
router.include_router(knowledge_graph_query_reasoning_router)

from .routers.knowledge_graph_embeddings_gnn_router import router as knowledge_graph_embeddings_gnn_router
router.include_router(knowledge_graph_embeddings_gnn_router)

from .routers.knowledge_graph_ops_llm_observability_router import router as knowledge_graph_ops_llm_observability_router
router.include_router(knowledge_graph_ops_llm_observability_router)

from .routers.code_engine_diff_buffer_router import router as code_engine_diff_buffer_router
router.include_router(code_engine_diff_buffer_router)

from .routers.code_engine_mutation_router import router as code_engine_mutation_router
router.include_router(code_engine_mutation_router)

from .routers.algorithm_registry_router import router as algorithm_registry_router
router.include_router(algorithm_registry_router)

from .routers.folder_scaffold_router import router as folder_scaffold_router
router.include_router(folder_scaffold_router)

# -----------------------------------------------------------------------------
# RE-EXPORTS FOR FULL BACKWARD COMPATIBILITY
# -----------------------------------------------------------------------------

from .routers.system_router import (
    GraphQueryDTO,
    AlgoComposeRequestDTO,
    AlgoExecuteRequestDTO,
    health_check,
    index_policies,
    search_policies,
    run_agent,
    list_declarative_agents,
    run_specialized_agent,
    scan_repository,
    execute_algorithm_direct,
    build_graph,
    query_graph,
    get_rule_impact,
    list_algorithm_contracts,
    get_algorithm_contract,
    list_type_adapters,
    compose_algorithm_pipeline,
)

from .routers.search_router import (
    AlgoScanDTO,
    FilePathDTO,
    DirectoryPathDTO,
    SearchWalkDTO,
    SearchWorkStealingDTO,
    SearchGitAwareDTO,
    SearchGlobDTO,
    SearchSizeLineDTO,
    SearchTrigramDTO,
    SearchSimdMemchrDTO,
    SearchAhoCorasickDTO,
    SearchLazyDfaDTO,
    SearchContextSnippetDTO,
    SearchMmapDTO,
    search_recursive_walk,
    search_work_stealing_walk,
    search_git_aware_walk,
    search_glob_match,
    search_binary_check,
    search_content_type,
    search_size_line_check,
    search_generated_code_check,
    search_trigram_index,
    search_simd_memchr,
    search_aho_corasick,
    search_lazy_dfa,
    search_streaming_chunk_scan,
    search_context_snippet,
    search_mmap_scan,
    scan_multipattern,
)

from .routers.observability_router import (
    ObsSpanTrackDTO,
    ObsAstDTO,
    ObsSymbolsDTO,
    obs_span_track,
    obs_ast_extract,
    obs_symbols_resolve,
    lint_comments,
    analyze_dependencies,
    generate_file_outline,
)

from .routers.update_router import (
    PatchOperationDTO,
    BatchPatchRequestDTO,
    DiffRequestDTO,
    UpdateCstMatchDTO,
    update_cst_match,
    apply_patch,
    generate_diff,
)

from .routers.vector_router import (
    VectorNormalizeDTO,
    VectorCenterDTO,
    VectorLayerNormDTO,
    VectorScaleDTO,
    VectorSliceDTO,
    VectorScalarQuantizeDTO,
    VectorBinaryQuantizeDTO,
    VectorPoolDTO,
    VectorChunkDTO,
    normalize_vector_endpoint,
    center_vectors_endpoint,
    layer_norm_endpoint,
    scale_vector_endpoint,
    slice_vector_endpoint,
    quantize_scalar_endpoint,
    quantize_binary_endpoint,
    pool_tokens_endpoint,
    chunk_text_endpoint,
)

from .routers.graph_router import (
    GraphBfsDTO,
    GraphDfsDTO,
    GraphDijkstraDTO,
    GraphAstarDTO,
    GraphPageRankDTO,
    GraphDegreeCentralityDTO,
    GraphConnectedComponentsDTO,
    GraphTarjanSccDTO,
    GraphSubgraphMatchDTO,
    graph_bfs_endpoint,
    graph_dfs_endpoint,
    graph_dijkstra_endpoint,
    graph_astar_endpoint,
    graph_pagerank_endpoint,
    graph_degree_centrality_endpoint,
    graph_connected_components_endpoint,
    graph_tarjan_scc_endpoint,
    graph_subgraph_match_endpoint,
)

from .routers.vector_filter_router import (
    VecFilterPreFilterDTO,
    VecFilterPostFilterDTO,
    VecFilterInGraphDTO,
    VecFilterSelectivityPlanDTO,
    VecFilterPartitionedDTO,
    vector_filter_pre_filter_endpoint,
    vector_filter_post_filter_endpoint,
    vector_filter_in_graph_endpoint,
    vector_filter_selectivity_plan_endpoint,
    vector_filter_partitioned_endpoint,
)

from .routers.vector_search_router import (
    VecSearchBruteForceGemmDTO,
    VecSearchSimdDistanceDTO,
    VecSearchHeapTopKDTO,
    VecSearchRadixTopKDTO,
    VecSearchEarlyAbandonDTO,
    VecSearchPivotPruneDTO,
    VecSearchKdTreeDTO,
    VecSearchBallTreeDTO,
    VecSearchVpTreeDTO,
    VecSearchRpForestDTO,
    VecSearchIvfDTO,
    VecSearchIvfPqDTO,
    VecSearchNprobeTunerDTO,
    VecSearchInvertedMultiIndexDTO,
    VecSearchNswDTO,
    VecSearchHnswSearchDTO,
    VecSearchHnswInsertDTO,
    VecSearchBeamSearchDTO,
    VecSearchVamanaDTO,
    VecSearchRobustPruneDTO,
    VecSearchNsgDTO,
    VecSearchCagraDTO,
    VecSearchEntryPointDTO,
    VecSearchConnectivityRepairDTO,
    VecSearchFilteredDiskannDTO,
    VecSearchSpannDTO,
    VecSearchRandomHyperplaneLshDTO,
    VecSearchMultiProbeLshDTO,
    VecSearchE2LshDTO,
    VecSearchBM25DTO,
    VecSearchSparseDenseHybridDTO,
    VecSearchRRFDTO,
    VecSearchConvexScoreFusionDTO,
    VecSearchMMRDTO,
    VecSearchRangeSearchDTO,
    VecSearchMaxSimDTO,
    VecSearchMultiQueryExpansionDTO,
    VecSearchFullPrecisionRescoreDTO,
    VecSearchCrossEncoderRerankDTO,
    VecSearchMultiStageFunnelDTO,
    VecSearchLLMListwiseRerankDTO,
    VecSearchHyDEDTO,
    VecSearchQueryRoutingDTO,
    VecSearchScatterGatherDTO,
    VecSearchPartitionAwareRoutingDTO,
    VecSearchReplicationLoadBalancerDTO,
    VecSearchHedgedRequestsDTO,
    VecSearchKWayMergeDTO,
    VecSearchQueryCacheDTO,
    VecSearchSemanticCacheDTO,
    VecSearchQueryBatchingDTO,
    VecSearchMemoryTieringDTO,
    VecSearchDiskIOSchedulerDTO,
    VecSearchAdmissionControlDTO,
    VecSearchSearchAutotuneDTO,
    vector_search_gemm_endpoint,
    vector_search_simd_dist_endpoint,
    vector_search_topk_endpoint,
    vector_search_radix_topk_endpoint,
    vector_search_early_abandon_endpoint,
    vector_search_pivot_prune_endpoint,
    vector_search_kdtree_endpoint,
    vector_search_ball_tree_endpoint,
    vector_search_vptree_endpoint,
    vector_search_rp_forest_endpoint,
    vector_search_ivf_endpoint,
    vector_search_ivf_pq_endpoint,
    vector_search_nprobe_tune_endpoint,
    vector_search_imi_endpoint,
    vector_search_nsw_endpoint,
    vector_search_hnsw_search_endpoint,
    vector_search_hnsw_insert_endpoint,
    vector_search_beam_search_endpoint,
    vector_search_vamana_endpoint,
    vector_search_robust_prune_endpoint,
    vector_search_nsg_endpoint,
    vector_search_cagra_endpoint,
    vector_search_entry_point_endpoint,
    vector_search_connectivity_repair_endpoint,
    vector_search_filtered_diskann_endpoint,
    vector_search_spann_endpoint,
    vector_search_random_hyperplane_lsh_endpoint,
    vector_search_multi_probe_lsh_endpoint,
    vector_search_e2lsh_endpoint,
    vector_search_bm25_endpoint,
    vector_search_sparse_dense_hybrid_endpoint,
    vector_search_rrf_endpoint,
    vector_search_convex_score_fusion_endpoint,
    vector_search_mmr_endpoint,
    vector_search_range_search_endpoint,
    vector_search_maxsim_endpoint,
    vector_search_multi_query_expansion_endpoint,
    vector_search_full_precision_rescore_endpoint,
    vector_search_cross_encoder_rerank_endpoint,
    vector_search_multi_stage_funnel_endpoint,
    vector_search_llm_listwise_rerank_endpoint,
    vector_search_hyde_endpoint,
    vector_search_query_routing_endpoint,
    vector_search_scatter_gather_endpoint,
    vector_search_partition_aware_routing_endpoint,
    vector_search_replication_load_balancer_endpoint,
    vector_search_hedged_requests_endpoint,
    vector_search_kway_merge_endpoint,
    vector_search_query_cache_endpoint,
    vector_search_semantic_cache_endpoint,
    vector_search_query_batching_endpoint,
    vector_search_memory_tiering_endpoint,
    vector_search_disk_io_scheduler_endpoint,
    vector_search_admission_control_endpoint,
    vector_search_search_autotune_endpoint,
)

from .routers.vector_transform_router import (
    VecTransformSubwordTokenizationDTO,
    VecTransformBiEncoderForwardDTO,
    VecTransformMeanPoolingDTO,
    VecTransformCLSPoolingDTO,
    VecTransformLastTokenPoolingDTO,
    VecTransformInstructionPrefixesDTO,
    VecTransformContrastiveInfoNCEDTO,
    VecTransformHardNegativeMiningDTO,
    VecTransformMatryoshkaLearningDTO,
    VecTransformLateChunkingDTO,
    VecTransformSlidingWindowDTO,
    VecTransformSemanticChunkingDTO,
    VecTransformRecursiveChunkingDTO,
    VecTransformDynamicPaddingBatchingDTO,
    VecTransformL2NormDTO,
    VecTransformMeanCenteringDTO,
    VecTransformWhiteningDTO,
    VecTransformRemoveDominantDirectionsDTO,
    VecTransformMIPSToNNSDTO,
    VecTransformScoreCalibrationDTO,
    VecTransformCSLSHubnessReductionDTO,
    VecTransformProcrustesAlignmentDTO,
    VecTransformPCADTO,
    VecTransformTruncatedSVDDTO,
    VecTransformRandomProjectionDTO,
    VecTransformAutoencoderCompressionDTO,
    VecTransformUMAPDTO,
    VecTransformTSNEDTO,
    VecTransformProjectionHeadDTO,
    VecTransformIncrementalPCADTO,
    VecTransformSimHashDTO,
    VecTransformLearnedSparseExpansionDTO,
    VecTransformScalarQuantizationDTO,
    VecTransformBinaryQuantizationDTO,
    VecTransformProductQuantizationDTO,
    VecTransformOptimizedProductQuantizationDTO,
    VecTransformResidualQuantizationDTO,
    VecTransformAnisotropicQuantizationDTO,
    VecTransformKMeansClusteringDTO,
    VecTransformKMeansPlusPlusDTO,
    VecTransformMinibatchKMeansDTO,
    VecTransformHierarchicalKMeansDTO,
    VecTransformADCLookupDTO,
    VecTransformFastScanPQDTO,
    VecTransformRaBiTQDTO,
    VecTransformHalfPrecisionDTO,
    VecTransformMultiVectorRepresentationDTO,
    VecTransformMultiVectorCompressionDTO,
    VecTransformSparseVectorRepresentationDTO,
    VecTransformEmbeddingCacheDTO,
    vector_transform_subword_tokenization_endpoint,
    vector_transform_bi_encoder_forward_endpoint,
    vector_transform_mean_pooling_endpoint,
    vector_transform_cls_pooling_endpoint,
    vector_transform_last_token_pooling_endpoint,
    vector_transform_instruction_prefixes_endpoint,
    vector_transform_contrastive_infonce_endpoint,
    vector_transform_hard_negative_mining_endpoint,
    vector_transform_matryoshka_learning_endpoint,
    vector_transform_late_chunking_endpoint,
    vector_transform_sliding_window_endpoint,
    vector_transform_semantic_chunking_endpoint,
    vector_transform_recursive_chunking_endpoint,
    vector_transform_dynamic_padding_batching_endpoint,
    vector_transform_l2_norm_endpoint,
    vector_transform_mean_centering_endpoint,
    vector_transform_whitening_endpoint,
    vector_transform_remove_dominant_directions_endpoint,
    vector_transform_mips_to_nns_endpoint,
    vector_transform_score_calibration_endpoint,
    vector_transform_csls_hubness_reduction_endpoint,
    vector_transform_procrustes_alignment_endpoint,
    vector_transform_pca_endpoint,
    vector_transform_truncated_svd_endpoint,
    vector_transform_random_projection_endpoint,
    vector_transform_autoencoder_compression_endpoint,
    vector_transform_umap_endpoint,
    vector_transform_tsne_endpoint,
    vector_transform_projection_head_endpoint,
    vector_transform_incremental_pca_endpoint,
    vector_transform_simhash_endpoint,
    vector_transform_learned_sparse_expansion_endpoint,
    vector_transform_scalar_quantization_endpoint,
    vector_transform_binary_quantization_endpoint,
    vector_transform_product_quantization_endpoint,
    vector_transform_optimized_product_quantization_endpoint,
    vector_transform_residual_quantization_endpoint,
    vector_transform_anisotropic_quantization_endpoint,
    vector_transform_kmeans_clustering_endpoint,
    vector_transform_kmeans_plus_plus_endpoint,
    vector_transform_minibatch_kmeans_endpoint,
    vector_transform_hierarchical_kmeans_endpoint,
    vector_transform_adc_lookup_endpoint,
    vector_transform_fast_scan_pq_endpoint,
    vector_transform_rabitq_endpoint,
    vector_transform_half_precision_endpoint,
    vector_transform_multi_vector_representation_endpoint,
    vector_transform_multi_vector_compression_endpoint,
    vector_transform_sparse_vector_representation_endpoint,
    vector_transform_embedding_cache_endpoint,
)
from .routers.vector_update_router import (
    VecUpdateUpsertStableIdDTO,
    VecUpdateWalDTO,
    VecUpdateFreshBufferDTO,
    VecUpdateLsmStorageDTO,
    VecUpdateSegmentCompactionDTO,
    VecUpdateTombstoneDeletionDTO,
    VecUpdateHnswDeletionRepairDTO,
    VecUpdateFreshDiskannUpdateDTO,
    VecUpdateIncrementalIvfDTO,
    VecUpdateCentroidDriftDTO,
    VecUpdateReembeddingPipelineDTO,
    VecUpdateDualWriteDTO,
    VecUpdateBlueGreenSwapDTO,
    VecUpdateIdempotentIngestionDTO,
    VecUpdateCdcDTO,
    VecUpdateTransactionalOutboxDTO,
    VecUpdateMerkleTreeSyncDTO,
    VecUpdateWatermarksFreshnessDTO,
    VecUpdateIdempotencyKeysDTO,
    VecUpdateBackfillCheckpointsDTO,
    VecUpdateMicroBatchingDTO,
    VecUpdateBackpressurePriorityDTO,
    VecUpdateMvccSnapshotsDTO,
    VecUpdateConsistencyLevelsDTO,
    VecUpdateLeaderFollowerDTO,
    VecUpdateRaftConsensusDTO,
    VecUpdateQuorumReadsWritesDTO,
    VecUpdateSnapshotReplayRecoveryDTO,
    VecUpdateConsistentHashingDTO,
    VecUpdateVersionVectorsDTO,
    VecUpdateSchemaVersioningDTO,
    VecUpdateMultiTenantIsolationDTO,
    VecUpdateTtlExpiryDTO,
    VecUpdateOrphanGcDTO,
    VecUpdateNearDuplicateDedupeDTO,
    VecUpdateRebuildSchedulingDTO,
    VecUpdateOnlineIndexBuildDTO,
    VecUpdateBulkLoadingDTO,
    VecUpdateRequantizationMigrationDTO,
    VecUpdateDeletionVerificationDTO,
    VecUpdateBackwardCompatibleTrainingDTO,
    VecUpdateLazyReembeddingDTO,
    VecUpdateCodebookRetrainingDTO,
    VecUpdateMetadataIndexMaintenanceDTO,
    VecUpdateAtomicCommitDTO,
    vector_update_upsert_stable_id_endpoint,
    vector_update_wal_endpoint,
    vector_update_fresh_buffer_endpoint,
    vector_update_lsm_storage_endpoint,
    vector_update_segment_compaction_endpoint,
    vector_update_tombstone_deletion_endpoint,
    vector_update_hnsw_deletion_repair_endpoint,
    vector_update_fresh_diskann_update_endpoint,
    vector_update_incremental_ivf_endpoint,
    vector_update_centroid_drift_endpoint,
    vector_update_reembedding_pipeline_endpoint,
    vector_update_dual_write_endpoint,
    vector_update_blue_green_swap_endpoint,
    vector_update_idempotent_ingestion_endpoint,
    vector_update_cdc_endpoint,
    vector_update_transactional_outbox_endpoint,
    vector_update_merkle_tree_sync_endpoint,
    vector_update_watermarks_freshness_endpoint,
    vector_update_idempotency_keys_endpoint,
    vector_update_backfill_checkpoints_endpoint,
    vector_update_micro_batching_endpoint,
    vector_update_backpressure_priority_endpoint,
    vector_update_mvcc_snapshots_endpoint,
    vector_update_consistency_levels_endpoint,
    vector_update_leader_follower_endpoint,
    vector_update_raft_consensus_endpoint,
    vector_update_quorum_reads_writes_endpoint,
    vector_update_snapshot_replay_recovery_endpoint,
    vector_update_consistent_hashing_endpoint,
    vector_update_version_vectors_endpoint,
    vector_update_schema_versioning_endpoint,
    vector_update_multi_tenant_isolation_endpoint,
    vector_update_ttl_expiry_endpoint,
    vector_update_orphan_gc_endpoint,
    vector_update_near_duplicate_dedupe_endpoint,
    vector_update_rebuild_scheduling_endpoint,
    vector_update_online_index_build_endpoint,
    vector_update_bulk_loading_endpoint,
    vector_update_requantization_migration_endpoint,
    vector_update_deletion_verification_endpoint,
    vector_update_backward_compatible_training_endpoint,
    vector_update_lazy_reembedding_endpoint,
    vector_update_codebook_retraining_endpoint,
    vector_update_metadata_index_maintenance_endpoint,
    vector_update_atomic_commit_endpoint,
)

from .routers.vector_observability_router import (
    VecObsRecallAtKDTO,
    VecObsGroundTruthSamplingDTO,
    VecObsPrecisionAtKDTO,
    VecObsMrrDTO,
    VecObsNdcgDTO,
    VecObsHitRateDTO,
    VecObsRelativeDistanceErrorDTO,
    VecObsLlmAsJudgeDTO,
    VecObsGoldenQueryRegressionDTO,
    VecObsOnlineImplicitFeedbackDTO,
    VecObsInterleavingExperimentsDTO,
    VecObsFaithfulnessGroundednessDTO,
    VecObsCentroidShiftDTO,
    VecObsMmdDTO,
    VecObsPsiKsDriftDTO,
    VecObsSimilarityScoreDistributionDTO,
    VecObsVectorNormDistributionDTO,
    VecObsPartitionClusterBalanceDTO,
    VecObsHubnessMeasurementDTO,
    VecObsIntrinsicDimensionDTO,
    VecObsOutlierDetectionDTO,
    VecObsQueryOodDetectionDTO,
    VecObsLatencyHistogramsDTO,
    VecObsRedUseMethodsDTO,
    VecObsSloErrorBudgetBurnDTO,
    VecObsDistributedTracingDTO,
    VecObsFreshnessLagDTO,
    VecObsGraphIndexHealthDTO,
    VecObsTombstoneRatioDTO,
    VecObsCacheHitRatioMemoryDTO,
    VecObsCapacityPlanningLittlesLawDTO,
    VecObsConsumerLagDTO,
    VecObsCardinalitySafeLabelsDTO,
    VecObsMetricAnomalyDetectionDTO,
    VecObsQuantileSketchesDTO,
    VecObsQueryExplainDTO,
    VecObsRetrievalTraceLoggingDTO,
    VecObsEmbeddingVisualizationDTO,
    VecObsFailureClusteringDTO,
    VecObsCanaryProbesDTO,
    VecObsShadowTrafficComparisonDTO,
    VecObsDataLineageDTO,
    VecObsReconciliationChecksDTO,
    VecObsCostAccountingDTO,
    VecObsFeedbackImprovementLoopDTO,
    vector_observability_recall_at_k_endpoint,
    vector_observability_ground_truth_sampling_endpoint,
    vector_observability_precision_at_k_endpoint,
    vector_observability_mrr_endpoint,
    vector_observability_ndcg_endpoint,
    vector_observability_hit_rate_endpoint,
    vector_observability_relative_distance_error_endpoint,
    vector_observability_llm_as_judge_endpoint,
    vector_observability_golden_query_regression_endpoint,
    vector_observability_online_implicit_feedback_endpoint,
    vector_observability_interleaving_experiments_endpoint,
    vector_observability_faithfulness_groundedness_endpoint,
    vector_observability_centroid_shift_endpoint,
    vector_observability_mmd_endpoint,
    vector_observability_psi_ks_drift_endpoint,
    vector_observability_similarity_score_distribution_endpoint,
    vector_observability_vector_norm_distribution_endpoint,
    vector_observability_partition_cluster_balance_endpoint,
    vector_observability_hubness_measurement_endpoint,
    vector_observability_intrinsic_dimension_endpoint,
    vector_observability_outlier_detection_endpoint,
    vector_observability_query_ood_detection_endpoint,
    vector_observability_latency_histograms_endpoint,
    vector_observability_red_use_methods_endpoint,
    vector_observability_slo_error_budget_burn_endpoint,
    vector_observability_distributed_tracing_endpoint,
    vector_observability_freshness_lag_endpoint,
    vector_observability_graph_index_health_endpoint,
    vector_observability_tombstone_ratio_endpoint,
    vector_observability_cache_hit_ratio_memory_endpoint,
    vector_observability_capacity_planning_littles_law_endpoint,
    vector_observability_consumer_lag_endpoint,
    vector_observability_cardinality_safe_labels_endpoint,
    vector_observability_metric_anomaly_detection_endpoint,
    vector_observability_quantile_sketches_endpoint,
    vector_observability_query_explain_endpoint,
    vector_observability_retrieval_trace_logging_endpoint,
    vector_observability_embedding_visualization_endpoint,
    vector_observability_failure_clustering_endpoint,
    vector_observability_canary_probes_endpoint,
    vector_observability_shadow_traffic_comparison_endpoint,
    vector_observability_data_lineage_endpoint,
    vector_observability_reconciliation_checks_endpoint,
    vector_observability_cost_accounting_endpoint,
    vector_observability_feedback_improvement_loop_endpoint,
)

