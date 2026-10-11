// Comprehensive 312 Algorithm and System Architecture Diagram Manifest
const ALL_DIAGRAMS = [
  {
    "file": "advanced_reinforcement_learning__imitation_learning_dagger.html",
    "title": "Imitation Learning Dagger",
    "domain": "Advanced Reinforcement Learning"
  },
  {
    "file": "advanced_reinforcement_learning__mcts_alphazero_muzero.html",
    "title": "Mcts Alphazero Muzero",
    "domain": "Advanced Reinforcement Learning"
  },
  {
    "file": "advanced_reinforcement_learning__offline_rl_cql_iql.html",
    "title": "Offline Rl Cql Iql",
    "domain": "Advanced Reinforcement Learning"
  },
  {
    "file": "advanced_reinforcement_learning__soft_actor_critic.html",
    "title": "Soft Actor Critic",
    "domain": "Advanced Reinforcement Learning"
  },
  {
    "file": "advanced_reinforcement_learning__twin_delayed_ddpg_td3.html",
    "title": "Twin Delayed Ddpg Td3",
    "domain": "Advanced Reinforcement Learning"
  },
  {
    "file": "advanced_reinforcement_learning__world_models_dreamer.html",
    "title": "World Models Dreamer",
    "domain": "Advanced Reinforcement Learning"
  },
  {
    "file": "attention_objectives__causal_language_modeling.html",
    "title": "Causal Language Modeling",
    "domain": "Attention Objectives"
  },
  {
    "file": "attention_objectives__masked_language_modeling.html",
    "title": "Masked Language Modeling",
    "domain": "Attention Objectives"
  },
  {
    "file": "attention_objectives__multi_head_attention.html",
    "title": "Multi Head Attention",
    "domain": "Attention Objectives"
  },
  {
    "file": "attention_objectives__scaled_dot_product_attention.html",
    "title": "Scaled Dot Product Attention",
    "domain": "Attention Objectives"
  },
  {
    "file": "attention_objectives__span_corruption.html",
    "title": "Span Corruption",
    "domain": "Attention Objectives"
  },
  {
    "file": "autodiff__forward_mode_differentiation.html",
    "title": "Forward Mode Differentiation",
    "domain": "Autodiff"
  },
  {
    "file": "autodiff__gradient_checking.html",
    "title": "Gradient Checking",
    "domain": "Autodiff"
  },
  {
    "file": "autodiff__gradient_checkpointing.html",
    "title": "Gradient Checkpointing",
    "domain": "Autodiff"
  },
  {
    "file": "autodiff__gradient_clipping.html",
    "title": "Gradient Clipping",
    "domain": "Autodiff"
  },
  {
    "file": "autodiff__layer_backpropagation.html",
    "title": "Layer Backpropagation",
    "domain": "Autodiff"
  },
  {
    "file": "autodiff__reparameterization_gumbel.html",
    "title": "Reparameterization Gumbel",
    "domain": "Autodiff"
  },
  {
    "file": "autodiff__reverse_mode_autodiff.html",
    "title": "Reverse Mode Autodiff",
    "domain": "Autodiff"
  },
  {
    "file": "autodiff__straight_through_estimator.html",
    "title": "Straight Through Estimator",
    "domain": "Autodiff"
  },
  {
    "file": "building_blocks__dense_highway_connection.html",
    "title": "Dense Highway Connection",
    "domain": "Building Blocks"
  },
  {
    "file": "building_blocks__embedding_lookup.html",
    "title": "Embedding Lookup",
    "domain": "Building Blocks"
  },
  {
    "file": "building_blocks__gated_linear_units.html",
    "title": "Gated Linear Units",
    "domain": "Building Blocks"
  },
  {
    "file": "building_blocks__linear_layer.html",
    "title": "Linear Layer",
    "domain": "Building Blocks"
  },
  {
    "file": "building_blocks__mlp_feedforward.html",
    "title": "Mlp Feedforward",
    "domain": "Building Blocks"
  },
  {
    "file": "building_blocks__perceptron_learning.html",
    "title": "Perceptron Learning",
    "domain": "Building Blocks"
  },
  {
    "file": "building_blocks__relu_family.html",
    "title": "Relu Family",
    "domain": "Building Blocks"
  },
  {
    "file": "building_blocks__residual_connection.html",
    "title": "Residual Connection",
    "domain": "Building Blocks"
  },
  {
    "file": "building_blocks__sigmoid_tanh.html",
    "title": "Sigmoid Tanh",
    "domain": "Building Blocks"
  },
  {
    "file": "building_blocks__smooth_activations.html",
    "title": "Smooth Activations",
    "domain": "Building Blocks"
  },
  {
    "file": "building_blocks__softmax_stable.html",
    "title": "Softmax Stable",
    "domain": "Building Blocks"
  },
  {
    "file": "building_blocks__weight_tying.html",
    "title": "Weight Tying",
    "domain": "Building Blocks"
  },
  {
    "file": "control-plane.html",
    "title": "Control Plane",
    "domain": "System Architecture"
  },
  {
    "file": "convolutional__compound_model_scaling.html",
    "title": "Compound Model Scaling",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__convnext_block.html",
    "title": "Convnext Block",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__convolution_2d_mechanics.html",
    "title": "Convolution 2D Mechanics",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__deformable_convolution.html",
    "title": "Deformable Convolution",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__depthwise_separable_convolution.html",
    "title": "Depthwise Separable Convolution",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__dilated_atrous_convolution.html",
    "title": "Dilated Atrous Convolution",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__fast_convolution_im2col_winograd.html",
    "title": "Fast Convolution Im2Col Winograd",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__feature_pyramid_networks.html",
    "title": "Feature Pyramid Networks",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__inception_multibranch_blocks.html",
    "title": "Inception Multibranch Blocks",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__inverted_residuals_linear_bottleneck.html",
    "title": "Inverted Residuals Linear Bottleneck",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__one_by_one_convolution_bottleneck.html",
    "title": "One By One Convolution Bottleneck",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__pooling_layers.html",
    "title": "Pooling Layers",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__receptive_field_analysis.html",
    "title": "Receptive Field Analysis",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__resnet_residual_blocks.html",
    "title": "Resnet Residual Blocks",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__squeeze_excitation_cbam.html",
    "title": "Squeeze Excitation Cbam",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__transposed_convolution_pixel_shuffle.html",
    "title": "Transposed Convolution Pixel Shuffle",
    "domain": "Convolutional"
  },
  {
    "file": "convolutional__unet_encoder_decoder.html",
    "title": "Unet Encoder Decoder",
    "domain": "Convolutional"
  },
  {
    "file": "data-plane.html",
    "title": "Data Plane",
    "domain": "System Architecture"
  },
  {
    "file": "decoding__constrained_grammar_decoding.html",
    "title": "Constrained Grammar Decoding",
    "domain": "Decoding"
  },
  {
    "file": "decoding__contrastive_search_decoding.html",
    "title": "Contrastive Search Decoding",
    "domain": "Decoding"
  },
  {
    "file": "decoding__greedy_temperature_sampling.html",
    "title": "Greedy Temperature Sampling",
    "domain": "Decoding"
  },
  {
    "file": "decoding__lookahead_jacobi_decoding.html",
    "title": "Lookahead Jacobi Decoding",
    "domain": "Decoding"
  },
  {
    "file": "decoding__nucleus_min_p_sampling.html",
    "title": "Nucleus Min P Sampling",
    "domain": "Decoding"
  },
  {
    "file": "decoding__repetition_frequency_penalties.html",
    "title": "Repetition Frequency Penalties",
    "domain": "Decoding"
  },
  {
    "file": "decoding__self_consistency_sampling.html",
    "title": "Self Consistency Sampling",
    "domain": "Decoding"
  },
  {
    "file": "decoding__self_drafting_medusa_eagle.html",
    "title": "Self Drafting Medusa Eagle",
    "domain": "Decoding"
  },
  {
    "file": "decoding__speculative_decoding.html",
    "title": "Speculative Decoding",
    "domain": "Decoding"
  },
  {
    "file": "decoding__top_k_sampling.html",
    "title": "Top K Sampling",
    "domain": "Decoding"
  },
  {
    "file": "detection__anchor_free_detector_fcos.html",
    "title": "Anchor Free Detector Fcos",
    "domain": "Detection"
  },
  {
    "file": "detection__detr_bipartite_matching.html",
    "title": "Detr Bipartite Matching",
    "domain": "Detection"
  },
  {
    "file": "detection__non_maximum_suppression.html",
    "title": "Non Maximum Suppression",
    "domain": "Detection"
  },
  {
    "file": "detection__roi_align_mask_rcnn.html",
    "title": "Roi Align Mask Rcnn",
    "domain": "Detection"
  },
  {
    "file": "detection__rpn_faster_rcnn.html",
    "title": "Rpn Faster Rcnn",
    "domain": "Detection"
  },
  {
    "file": "detection__single_stage_detector_yolo.html",
    "title": "Single Stage Detector Yolo",
    "domain": "Detection"
  },
  {
    "file": "diffusion_flow_models__classifier_free_guidance.html",
    "title": "Classifier Free Guidance",
    "domain": "Diffusion Flow Models"
  },
  {
    "file": "diffusion_flow_models__consistency_models_distillation.html",
    "title": "Consistency Models Distillation",
    "domain": "Diffusion Flow Models"
  },
  {
    "file": "diffusion_flow_models__controlnet_conditioning_adapters.html",
    "title": "Controlnet Conditioning Adapters",
    "domain": "Diffusion Flow Models"
  },
  {
    "file": "diffusion_flow_models__ddim_deterministic_sampling.html",
    "title": "Ddim Deterministic Sampling",
    "domain": "Diffusion Flow Models"
  },
  {
    "file": "diffusion_flow_models__ddpm_diffusion_process.html",
    "title": "Ddpm Diffusion Process",
    "domain": "Diffusion Flow Models"
  },
  {
    "file": "diffusion_flow_models__diffusion_transformers_dit.html",
    "title": "Diffusion Transformers Dit",
    "domain": "Diffusion Flow Models"
  },
  {
    "file": "diffusion_flow_models__fast_diffusion_samplers.html",
    "title": "Fast Diffusion Samplers",
    "domain": "Diffusion Flow Models"
  },
  {
    "file": "diffusion_flow_models__flow_matching_rectified_flow.html",
    "title": "Flow Matching Rectified Flow",
    "domain": "Diffusion Flow Models"
  },
  {
    "file": "diffusion_flow_models__latent_diffusion_models.html",
    "title": "Latent Diffusion Models",
    "domain": "Diffusion Flow Models"
  },
  {
    "file": "diffusion_flow_models__noise_schedules_prediction_targets.html",
    "title": "Noise Schedules Prediction Targets",
    "domain": "Diffusion Flow Models"
  },
  {
    "file": "diffusion_flow_models__score_based_sde.html",
    "title": "Score Based Sde",
    "domain": "Diffusion Flow Models"
  },
  {
    "file": "distillation_continual_learning__elastic_weight_consolidation_ewc.html",
    "title": "Elastic Weight Consolidation Ewc",
    "domain": "Distillation Continual Learning"
  },
  {
    "file": "distillation_continual_learning__experience_replay_continual_learning.html",
    "title": "Experience Replay Continual Learning",
    "domain": "Distillation Continual Learning"
  },
  {
    "file": "distillation_continual_learning__knowledge_distillation_soft_targets.html",
    "title": "Knowledge Distillation Soft Targets",
    "domain": "Distillation Continual Learning"
  },
  {
    "file": "distillation_continual_learning__sequence_on_policy_distillation.html",
    "title": "Sequence On Policy Distillation",
    "domain": "Distillation Continual Learning"
  },
  {
    "file": "e2e.html",
    "title": "E2E Data Flow",
    "domain": "System Architecture"
  },
  {
    "file": "editing_unlearning_security__automated_red_teaming_evaluation.html",
    "title": "Automated red-teaming and adversarial evaluation",
    "domain": "Model Editing, Unlearning &amp; Security"
  },
  {
    "file": "editing_unlearning_security__backdoor_data_poisoning_detection.html",
    "title": "Backdoor and data-poisoning detection",
    "domain": "Model Editing, Unlearning &amp; Security"
  },
  {
    "file": "editing_unlearning_security__guard_models.html",
    "title": "Guard models",
    "domain": "Model Editing, Unlearning &amp; Security"
  },
  {
    "file": "editing_unlearning_security__machine_unlearning.html",
    "title": "Machine unlearning",
    "domain": "Model Editing, Unlearning &amp; Security"
  },
  {
    "file": "editing_unlearning_security__membership_inference_testing.html",
    "title": "Membership inference testing",
    "domain": "Model Editing, Unlearning &amp; Security"
  },
  {
    "file": "editing_unlearning_security__model_editing.html",
    "title": "Model editing",
    "domain": "Model Editing, Unlearning &amp; Security"
  },
  {
    "file": "editing_unlearning_security__watermarking_generated_text.html",
    "title": "Watermarking generated text",
    "domain": "Model Editing, Unlearning &amp; Security"
  },
  {
    "file": "er-schema.html",
    "title": "ER Schema",
    "domain": "System Architecture"
  },
  {
    "file": "evaluation_tuning_diagnostics__benchmark_contamination_detection.html",
    "title": "Benchmark contamination detection",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "evaluation_tuning_diagnostics__class_imbalance_handling.html",
    "title": "Class imbalance handling",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "evaluation_tuning_diagnostics__hyperband_and_asha.html",
    "title": "Hyperband and ASHA",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "evaluation_tuning_diagnostics__learning_curves_and_data_scaling_estimation.html",
    "title": "Learning curves and data scaling estimation",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "evaluation_tuning_diagnostics__learning_with_noisy_labels.html",
    "title": "Learning with noisy labels",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "evaluation_tuning_diagnostics__once_for_all_elastic_networks.html",
    "title": "Once-for-all and elastic networks",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "evaluation_tuning_diagnostics__perplexity_and_bits_per_byte.html",
    "title": "Perplexity and bits per byte",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "evaluation_tuning_diagnostics__population_based_training.html",
    "title": "Population-based training",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "evaluation_tuning_diagnostics__reproducibility_determinism_training.html",
    "title": "Reproducibility and determinism in training",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "evaluation_tuning_diagnostics__sanity_checks.html",
    "title": "Sanity checks",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "evaluation_tuning_diagnostics__stochastic_weight_averaging.html",
    "title": "Stochastic weight averaging",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "evaluation_tuning_diagnostics__test_time_augmentation_adaptation.html",
    "title": "Test-time augmentation and test-time adaptation",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "evaluation_tuning_diagnostics__training_dynamics_monitoring.html",
    "title": "Training dynamics monitoring",
    "domain": "Evaluation, Tuning &amp; Diagnostics"
  },
  {
    "file": "generative_autoencoders_gans__autoregressive_pixelcnn.html",
    "title": "Autoregressive Pixelcnn",
    "domain": "Generative Autoencoders Gans"
  },
  {
    "file": "generative_autoencoders_gans__conditional_image_to_image_gan.html",
    "title": "Conditional Image To Image Gan",
    "domain": "Generative Autoencoders Gans"
  },
  {
    "file": "generative_autoencoders_gans__denoising_autoencoder.html",
    "title": "Denoising Autoencoder",
    "domain": "Generative Autoencoders Gans"
  },
  {
    "file": "generative_autoencoders_gans__energy_based_models.html",
    "title": "Energy Based Models",
    "domain": "Generative Autoencoders Gans"
  },
  {
    "file": "generative_autoencoders_gans__generative_adversarial_network.html",
    "title": "Generative Adversarial Network",
    "domain": "Generative Autoencoders Gans"
  },
  {
    "file": "generative_autoencoders_gans__normalizing_flows.html",
    "title": "Normalizing Flows",
    "domain": "Generative Autoencoders Gans"
  },
  {
    "file": "generative_autoencoders_gans__stylegan_modulation.html",
    "title": "Stylegan Modulation",
    "domain": "Generative Autoencoders Gans"
  },
  {
    "file": "generative_autoencoders_gans__variational_autoencoder.html",
    "title": "Variational Autoencoder",
    "domain": "Generative Autoencoders Gans"
  },
  {
    "file": "generative_autoencoders_gans__vector_quantized_vae.html",
    "title": "Vector Quantized Vae",
    "domain": "Generative Autoencoders Gans"
  },
  {
    "file": "generative_autoencoders_gans__wasserstein_gan_gp.html",
    "title": "Wasserstein Gan Gp",
    "domain": "Generative Autoencoders Gans"
  },
  {
    "file": "inference_serving__chunked_prefill_decode_disaggregation.html",
    "title": "Chunked prefill and prefill/decode disaggregation",
    "domain": "Inference Optimization &amp; Serving"
  },
  {
    "file": "inference_serving__continuous_batching.html",
    "title": "Continuous batching",
    "domain": "Inference Optimization &amp; Serving"
  },
  {
    "file": "inference_serving__early_exit_and_adaptive_computation.html",
    "title": "Early exit and adaptive computation",
    "domain": "Inference Optimization &amp; Serving"
  },
  {
    "file": "inference_serving__graph_level_model_optimization.html",
    "title": "Graph-level model optimization",
    "domain": "Inference Optimization &amp; Serving"
  },
  {
    "file": "inference_serving__kernel_fusion_and_compilation.html",
    "title": "Kernel fusion and compilation",
    "domain": "Inference Optimization &amp; Serving"
  },
  {
    "file": "inference_serving__low_rank_weight_compression.html",
    "title": "Low-rank weight compression",
    "domain": "Inference Optimization &amp; Serving"
  },
  {
    "file": "inference_serving__model_cascades_and_routing.html",
    "title": "Model cascades and routing",
    "domain": "Inference Optimization &amp; Serving"
  },
  {
    "file": "inference_serving__prefix_caching.html",
    "title": "Prefix caching",
    "domain": "Inference Optimization &amp; Serving"
  },
  {
    "file": "interpretability_explainability__activation_patching.html",
    "title": "Activation patching",
    "domain": "Interpretability &amp; Explanation"
  },
  {
    "file": "interpretability_explainability__activation_steering.html",
    "title": "Activation steering",
    "domain": "Interpretability &amp; Explanation"
  },
  {
    "file": "interpretability_explainability__attention_attribution_rollout.html",
    "title": "Attention attribution and attention rollout",
    "domain": "Interpretability &amp; Explanation"
  },
  {
    "file": "interpretability_explainability__circuit_discovery.html",
    "title": "Circuit discovery",
    "domain": "Interpretability &amp; Explanation"
  },
  {
    "file": "interpretability_explainability__grad_cam.html",
    "title": "Grad-CAM",
    "domain": "Interpretability &amp; Explanation"
  },
  {
    "file": "interpretability_explainability__integrated_gradients_saliency.html",
    "title": "Integrated gradients and gradient saliency",
    "domain": "Interpretability &amp; Explanation"
  },
  {
    "file": "interpretability_explainability__logit_lens_and_tuned_lens.html",
    "title": "Logit lens and tuned lens",
    "domain": "Interpretability &amp; Explanation"
  },
  {
    "file": "interpretability_explainability__probing_classifiers.html",
    "title": "Probing classifiers",
    "domain": "Interpretability &amp; Explanation"
  },
  {
    "file": "interpretability_explainability__shap.html",
    "title": "SHAP",
    "domain": "Interpretability &amp; Explanation"
  },
  {
    "file": "interpretability_explainability__sparse_autoencoders_interpretable_features.html",
    "title": "Sparse autoencoders for interpretable features",
    "domain": "Interpretability &amp; Explanation"
  },
  {
    "file": "llm_alignment_post_training__curriculum_learning.html",
    "title": "Curriculum Learning",
    "domain": "Llm Alignment Post Training"
  },
  {
    "file": "llm_alignment_post_training__direct_preference_optimization_dpo.html",
    "title": "Direct Preference Optimization Dpo",
    "domain": "Llm Alignment Post Training"
  },
  {
    "file": "llm_alignment_post_training__dpo_variants_simpo_kto.html",
    "title": "Dpo Variants Simpo Kto",
    "domain": "Llm Alignment Post Training"
  },
  {
    "file": "llm_alignment_post_training__group_relative_policy_optimization_grpo.html",
    "title": "Group Relative Policy Optimization Grpo",
    "domain": "Llm Alignment Post Training"
  },
  {
    "file": "llm_alignment_post_training__process_reward_models_prm.html",
    "title": "Process Reward Models Prm",
    "domain": "Llm Alignment Post Training"
  },
  {
    "file": "llm_alignment_post_training__rejection_sampling_best_of_n.html",
    "title": "Rejection Sampling Best Of N",
    "domain": "Llm Alignment Post Training"
  },
  {
    "file": "llm_alignment_post_training__reward_modeling_bradley_terry.html",
    "title": "Reward Modeling Bradley Terry",
    "domain": "Llm Alignment Post Training"
  },
  {
    "file": "llm_alignment_post_training__rl_verifiable_rewards_rlvr.html",
    "title": "Rl Verifiable Rewards Rlvr",
    "domain": "Llm Alignment Post Training"
  },
  {
    "file": "llm_alignment_post_training__rlaif_constitution_critique.html",
    "title": "Rlaif Constitution Critique",
    "domain": "Llm Alignment Post Training"
  },
  {
    "file": "llm_alignment_post_training__rlhf_ppo_kl_penalty.html",
    "title": "Rlhf Ppo Kl Penalty",
    "domain": "Llm Alignment Post Training"
  },
  {
    "file": "llm_alignment_post_training__supervised_fine_tuning.html",
    "title": "Supervised Fine Tuning",
    "domain": "Llm Alignment Post Training"
  },
  {
    "file": "losses__binary_cross_entropy_logits.html",
    "title": "Binary Cross Entropy Logits",
    "domain": "Losses"
  },
  {
    "file": "losses__cross_entropy_nll.html",
    "title": "Cross Entropy Nll",
    "domain": "Losses"
  },
  {
    "file": "losses__ctc_loss.html",
    "title": "Ctc Loss",
    "domain": "Losses"
  },
  {
    "file": "losses__focal_loss.html",
    "title": "Focal Loss",
    "domain": "Losses"
  },
  {
    "file": "losses__kl_divergence_loss.html",
    "title": "Kl Divergence Loss",
    "domain": "Losses"
  },
  {
    "file": "losses__label_smoothing.html",
    "title": "Label Smoothing",
    "domain": "Losses"
  },
  {
    "file": "losses__metric_learning_losses.html",
    "title": "Metric Learning Losses",
    "domain": "Losses"
  },
  {
    "file": "losses__multitask_loss_balancing.html",
    "title": "Multitask Loss Balancing",
    "domain": "Losses"
  },
  {
    "file": "losses__overlap_segmentation_losses.html",
    "title": "Overlap Segmentation Losses",
    "domain": "Losses"
  },
  {
    "file": "losses__regression_losses.html",
    "title": "Regression Losses",
    "domain": "Losses"
  },
  {
    "file": "master.html",
    "title": "Master Blueprint",
    "domain": "System Architecture"
  },
  {
    "file": "mixture_of_experts__moe_expert_choice_routing.html",
    "title": "Moe Expert Choice Routing",
    "domain": "Mixture Of Experts"
  },
  {
    "file": "mixture_of_experts__moe_load_balancing.html",
    "title": "Moe Load Balancing",
    "domain": "Mixture Of Experts"
  },
  {
    "file": "mixture_of_experts__moe_top_k_gating.html",
    "title": "Moe Top K Gating",
    "domain": "Mixture Of Experts"
  },
  {
    "file": "multimodal_memory__clip_contrastive_pretraining.html",
    "title": "Clip Contrastive Pretraining",
    "domain": "Multimodal Memory"
  },
  {
    "file": "multimodal_memory__cross_attention_conditioning.html",
    "title": "Cross Attention Conditioning",
    "domain": "Multimodal Memory"
  },
  {
    "file": "multimodal_memory__multi_token_prediction.html",
    "title": "Multi Token Prediction",
    "domain": "Multimodal Memory"
  },
  {
    "file": "multimodal_memory__multimodal_llm_connectors.html",
    "title": "Multimodal Llm Connectors",
    "domain": "Multimodal Memory"
  },
  {
    "file": "multimodal_memory__perceiver_latent_attention.html",
    "title": "Perceiver Latent Attention",
    "domain": "Multimodal Memory"
  },
  {
    "file": "multimodal_memory__prefix_lm_mixture_denoisers.html",
    "title": "Prefix Lm Mixture Denoisers",
    "domain": "Multimodal Memory"
  },
  {
    "file": "multimodal_memory__retrieval_enhanced_transformers.html",
    "title": "Retrieval Enhanced Transformers",
    "domain": "Multimodal Memory"
  },
  {
    "file": "multimodal_memory__segment_level_recurrence.html",
    "title": "Segment Level Recurrence",
    "domain": "Multimodal Memory"
  },
  {
    "file": "multimodal_memory__swin_transformer.html",
    "title": "Swin Transformer",
    "domain": "Multimodal Memory"
  },
  {
    "file": "multimodal_memory__vision_transformer.html",
    "title": "Vision Transformer",
    "domain": "Multimodal Memory"
  },
  {
    "file": "normalization__batch_normalization.html",
    "title": "Batch Normalization",
    "domain": "Normalization"
  },
  {
    "file": "normalization__group_instance_norm.html",
    "title": "Group Instance Norm",
    "domain": "Normalization"
  },
  {
    "file": "normalization__layer_normalization.html",
    "title": "Layer Normalization",
    "domain": "Normalization"
  },
  {
    "file": "normalization__normalization_placement.html",
    "title": "Normalization Placement",
    "domain": "Normalization"
  },
  {
    "file": "normalization__qk_norm_logit_soft_capping.html",
    "title": "Qk Norm Logit Soft Capping",
    "domain": "Normalization"
  },
  {
    "file": "normalization__rms_norm.html",
    "title": "Rms Norm",
    "domain": "Normalization"
  },
  {
    "file": "normalization__weight_spectral_norm.html",
    "title": "Weight Spectral Norm",
    "domain": "Normalization"
  },
  {
    "file": "optimizers__adafactor_optimizer.html",
    "title": "Adafactor Optimizer",
    "domain": "Optimizers"
  },
  {
    "file": "optimizers__adagrad_optimizer.html",
    "title": "Adagrad Optimizer",
    "domain": "Optimizers"
  },
  {
    "file": "optimizers__adam_optimizer.html",
    "title": "Adam Optimizer",
    "domain": "Optimizers"
  },
  {
    "file": "optimizers__adamw_optimizer.html",
    "title": "Adamw Optimizer",
    "domain": "Optimizers"
  },
  {
    "file": "optimizers__lars_lamb_optimizer.html",
    "title": "Lars Lamb Optimizer",
    "domain": "Optimizers"
  },
  {
    "file": "optimizers__lion_optimizer.html",
    "title": "Lion Optimizer",
    "domain": "Optimizers"
  },
  {
    "file": "optimizers__lookahead_weight_averaging.html",
    "title": "Lookahead Weight Averaging",
    "domain": "Optimizers"
  },
  {
    "file": "optimizers__momentum_nesterov.html",
    "title": "Momentum Nesterov",
    "domain": "Optimizers"
  },
  {
    "file": "optimizers__rmsprop_optimizer.html",
    "title": "Rmsprop Optimizer",
    "domain": "Optimizers"
  },
  {
    "file": "optimizers__second_order_preconditioning.html",
    "title": "Second Order Preconditioning",
    "domain": "Optimizers"
  },
  {
    "file": "optimizers__sharpness_aware_minimization.html",
    "title": "Sharpness Aware Minimization",
    "domain": "Optimizers"
  },
  {
    "file": "optimizers__stochastic_gradient_descent.html",
    "title": "Stochastic Gradient Descent",
    "domain": "Optimizers"
  },
  {
    "file": "peft_model_merging__bottleneck_adapter_modules.html",
    "title": "Bottleneck Adapter Modules",
    "domain": "Peft Model Merging"
  },
  {
    "file": "peft_model_merging__dora_weight_decomposed_lora.html",
    "title": "Dora Weight Decomposed Lora",
    "domain": "Peft Model Merging"
  },
  {
    "file": "peft_model_merging__ia3_bitfit_minimal_tuning.html",
    "title": "Ia3 Bitfit Minimal Tuning",
    "domain": "Peft Model Merging"
  },
  {
    "file": "peft_model_merging__lora_low_rank_adaptation.html",
    "title": "Lora Low Rank Adaptation",
    "domain": "Peft Model Merging"
  },
  {
    "file": "peft_model_merging__model_merging_ties_dare.html",
    "title": "Model Merging Ties Dare",
    "domain": "Peft Model Merging"
  },
  {
    "file": "peft_model_merging__prefix_prompt_tuning.html",
    "title": "Prefix Prompt Tuning",
    "domain": "Peft Model Merging"
  },
  {
    "file": "peft_model_merging__qlora_nf4_quantization.html",
    "title": "Qlora Nf4 Quantization",
    "domain": "Peft Model Merging"
  },
  {
    "file": "phase-1.html",
    "title": "Phase 1 \u00b7 Curation",
    "domain": "System Architecture"
  },
  {
    "file": "phase-2.html",
    "title": "Phase 2 \u00b7 Micro-Arch",
    "domain": "System Architecture"
  },
  {
    "file": "phase-3.html",
    "title": "Phase 3 \u00b7 5D Scaling",
    "domain": "System Architecture"
  },
  {
    "file": "phase-4.html",
    "title": "Phase 4 \u00b7 Loss & Opt",
    "domain": "System Architecture"
  },
  {
    "file": "phase-5.html",
    "title": "Phase 5 \u00b7 Alignment",
    "domain": "System Architecture"
  },
  {
    "file": "phase-6.html",
    "title": "Phase 6 \u00b7 PEFT",
    "domain": "System Architecture"
  },
  {
    "file": "phase-7.html",
    "title": "Phase 7 \u00b7 Serving",
    "domain": "System Architecture"
  },
  {
    "file": "position_encodings__alibi_attention_linear_biases.html",
    "title": "Alibi Attention Linear Biases",
    "domain": "Position Encodings"
  },
  {
    "file": "position_encodings__context_window_extension.html",
    "title": "Context Window Extension",
    "domain": "Position Encodings"
  },
  {
    "file": "position_encodings__learned_absolute_position_embeddings.html",
    "title": "Learned Absolute Position Embeddings",
    "domain": "Position Encodings"
  },
  {
    "file": "position_encodings__relative_position_bias.html",
    "title": "Relative Position Bias",
    "domain": "Position Encodings"
  },
  {
    "file": "position_encodings__rotary_position_embeddings.html",
    "title": "Rotary Position Embeddings",
    "domain": "Position Encodings"
  },
  {
    "file": "position_encodings__sinusoidal_positional_encoding.html",
    "title": "Sinusoidal Positional Encoding",
    "domain": "Position Encodings"
  },
  {
    "file": "pruning_sparsity__magnitude_pruning_lottery_ticket.html",
    "title": "Magnitude Pruning Lottery Ticket",
    "domain": "Pruning Sparsity"
  },
  {
    "file": "pruning_sparsity__semistructured_2to4_sparsity.html",
    "title": "Semistructured 2To4 Sparsity",
    "domain": "Pruning Sparsity"
  },
  {
    "file": "pruning_sparsity__structured_oneshot_pruning_llms.html",
    "title": "Structured Oneshot Pruning Llms",
    "domain": "Pruning Sparsity"
  },
  {
    "file": "quantization__awq.html",
    "title": "AWQ",
    "domain": "Quantization"
  },
  {
    "file": "quantization__gptq.html",
    "title": "GPTQ",
    "domain": "Quantization"
  },
  {
    "file": "quantization__kv_cache_quantization.html",
    "title": "KV-cache quantization",
    "domain": "Quantization"
  },
  {
    "file": "quantization__low_bit_number_formats.html",
    "title": "Low-bit number formats",
    "domain": "Quantization"
  },
  {
    "file": "quantization__post_training_quantization.html",
    "title": "Post-training quantization",
    "domain": "Quantization"
  },
  {
    "file": "quantization__quantization_aware_training.html",
    "title": "Quantization-aware training",
    "domain": "Quantization"
  },
  {
    "file": "quantization__smoothquant_activation_outliers.html",
    "title": "SmoothQuant",
    "domain": "Quantization"
  },
  {
    "file": "recurrent_sequence__attention_bahdanau_luong.html",
    "title": "Attention Bahdanau Luong",
    "domain": "Recurrent Sequence"
  },
  {
    "file": "recurrent_sequence__beam_search_length_normalization.html",
    "title": "Beam Search Length Normalization",
    "domain": "Recurrent Sequence"
  },
  {
    "file": "recurrent_sequence__bidirectional_rnn.html",
    "title": "Bidirectional Rnn",
    "domain": "Recurrent Sequence"
  },
  {
    "file": "recurrent_sequence__gru_cell.html",
    "title": "Gru Cell",
    "domain": "Recurrent Sequence"
  },
  {
    "file": "recurrent_sequence__lstm_cell.html",
    "title": "Lstm Cell",
    "domain": "Recurrent Sequence"
  },
  {
    "file": "recurrent_sequence__pointer_networks_copy_mechanism.html",
    "title": "Pointer Networks Copy Mechanism",
    "domain": "Recurrent Sequence"
  },
  {
    "file": "recurrent_sequence__rnn_bptt.html",
    "title": "Rnn Bptt",
    "domain": "Recurrent Sequence"
  },
  {
    "file": "recurrent_sequence__seq2seq_encoder_decoder.html",
    "title": "Seq2Seq Encoder Decoder",
    "domain": "Recurrent Sequence"
  },
  {
    "file": "recurrent_sequence__teacher_forcing_scheduled_sampling.html",
    "title": "Teacher Forcing Scheduled Sampling",
    "domain": "Recurrent Sequence"
  },
  {
    "file": "recurrent_sequence__temporal_convolutional_network.html",
    "title": "Temporal Convolutional Network",
    "domain": "Recurrent Sequence"
  },
  {
    "file": "recurrent_sequence__truncated_bptt.html",
    "title": "Truncated Bptt",
    "domain": "Recurrent Sequence"
  },
  {
    "file": "regularization_augmentation__adversarial_training_fgsm_pgd.html",
    "title": "Adversarial Training Fgsm Pgd",
    "domain": "Regularization Augmentation"
  },
  {
    "file": "regularization_augmentation__consistency_pseudo_labeling.html",
    "title": "Consistency Pseudo Labeling",
    "domain": "Regularization Augmentation"
  },
  {
    "file": "regularization_augmentation__dropout_inverted.html",
    "title": "Dropout Inverted",
    "domain": "Regularization Augmentation"
  },
  {
    "file": "regularization_augmentation__early_stopping_checkpointing.html",
    "title": "Early Stopping Checkpointing",
    "domain": "Regularization Augmentation"
  },
  {
    "file": "regularization_augmentation__image_data_augmentation.html",
    "title": "Image Data Augmentation",
    "domain": "Regularization Augmentation"
  },
  {
    "file": "regularization_augmentation__mixup_cutmix.html",
    "title": "Mixup Cutmix",
    "domain": "Regularization Augmentation"
  },
  {
    "file": "regularization_augmentation__spec_augment.html",
    "title": "Spec Augment",
    "domain": "Regularization Augmentation"
  },
  {
    "file": "regularization_augmentation__stochastic_depth.html",
    "title": "Stochastic Depth",
    "domain": "Regularization Augmentation"
  },
  {
    "file": "regularization_augmentation__text_augmentation.html",
    "title": "Text Augmentation",
    "domain": "Regularization Augmentation"
  },
  {
    "file": "reinforcement_learning__actor_critic_gae.html",
    "title": "Actor Critic Gae",
    "domain": "Reinforcement Learning"
  },
  {
    "file": "reinforcement_learning__deep_q_networks_dqn.html",
    "title": "Deep Q Networks Dqn",
    "domain": "Reinforcement Learning"
  },
  {
    "file": "reinforcement_learning__policy_gradients_reinforce.html",
    "title": "Policy Gradients Reinforce",
    "domain": "Reinforcement Learning"
  },
  {
    "file": "reinforcement_learning__proximal_policy_optimization_ppo.html",
    "title": "Proximal Policy Optimization Ppo",
    "domain": "Reinforcement Learning"
  },
  {
    "file": "reinforcement_learning__rainbow_dqn_extensions.html",
    "title": "Rainbow Dqn Extensions",
    "domain": "Reinforcement Learning"
  },
  {
    "file": "robustness_uncertainty__certified_robustness.html",
    "title": "Certified robustness",
    "domain": "Robustness &amp; Uncertainty"
  },
  {
    "file": "robustness_uncertainty__conformal_prediction.html",
    "title": "Conformal prediction",
    "domain": "Robustness &amp; Uncertainty"
  },
  {
    "file": "robustness_uncertainty__deep_ensembles.html",
    "title": "Deep ensembles",
    "domain": "Robustness &amp; Uncertainty"
  },
  {
    "file": "robustness_uncertainty__monte_carlo_dropout_for_uncertainty.html",
    "title": "Monte Carlo dropout for uncertainty",
    "domain": "Robustness &amp; Uncertainty"
  },
  {
    "file": "robustness_uncertainty__out_of_distribution_detection.html",
    "title": "Out-of-distribution detection",
    "domain": "Robustness &amp; Uncertainty"
  },
  {
    "file": "scaling_stability__attention_sinks_streaming.html",
    "title": "Attention Sinks Streaming",
    "domain": "Scaling Stability"
  },
  {
    "file": "scaling_stability__logit_stabilization_z_loss.html",
    "title": "Logit Stabilization Z Loss",
    "domain": "Scaling Stability"
  },
  {
    "file": "scaling_stability__scaling_laws_compute_optimal.html",
    "title": "Scaling Laws Compute Optimal",
    "domain": "Scaling Stability"
  },
  {
    "file": "schedules_init__batch_size_lr_scaling.html",
    "title": "Batch Size Lr Scaling",
    "domain": "Schedules Init"
  },
  {
    "file": "schedules_init__cosine_decay_restarts.html",
    "title": "Cosine Decay Restarts",
    "domain": "Schedules Init"
  },
  {
    "file": "schedules_init__deep_residual_init.html",
    "title": "Deep Residual Init",
    "domain": "Schedules Init"
  },
  {
    "file": "schedules_init__he_kaiming_init.html",
    "title": "He Kaiming Init",
    "domain": "Schedules Init"
  },
  {
    "file": "schedules_init__learning_rate_warmup.html",
    "title": "Learning Rate Warmup",
    "domain": "Schedules Init"
  },
  {
    "file": "schedules_init__maximal_update_param.html",
    "title": "Maximal Update Param",
    "domain": "Schedules Init"
  },
  {
    "file": "schedules_init__wsd_one_cycle_schedules.html",
    "title": "Wsd One Cycle Schedules",
    "domain": "Schedules Init"
  },
  {
    "file": "schedules_init__xavier_glorot_init.html",
    "title": "Xavier Glorot Init",
    "domain": "Schedules Init"
  },
  {
    "file": "self_supervised_learning__joint_embedding_predictive_jepa.html",
    "title": "Joint Embedding Predictive Jepa",
    "domain": "Self Supervised Learning"
  },
  {
    "file": "self_supervised_learning__masked_autoencoders_vision.html",
    "title": "Masked Autoencoders Vision",
    "domain": "Self Supervised Learning"
  },
  {
    "file": "self_supervised_learning__non_contrastive_self_supervision.html",
    "title": "Non Contrastive Self Supervision",
    "domain": "Self Supervised Learning"
  },
  {
    "file": "self_supervised_learning__redundancy_reduction_ssl.html",
    "title": "Redundancy Reduction Ssl",
    "domain": "Self Supervised Learning"
  },
  {
    "file": "self_supervised_learning__self_distillation_dino.html",
    "title": "Self Distillation Dino",
    "domain": "Self Supervised Learning"
  },
  {
    "file": "specialized_scientific_meta__differentiable_architecture_search_darts.html",
    "title": "Differentiable Architecture Search Darts",
    "domain": "Specialized Scientific Meta"
  },
  {
    "file": "specialized_scientific_meta__differentiable_neural_memory.html",
    "title": "Differentiable Neural Memory",
    "domain": "Specialized Scientific Meta"
  },
  {
    "file": "specialized_scientific_meta__fourier_neural_operator.html",
    "title": "Fourier Neural Operator",
    "domain": "Specialized Scientific Meta"
  },
  {
    "file": "specialized_scientific_meta__gaussian_splatting_3d.html",
    "title": "Gaussian Splatting 3D",
    "domain": "Specialized Scientific Meta"
  },
  {
    "file": "specialized_scientific_meta__hypernetworks_conditioning.html",
    "title": "Hypernetworks Conditioning",
    "domain": "Specialized Scientific Meta"
  },
  {
    "file": "specialized_scientific_meta__maml_meta_learning.html",
    "title": "Maml Meta Learning",
    "domain": "Specialized Scientific Meta"
  },
  {
    "file": "specialized_scientific_meta__mixture_density_networks.html",
    "title": "Mixture Density Networks",
    "domain": "Specialized Scientific Meta"
  },
  {
    "file": "specialized_scientific_meta__neural_ordinary_differential_equations.html",
    "title": "Neural Ordinary Differential Equations",
    "domain": "Specialized Scientific Meta"
  },
  {
    "file": "specialized_scientific_meta__neural_radiance_fields_nerf.html",
    "title": "Neural Radiance Fields Nerf",
    "domain": "Specialized Scientific Meta"
  },
  {
    "file": "specialized_scientific_meta__physics_informed_neural_networks.html",
    "title": "Physics Informed Neural Networks",
    "domain": "Specialized Scientific Meta"
  },
  {
    "file": "specialized_scientific_meta__prototypical_networks.html",
    "title": "Prototypical Networks",
    "domain": "Specialized Scientific Meta"
  },
  {
    "file": "specialized_scientific_meta__spiking_neural_networks_lif.html",
    "title": "Spiking Neural Networks Lif",
    "domain": "Specialized Scientific Meta"
  },
  {
    "file": "timeseries_recommendation__basis_expansion_nbeats.html",
    "title": "Basis Expansion Nbeats",
    "domain": "Timeseries Recommendation"
  },
  {
    "file": "timeseries_recommendation__deep_cross_network_v2.html",
    "title": "Deep Cross Network V2",
    "domain": "Timeseries Recommendation"
  },
  {
    "file": "timeseries_recommendation__deep_interest_network_attention.html",
    "title": "Deep Interest Network Attention",
    "domain": "Timeseries Recommendation"
  },
  {
    "file": "timeseries_recommendation__deep_learning_recommendation_dlrm.html",
    "title": "Deep Learning Recommendation Dlrm",
    "domain": "Timeseries Recommendation"
  },
  {
    "file": "timeseries_recommendation__multi_gate_mixture_of_experts_mmoe.html",
    "title": "Multi Gate Mixture Of Experts Mmoe",
    "domain": "Timeseries Recommendation"
  },
  {
    "file": "timeseries_recommendation__patch_time_series_transformer.html",
    "title": "Patch Time Series Transformer",
    "domain": "Timeseries Recommendation"
  },
  {
    "file": "timeseries_recommendation__sequential_recommendation_sasrec.html",
    "title": "Sequential Recommendation Sasrec",
    "domain": "Timeseries Recommendation"
  },
  {
    "file": "training_at_scale__context_parallelism_ring_attention.html",
    "title": "Context Parallelism Ring Attention",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__critical_batch_size_noise_scale.html",
    "title": "Critical Batch Size Noise Scale",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__data_mixture_optimization_doremi.html",
    "title": "Data Mixture Optimization Doremi",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__data_parallelism_allreduce.html",
    "title": "Data Parallelism Allreduce",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__differentially_private_training_dpsgd.html",
    "title": "Differentially Private Training Dpsgd",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__distributed_checkpointing_resumption.html",
    "title": "Distributed Checkpointing Resumption",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__expert_parallelism_alltoall.html",
    "title": "Expert Parallelism Alltoall",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__federated_learning_fedavg.html",
    "title": "Federated Learning Fedavg",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__fp8_training.html",
    "title": "Fp8 Training",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__gradient_accumulation.html",
    "title": "Gradient Accumulation",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__long_context_training.html",
    "title": "Long Context Training",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__loss_spike_detection_recovery.html",
    "title": "Loss Spike Detection Recovery",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__mixed_precision_training.html",
    "title": "Mixed Precision Training",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__offloading_activation_parameter.html",
    "title": "Offloading Activation Parameter",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__pipeline_parallelism_1f1b.html",
    "title": "Pipeline Parallelism 1F1B",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__sequence_packing_doc_masking.html",
    "title": "Sequence Packing Doc Masking",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__synthetic_data_generation_filtering.html",
    "title": "Synthetic Data Generation Filtering",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__tensor_parallelism_megatron.html",
    "title": "Tensor Parallelism Megatron",
    "domain": "Training At Scale"
  },
  {
    "file": "training_at_scale__zero_fsdp_sharded_training.html",
    "title": "Zero Fsdp Sharded Training",
    "domain": "Training At Scale"
  },
  {
    "file": "transformer_efficiency__encoder_decoder_transformers.html",
    "title": "Encoder Decoder Transformers",
    "domain": "Transformer Efficiency"
  },
  {
    "file": "transformer_efficiency__flash_attention.html",
    "title": "Flash Attention",
    "domain": "Transformer Efficiency"
  },
  {
    "file": "transformer_efficiency__grouped_query_attention.html",
    "title": "Grouped Query Attention",
    "domain": "Transformer Efficiency"
  },
  {
    "file": "transformer_efficiency__kv_cache.html",
    "title": "Kv Cache",
    "domain": "Transformer Efficiency"
  },
  {
    "file": "transformer_efficiency__linear_attention.html",
    "title": "Linear Attention",
    "domain": "Transformer Efficiency"
  },
  {
    "file": "transformer_efficiency__mamba_selective_ssm.html",
    "title": "Mamba Selective Ssm",
    "domain": "Transformer Efficiency"
  },
  {
    "file": "transformer_efficiency__multi_head_latent_attention.html",
    "title": "Multi Head Latent Attention",
    "domain": "Transformer Efficiency"
  },
  {
    "file": "transformer_efficiency__paged_attention.html",
    "title": "Paged Attention",
    "domain": "Transformer Efficiency"
  },
  {
    "file": "transformer_efficiency__rwkv_retnet_recurrent.html",
    "title": "Rwkv Retnet Recurrent",
    "domain": "Transformer Efficiency"
  },
  {
    "file": "transformer_efficiency__sliding_window_attention.html",
    "title": "Sliding Window Attention",
    "domain": "Transformer Efficiency"
  },
  {
    "file": "transformer_efficiency__sparse_attention_patterns.html",
    "title": "Sparse Attention Patterns",
    "domain": "Transformer Efficiency"
  },
  {
    "file": "transformer_efficiency__structured_state_space_s4.html",
    "title": "Structured State Space S4",
    "domain": "Transformer Efficiency"
  },
  {
    "file": "transformer_efficiency__transformer_block.html",
    "title": "Transformer Block",
    "domain": "Transformer Efficiency"
  }
];
