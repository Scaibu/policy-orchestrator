import pytest
import math

from src.features.code_engine.algos.nn.diffusion_flow_models.ddpm_diffusion_process.impl import (
    NnAlgoDdpmDiffusionProcess,
)
from src.features.code_engine.algos.nn.diffusion_flow_models.score_based_sde.impl import (
    NnAlgoScoreBasedSde,
)
from src.features.code_engine.algos.nn.diffusion_flow_models.ddim_deterministic_sampling.impl import (
    NnAlgoDdimDeterministicSampling,
)
from src.features.code_engine.algos.nn.diffusion_flow_models.fast_diffusion_samplers.impl import (
    NnAlgoFastDiffusionSamplers,
)
from src.features.code_engine.algos.nn.diffusion_flow_models.classifier_free_guidance.impl import (
    NnAlgoClassifierFreeGuidance,
)
from src.features.code_engine.algos.nn.diffusion_flow_models.latent_diffusion_models.impl import (
    NnAlgoLatentDiffusionModels,
)
from src.features.code_engine.algos.nn.diffusion_flow_models.diffusion_transformers_dit.impl import (
    NnAlgoDiffusionTransformersDit,
)
from src.features.code_engine.algos.nn.diffusion_flow_models.flow_matching_rectified_flow.impl import (
    NnAlgoFlowMatchingRectifiedFlow,
)
from src.features.code_engine.algos.nn.diffusion_flow_models.consistency_models_distillation.impl import (
    NnAlgoConsistencyModelsDistillation,
)
from src.features.code_engine.algos.nn.diffusion_flow_models.controlnet_conditioning_adapters.impl import (
    NnAlgoControlnetConditioningAdapters,
)
from src.features.code_engine.algos.nn.diffusion_flow_models.noise_schedules_prediction_targets.impl import (
    NnAlgoNoiseSchedulesPredictionTargets,
)


def test_algo_nn_161_ddpm_diffusion_process():
    x0 = [1.0, -0.5]
    eps = [0.2, 0.4]
    pred_eps = [0.21, 0.39]
    res = NnAlgoDdpmDiffusionProcess.forward(x0, eps, pred_eps, alpha_bar_t=0.64, beta_t=0.02)
    assert len(res["noisy_sample_xt"]) == 2
    assert len(res["predicted_x0"]) == 2
    assert len(res["posterior_mean"]) == 2
    assert res["noise_mse_loss"] > 0.0


def test_algo_nn_162_score_based_sde():
    x = [0.5, 1.2]
    score = [-0.4, -1.0]
    dw = [0.1, -0.1]
    res = NnAlgoScoreBasedSde.forward(x, score, beta_t=0.1, dt=0.01, brownian_noise=dw)
    assert len(res["sde_reverse_step"]) == 2
    assert len(res["ode_reverse_step"]) == 2
    assert res["score_norm"] > 0.0


def test_algo_nn_163_ddim_deterministic_sampling():
    xt = [1.0, 0.5]
    pred_eps = [0.1, 0.2]
    res = NnAlgoDdimDeterministicSampling.forward(xt, pred_eps, alpha_bar_t=0.5, alpha_bar_prev=0.7, eta=0.0)
    assert len(res["next_state_xs"]) == 2
    assert len(res["predicted_x0"]) == 2
    assert res["sigma_t"] == 0.0


def test_algo_nn_164_fast_diffusion_samplers():
    x = [1.0, 2.0]
    vt = [0.5, -0.5]
    vs = [0.4, -0.4]
    res = NnAlgoFastDiffusionSamplers.forward(x, vt, vs, h_step=0.1)
    assert len(res["euler_step"]) == 2
    assert len(res["heun_step"]) == 2
    assert res["local_truncation_error"] > 0.0


def test_algo_nn_165_classifier_free_guidance():
    uncond = [1.0, 2.0]
    cond = [1.5, 2.8]
    res = NnAlgoClassifierFreeGuidance.forward(uncond, cond, guidance_scale=2.0)
    # val = uncond + 2 * (cond - uncond) = 1 + 2*(0.5) = 2.0, 2 + 2*(0.8) = 3.6
    assert math.isclose(res["guided_noise"][0], 2.0, abs_tol=1e-5)
    assert math.isclose(res["guided_noise"][1], 3.6, abs_tol=1e-5)
    assert res["guidance_delta_norm"] > 0.0


def test_algo_nn_166_latent_diffusion_models():
    z0 = [[1.0, 2.0], [3.0, 4.0]]
    noise = [[0.1, -0.1], [0.2, -0.2]]
    res = NnAlgoLatentDiffusionModels.forward(z0, noise, alpha_bar_t=0.8, compression_factor=8)
    assert len(res["diffused_latent_zt"]) == 2
    assert res["memory_reduction_ratio"] == 64.0


def test_algo_nn_167_diffusion_transformers_dit():
    tokens = [[1.0, 2.0], [3.0, 4.0]]
    gamma = [0.1, -0.1]
    beta = [0.0, 0.0]
    alpha_gate = [0.0, 0.0]
    sublayer = [[0.5, 0.5], [1.0, 1.0]]

    res = NnAlgoDiffusionTransformersDit.forward(tokens, gamma, beta, alpha_gate, sublayer)
    assert len(res["normalized_modulated_tokens"]) == 2
    # With alpha_gate = 0, gated_residual_output must strictly equal input tokens
    for i in range(2):
        for j in range(2):
            assert math.isclose(res["gated_residual_output"][i][j], tokens[i][j], abs_tol=1e-5)


def test_algo_nn_168_flow_matching_rectified_flow():
    x0 = [0.0, 0.0]
    x1 = [2.0, 4.0]
    pred_v = [1.9, 4.1]
    res = NnAlgoFlowMatchingRectifiedFlow.forward(x0, x1, pred_v, timestep_t=0.5)
    assert res["target_velocity"] == [2.0, 4.0]
    assert res["interpolated_xt"] == [1.0, 2.0]
    assert res["velocity_matching_loss"] > 0.0


def test_algo_nn_169_consistency_models_distillation():
    xt = [1.0, 2.0]
    raw_f = [0.8, 1.9]
    res = NnAlgoConsistencyModelsDistillation.forward(xt, raw_f, t_step=0.002, epsilon=0.002)
    # At t = epsilon, boundary condition ensures output strictly equals xt
    assert math.isclose(res["consistency_output"][0], xt[0], abs_tol=1e-5)
    assert math.isclose(res["consistency_output"][1], xt[1], abs_tol=1e-5)


def test_algo_nn_170_controlnet_conditioning_adapters():
    x = [1.0, 2.0]
    c = [0.5, 0.5]
    locked_w = [[1.0, 0.0], [0.0, 1.0]]
    adapter_w = [[1.0, 0.0], [0.0, 1.0]]
    zero_in = [[0.0, 0.0], [0.0, 0.0]]
    zero_out = [[0.0, 0.0], [0.0, 0.0]]

    res = NnAlgoControlnetConditioningAdapters.forward(x, c, locked_w, adapter_w, zero_in, zero_out)
    # With zero-conv initialized to 0, combined_output must strictly equal base_feature_output
    assert res["combined_output"] == res["base_feature_output"]
    assert res["zero_injection_norm"] == 0.0


def test_algo_nn_171_noise_schedules_prediction_targets():
    x0 = [1.0, 2.0]
    eps = [0.5, -0.5]
    res = NnAlgoNoiseSchedulesPredictionTargets.forward(timestep_t=50, total_steps_T=100, clean_x0=x0, noise_epsilon=eps)
    assert 0.0 < res["alpha_bar_t"] < 1.0
    assert len(res["target_v"]) == 2
    # Inverted x0 from v must match original clean_x0
    for i in range(2):
        assert math.isclose(res["reconstructed_x0_from_v"][i], x0[i], abs_tol=1e-5)
