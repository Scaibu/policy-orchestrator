import pytest
import math

from src.features.code_engine.algos.nn.specialized_scientific_meta.prototypical_networks.impl import NnAlgoPrototypicalNetworks
from src.features.code_engine.algos.nn.specialized_scientific_meta.maml_meta_learning.impl import NnAlgoMamlMetaLearning
from src.features.code_engine.algos.nn.specialized_scientific_meta.neural_ordinary_differential_equations.impl import NnAlgoNeuralOrdinaryDifferentialEquations
from src.features.code_engine.algos.nn.specialized_scientific_meta.neural_radiance_fields_nerf.impl import NnAlgoNeuralRadianceFieldsNerf
from src.features.code_engine.algos.nn.specialized_scientific_meta.gaussian_splatting_3d.impl import NnAlgoGaussianSplatting3d
from src.features.code_engine.algos.nn.specialized_scientific_meta.physics_informed_neural_networks.impl import NnAlgoPhysicsInformedNeuralNetworks
from src.features.code_engine.algos.nn.specialized_scientific_meta.fourier_neural_operator.impl import NnAlgoFourierNeuralOperator
from src.features.code_engine.algos.nn.specialized_scientific_meta.hypernetworks_conditioning.impl import NnAlgoHypernetworksConditioning
from src.features.code_engine.algos.nn.specialized_scientific_meta.differentiable_neural_memory.impl import NnAlgoDifferentiableNeuralMemory
from src.features.code_engine.algos.nn.specialized_scientific_meta.differentiable_architecture_search_darts.impl import NnAlgoDifferentiableArchitectureSearchDarts
from src.features.code_engine.algos.nn.specialized_scientific_meta.spiking_neural_networks_lif.impl import NnAlgoSpikingNeuralNetworksLif
from src.features.code_engine.algos.nn.specialized_scientific_meta.mixture_density_networks.impl import NnAlgoMixtureDensityNetworks


def test_algo_177_prototypical_networks():
    supp = [[1.0, 1.0], [1.1, 0.9], [5.0, 5.0], [5.1, 4.9]]
    labels = [0, 0, 1, 1]
    query = [[1.05, 0.95], [5.05, 5.05]]

    res = NnAlgoPrototypicalNetworks.forward(supp, labels, query, num_classes=2)
    assert len(res["class_prototypes"]) == 2
    assert res["predicted_classes"] == [0, 1]
    assert len(res["query_log_probabilities"]) == 2


def test_algo_178_maml_meta_learning():
    theta = [1.0, 2.0]
    supp_g = [[0.1, 0.2], [0.2, 0.1]]
    query_g = [[0.05, 0.1], [0.1, 0.05]]

    res = NnAlgoMamlMetaLearning.forward(theta, supp_g, query_g, alpha_inner_lr=0.1, beta_outer_lr=0.01)
    assert len(res["updated_meta_weights"]) == 2
    assert len(res["adapted_task_weights"]) == 2
    assert res["meta_gradient_norm"] > 0.0


def test_algo_179_neural_ode():
    h0 = [1.0, 2.0]
    k1 = [0.1, 0.2]
    k2 = [0.12, 0.22]
    k3 = [0.12, 0.22]
    k4 = [0.14, 0.24]

    res = NnAlgoNeuralOrdinaryDifferentialEquations.forward(h0, k1, k2, k3, k4, dt_step=0.1)
    assert len(res["integrated_state_next"]) == 2
    assert res["state_displacement_norm"] > 0.0


def test_algo_180_nerf():
    densities = [1.0, 2.0, 0.5]
    colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    dists = [0.1, 0.1, 0.1]

    res = NnAlgoNeuralRadianceFieldsNerf.forward(densities, colors, dists)
    assert len(res["composited_rgb"]) == 3
    assert 0.0 <= res["total_opacity"] <= 1.0
    assert 0.0 <= res["accumulated_transmittance"] <= 1.0


def test_algo_181_gaussian_splatting():
    px = [0.0, 0.0]
    centers = [[0.0, 0.0], [1.0, 1.0]]
    inv_covs = [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    opacities = [0.8, 0.5]
    colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    res = NnAlgoGaussianSplatting3d.forward(px, centers, inv_covs, opacities, colors)
    assert len(res["pixel_rgb"]) == 3
    assert 0.0 <= res["terminal_transmittance"] <= 1.0


def test_algo_182_pinns():
    pde_res = [0.01, -0.02, 0.015]
    ic_pred = [1.0, 0.5]
    ic_targ = [1.0, 0.5]
    bc_pred = [0.0, 0.0]
    bc_targ = [0.0, 0.0]

    res = NnAlgoPhysicsInformedNeuralNetworks.forward(pde_res, ic_pred, ic_targ, bc_pred, bc_targ)
    assert res["total_pinn_loss"] > 0.0
    assert res["ic_loss"] == 0.0
    assert res["bc_loss"] == 0.0


def test_algo_183_fno():
    signal = [1.0, 2.0, 1.5, 0.5, -0.5, -1.0, -0.5, 0.5]
    w_r = [0.5, 0.2]
    w_i = [0.0, 0.1]

    res = NnAlgoFourierNeuralOperator.forward(signal, w_r, w_i, local_weight=1.0, num_modes=2)
    assert len(res["transformed_signal"]) == len(signal)
    assert res["spectral_energy_retained"] > 0.0


def test_algo_184_hypernetworks():
    x = [1.0, 2.0]
    c = [0.5, -0.5]
    base_w = [[1.0, 0.0], [0.0, 1.0]]
    # rank = 1, D_out = 2, D_in = 2, d_c = 2
    h_a = [[0.1, 0.2], [0.3, 0.4]]
    h_b = [[0.5, 0.6], [0.7, 0.8]]

    res = NnAlgoHypernetworksConditioning.forward(x, c, base_w, h_a, h_b, rank_r=1)
    assert len(res["output_vector_y"]) == 2
    assert res["effective_weight_norm"] > 0.0


def test_algo_185_differentiable_memory():
    mem = [[1.0, 0.0], [0.0, 1.0]]
    key = [1.0, 0.0]

    res = NnAlgoDifferentiableNeuralMemory.forward(mem, key, key_strength_beta=1.0)
    assert len(res["read_vector"]) == 2
    assert len(res["read_weights"]) == 2
    assert res["read_weights"][0] > res["read_weights"][1]


def test_algo_186_darts():
    logits = [1.0, -1.0]
    ops = [[2.0, 3.0], [0.0, 1.0]]

    res = NnAlgoDifferentiableArchitectureSearchDarts.forward(logits, ops)
    assert len(res["mixed_edge_output"]) == 2
    assert res["winning_op_index"] == 0


def test_algo_187_lif_snn():
    currents = [[0.5, 1.5], [0.5, 0.0], [0.5, 0.0]]
    v0 = [0.0, 0.0]

    res = NnAlgoSpikingNeuralNetworksLif.forward(currents, v0, leak_factor_beta=0.9, threshold_vth=1.0)
    assert len(res["output_spike_trains"]) == 3
    assert res["total_spikes_emitted"] >= 1


def test_algo_188_mdn():
    logits = [0.0, 0.0]
    means = [1.0, 5.0]
    log_stds = [0.0, 0.0]
    target = 1.0

    res = NnAlgoMixtureDensityNetworks.forward(logits, means, log_stds, target_y=target)
    assert res["negative_log_likelihood"] > 0.0
    assert abs(sum(res["mixture_probabilities"]) - 1.0) < 1e-6
