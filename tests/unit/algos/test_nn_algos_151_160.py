import pytest
import math

from src.features.code_engine.algos.nn.generative_autoencoders_gans.denoising_autoencoder.impl import (
    NnAlgoDenoisingAutoencoder,
)
from src.features.code_engine.algos.nn.generative_autoencoders_gans.variational_autoencoder.impl import (
    NnAlgoVariationalAutoencoder,
)
from src.features.code_engine.algos.nn.generative_autoencoders_gans.vector_quantized_vae.impl import (
    NnAlgoVectorQuantizedVae,
)
from src.features.code_engine.algos.nn.generative_autoencoders_gans.generative_adversarial_network.impl import (
    NnAlgoGenerativeAdversarialNetwork,
)
from src.features.code_engine.algos.nn.generative_autoencoders_gans.wasserstein_gan_gp.impl import (
    NnAlgoWassersteinGanGp,
)
from src.features.code_engine.algos.nn.generative_autoencoders_gans.stylegan_modulation.impl import (
    NnAlgoStyleganModulation,
)
from src.features.code_engine.algos.nn.generative_autoencoders_gans.conditional_image_to_image_gan.impl import (
    NnAlgoConditionalImageToImageGan,
)
from src.features.code_engine.algos.nn.generative_autoencoders_gans.normalizing_flows.impl import (
    NnAlgoNormalizingFlows,
)
from src.features.code_engine.algos.nn.generative_autoencoders_gans.autoregressive_pixelcnn.impl import (
    NnAlgoAutoregressivePixelcnn,
)
from src.features.code_engine.algos.nn.generative_autoencoders_gans.energy_based_models.impl import (
    NnAlgoEnergyBasedModels,
)


def test_algo_nn_151_denoising_autoencoder():
    clean = [1.0, 2.0]
    corrupted = [1.1, 1.9]
    w_e = [[0.5, 0.5], [0.2, 0.8]]
    b_e = [0.0, 0.0]
    w_d = [[0.5, 0.2], [0.5, 0.8]]
    b_d = [0.0, 0.0]

    res = NnAlgoDenoisingAutoencoder.forward(clean, corrupted, w_e, b_e, w_d, b_d)
    assert len(res["reconstruction"]) == 2
    assert len(res["latent_representation"]) == 2
    assert res["reconstruction_loss"] >= 0.0
    assert res["anomaly_score"] == res["reconstruction_loss"]

    # Precondition check
    with pytest.raises(ValueError):
        NnAlgoDenoisingAutoencoder.forward([], corrupted, w_e, b_e, w_d, b_d)


def test_algo_nn_152_variational_autoencoder():
    x = [0.5, -0.2]
    w_mu = [[0.1, 0.2]]
    b_mu = [0.0]
    w_lv = [[-0.1, 0.1]]
    b_lv = [0.0]
    eps = [0.5]
    w_d = [[0.2], [0.4]]
    b_d = [0.0, 0.0]

    res = NnAlgoVariationalAutoencoder.forward(x, w_mu, b_mu, w_lv, b_lv, eps, w_d, b_d, beta=1.0)
    assert len(res["reconstruction"]) == 2
    assert len(res["latent_z"]) == 1
    assert res["kl_divergence"] >= 0.0
    assert math.isclose(res["total_loss"], res["reconstruction_loss"] + res["kl_divergence"], rel_tol=1e-5)


def test_algo_nn_153_vector_quantized_vae():
    z_e = [[0.1, 0.2], [0.9, 0.8]]
    codebook = [[0.0, 0.0], [1.0, 1.0]]

    res = NnAlgoVectorQuantizedVae.forward(z_e, codebook, beta=0.25)
    assert len(res["quantized_vectors"]) == 2
    assert res["codebook_indices"] == [0, 1]
    assert res["quantization_loss"] >= 0.0
    assert res["perplexity"] >= 1.0


def test_algo_nn_154_generative_adversarial_network():
    real_p = [0.9, 0.8, 0.7]
    fake_p = [0.1, 0.2, 0.3]

    res = NnAlgoGenerativeAdversarialNetwork.forward(real_p, fake_p)
    assert res["discriminator_loss"] > 0.0
    assert res["generator_loss"] > 0.0
    assert res["d_real_accuracy"] == 1.0
    assert res["d_fake_accuracy"] == 1.0


def test_algo_nn_155_wasserstein_gan_gp():
    real_scores = [2.5, 3.0]
    fake_scores = [-1.0, -1.5]
    interp_grads = [[0.6, 0.8], [1.0, 0.0]]  # both have norm = 1.0

    res = NnAlgoWassersteinGanGp.forward(real_scores, fake_scores, interp_grads, lambda_gp=10.0)
    assert res["wasserstein_distance"] == (2.75 - (-1.25))  # 4.0
    assert math.isclose(res["gradient_penalty"], 0.0, abs_tol=1e-6)
    assert math.isclose(res["critic_loss"], -4.0, abs_tol=1e-6)


def test_algo_nn_156_stylegan_modulation():
    w = [0.5, -0.5]
    affine_w = [[0.2, 0.4], [0.1, -0.1]]
    affine_b = [0.0, 0.0]
    # shape: C_out=2, C_in=2, K=1
    conv_w = [[[1.0], [2.0]], [[0.5], [1.5]]]

    res = NnAlgoStyleganModulation.forward(w, affine_w, affine_b, conv_w)
    assert len(res["style_scales"]) == 2
    assert len(res["demodulated_weights"]) == 2
    assert len(res["channel_norms"]) == 2


def test_algo_nn_157_conditional_image_to_image_gan():
    src_a = [1.0, 0.5]
    tgt_b = [0.2, 0.8]
    fake_b = [0.25, 0.75]
    cycle_a = [0.95, 0.55]
    fake_a = [0.9, 0.6]
    cycle_b = [0.22, 0.78]

    res = NnAlgoConditionalImageToImageGan.forward(
        src_a, tgt_b, fake_b, cycle_a, fake_a, cycle_b, lambda_cycle=10.0, is_paired=True
    )
    assert res["cycle_consistency_loss"] > 0.0
    assert res["l1_paired_loss"] > 0.0
    assert res["total_generator_objective"] > 0.0


def test_algo_nn_158_normalizing_flows():
    z = [0.5, -0.5, 1.0, 2.0]
    scale_w = [[0.1, 0.2], [-0.1, 0.3]]
    scale_b = [0.0, 0.0]
    trans_w = [[0.5, 0.1], [0.2, -0.2]]
    trans_b = [0.0, 0.0]

    # Forward
    res_fwd = NnAlgoNormalizingFlows.forward(z, scale_w, scale_b, trans_w, trans_b, split_dim=2, inverse=False)
    x = res_fwd["output_vector"]
    assert len(x) == 4
    assert x[:2] == z[:2]

    # Inverse
    res_inv = NnAlgoNormalizingFlows.forward(x, scale_w, scale_b, trans_w, trans_b, split_dim=2, inverse=True)
    z_rec = res_inv["output_vector"]
    for i in range(4):
        assert math.isclose(z[i], z_rec[i], abs_tol=1e-5)


def test_algo_nn_159_autoregressive_pixelcnn():
    seq = [0, 1, 2]
    weights = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.0], [0.0, 0.3, 0.2]]  # V=3, ctx=3
    bias = [0.0, 0.0, 0.0]

    res = NnAlgoAutoregressivePixelcnn.forward(seq, weights, bias, vocab_size=3)
    assert res["total_nll"] > 0.0
    assert res["bits_per_dim"] > 0.0
    assert len(res["predicted_probabilities"]) == 3
    for p in res["predicted_probabilities"]:
        assert math.isclose(sum(p), 1.0, rel_tol=1e-5)


def test_algo_nn_160_energy_based_models():
    v = [1.0, 0.0]
    w = [[0.5, -0.5], [-0.2, 0.8]]
    a = [0.0, 0.0]
    b = [0.0, 0.0]

    res = NnAlgoEnergyBasedModels.forward(v, w, a, b)
    assert len(res["reconstructed_visible"]) == 2
    assert len(res["weight_gradient_cd1"]) == 2
    assert res["reconstruction_error"] >= 0.0
    assert isinstance(res["contrastive_divergence_loss"], float)
