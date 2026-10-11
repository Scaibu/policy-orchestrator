from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoSpikingNeuralNetworksLif:
    """
    ---
    contract:
      algo_id: ALGO-NN-187
      name: NnAlgoSpikingNeuralNetworksLif
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - neuromorphic
        - snn
        - leaky_integrate_and_fire
        - lif_neuron
        - surrogate_gradients
      inputs:
        type: object
        required:
          - input_currents
          - initial_membrane_potentials
        properties:
          input_currents:
            type: array
            items:
              type: array
              items:
                type: number
            description: Input synaptic currents I[t, i] across timesteps of shape (T_timesteps, N_neurons).
          initial_membrane_potentials:
            type: array
            items:
              type: number
            description: Starting membrane potentials V_0 of length N_neurons.
          leak_factor_beta:
            type: number
            default: 0.9
            minimum: 0.0
            maximum: 1.0
            description: Membrane potential decay factor beta in (0, 1).
          threshold_vth:
            type: number
            default: 1.0
            minimum: 0.01
            description: Firing voltage threshold V_th > 0.
          reset_mode:
            type: string
            default: hard
            enum:
              - hard
              - soft
            description: Reset mechanism after spike (hard to 0, soft subtract V_th).
          surrogate_scale:
            type: number
            default: 1.0
            description: Scale parameter alpha for ATan/Sigmoid surrogate gradient.
      outputs:
        type: object
        required:
          - output_spike_trains
          - final_membrane_potentials
          - mean_firing_rates
          - total_spikes_emitted
          - surrogate_gradients
        properties:
          output_spike_trains:
            type: array
            items:
              type: array
              items:
                type: integer
            description: Binary emitted spikes S[t, i] in {0, 1} of shape (T_timesteps, N_neurons).
          final_membrane_potentials:
            type: array
            items:
              type: number
            description: Terminal membrane voltages V_T of length N_neurons.
          mean_firing_rates:
            type: array
            items:
              type: number
            description: Average spike rate per neuron across all T timesteps.
          total_spikes_emitted:
            type: integer
            description: Total integer spike count across all neurons and timesteps.
          surrogate_gradients:
            type: array
            items:
              type: array
              items:
                type: number
            description: Evaluated ATan surrogate derivative values dS/dV at each timestep.
      parameters: {}
      input_assumptions:
        - input_currents has shape (T, N) with T >= 1, N >= 1
        - initial_membrane_potentials has length N
        - 0.0 < leak_factor_beta < 1.0
        - threshold_vth > 0.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact discrete time LIF update"
      uses_model: false
      complexity:
        variables:
          T: number of simulation timesteps
          N: number of spiking neurons
        time_worst: O(T * N)
        time_typical: O(T * N)
        space: O(T * N)
      preconditions:
        - len(input.input_currents) > 0 and len(input.input_currents[0]) > 0
        - len(input.initial_membrane_potentials) == len(input.input_currents[0])
        - 0.0 <= input.leak_factor_beta <= 1.0
        - input.threshold_vth > 0.0
      postconditions:
        - len(output.output_spike_trains) == len(input.input_currents)
        - len(output.final_membrane_potentials) == len(input.initial_membrane_potentials)
        - output.total_spikes_emitted >= 0
      certificate: "Neuron emits spike if and only if membrane potential reaches or exceeds threshold_vth"
      compatible_adapters:
        - ADAPTER-NEUROMORPHIC-LIF-LAYER
        - ADAPTER-SURROGATE-GRADIENT-SNN
      related_algos:
        - ALGO-NN-90
      references:
        - "https://doi.org/10.1109/JPROC.2021.3067594"
    ---
    """

    @classmethod
    def forward(
        cls,
        input_currents: List[List[float]],
        initial_membrane_potentials: List[float],
        leak_factor_beta: float = 0.9,
        threshold_vth: float = 1.0,
        reset_mode: str = "hard",
        surrogate_scale: float = 1.0,
    ) -> Dict[str, Any]:
        T = len(input_currents)
        if T == 0 or len(input_currents[0]) == 0:
            raise ValueError("Precondition failed: input_currents cannot be empty")
        N = len(input_currents[0])

        if len(initial_membrane_potentials) != N or any(len(r) != N for r in input_currents):
            raise ValueError("Precondition failed: dimension mismatch with initial_membrane_potentials")
        if not (0.0 <= leak_factor_beta <= 1.0) or threshold_vth <= 0.0:
            raise ValueError("Precondition failed: invalid leak factor or threshold")

        current_v = list(initial_membrane_potentials)
        spike_trains: List[List[int]] = []
        surrogate_grads: List[List[float]] = []
        spike_counts = [0] * N
        total_spikes = 0

        # Fast ATan surrogate gradient: sigma'(x) = 1 / (pi * (1 + (alpha * x)^2))
        inv_pi = 1.0 / math.pi

        for t in range(T):
            step_spikes: List[int] = []
            step_surrogates: List[float] = []

            for i in range(N):
                # 1. Leaky integration: V[t] = beta * V[t-1] + I[t]
                v_decayed = leak_factor_beta * current_v[i] + input_currents[t][i]

                # 2. Threshold check: S[t] = 1 if V[t] >= V_th else 0
                diff = v_decayed - threshold_vth
                spike = 1 if diff >= 0.0 else 0

                # 3. Surrogate gradient evaluation at threshold boundary
                u = surrogate_scale * diff
                surrogate_val = inv_pi * (surrogate_scale / (1.0 + u * u))

                # 4. Reset potential
                if spike == 1:
                    spike_counts[i] += 1
                    total_spikes += 1
                    if reset_mode.lower() == "soft":
                        current_v[i] = v_decayed - threshold_vth
                    else:
                        current_v[i] = 0.0  # Hard reset
                else:
                    current_v[i] = v_decayed

                step_spikes.append(spike)
                step_surrogates.append(surrogate_val)

            spike_trains.append(step_spikes)
            surrogate_grads.append(step_surrogates)

        firing_rates = [float(sc) / float(T) for sc in spike_counts]

        return {
            "output_spike_trains": spike_trains,
            "final_membrane_potentials": current_v,
            "mean_firing_rates": firing_rates,
            "total_spikes_emitted": total_spikes,
            "surrogate_gradients": surrogate_grads,
        }
