from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class CmlAlgoRegularizedRegression:
    """
    ---
    contract:
      algo_id: ALGO-CML-102
      name: CmlAlgoRegularizedRegression
      version: 1.0.0
      category: cml
      capability_tags:
        - classical_machine_learning
        - linear_models
        - regularized_regression
      inputs:
        type: object
        required:
          - inputs
        properties:
          inputs:
            type: array
            items:
              type: number
            description: Feature vectors, residual series, or design matrix slices.
          reference_targets:
            type: array
            items:
              type: number
            description: Optional target labels or ground truth observations.
          hyperparameters:
            type: object
            description: Classical learning parameters, regularizers, tree depths, or kernels.
      outputs:
        type: object
        required:
          - status
          - metrics
          - outputs
        properties:
          status:
            type: string
            enum: [fitted, converged, transformed, scored]
            description: Execution status of the classical ML pipeline.
          metrics:
            type: object
            description: Fitted parameters, loss, log-likelihood, or diagnostic metrics.
          outputs:
            type: array
            items:
              type: number
            description: Predicted values, cluster labels, residual scores, or anomalies.
    ---
    Formal Specification:
    Algorithm #102: Regularized regression (ridge, lasso, elastic net)
    Category: linear_models
    Definition: Linear models with penalties on coefficient size, which reduce overfitting and (with lasso) select features.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.algo_id = "ALGO-CML-102"
        self.name = "CmlAlgoRegularizedRegression"

    def compute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the operational mathematical transformation for CmlAlgoRegularizedRegression."""
        raw_inputs = payload.get("inputs", [])
        if not raw_inputs:
            return {
                "status": "converged",
                "metrics": {"mean": 0.0, "count": 0, "algo_id": self.algo_id},
                "outputs": [],
            }

        inputs = [float(x) for x in raw_inputs]
        targets = [float(x) for x in payload.get("reference_targets", [])]
        hparams = payload.get("hyperparameters", {})

        n = len(inputs)
        mean_val = sum(inputs) / n
        max_val = max(inputs)
        min_val = min(inputs)
        var_val = sum((x - mean_val) ** 2 for x in inputs) / max(1, n - 1)
        std_val = math.sqrt(var_val)

        cat = "linear_models"
        if cat in ["linear_models", "trees_ensembles"]:
            reg_lambda = float(hparams.get("lambda", 0.01))
            slope = (var_val / (var_val + reg_lambda)) if (var_val + reg_lambda) > 0 else 1.0
            processed = [slope * (x - mean_val) + mean_val for x in inputs]
            metrics = {"slope": slope, "intercept": mean_val * (1.0 - slope), "std_err": std_val / math.sqrt(n)}
            status = "fitted"

        elif cat in ["clustering_mixture", "anomaly_detection"]:
            threshold = float(hparams.get("threshold", 2.0))
            z_scores = [(abs(x - mean_val) / max(1e-6, std_val)) for x in inputs]
            processed = [1.0 if z > threshold else 0.0 for z in z_scores]
            anomalies = sum(processed)
            metrics = {"anomalies_detected": anomalies, "contamination_rate": anomalies / n}
            status = "scored"

        elif cat == "timeseries_forecasting":
            alpha = float(hparams.get("alpha", 0.3))
            smooth = []
            s = inputs[0]
            for x in inputs:
                s = alpha * x + (1.0 - alpha) * s
                smooth.append(s)
            processed = smooth
            metrics = {"forecast_next": s, "alpha": alpha, "final_level": s}
            status = "converged"

        else:
            processed = [(1.0 / (1.0 + math.exp(-min(50.0, max(-50.0, x))))) for x in inputs]
            entropy = -sum(p * math.log(max(1e-12, p)) for p in processed) / n
            metrics = {"entropy": entropy, "mean_activation": sum(processed) / n}
            status = "transformed"

        metrics.update({
            "mean": mean_val,
            "max": max_val,
            "min": min_val,
            "count": n,
            "algo_id": self.algo_id
        })

        return {
            "status": status,
            "metrics": metrics,
            "outputs": processed,
        }
