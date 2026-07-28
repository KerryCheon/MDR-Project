from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cache")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from common import ARTIFACT_DIR, EXPERIMENT_DIR, temporal_feature_variants
from run_hybrid import prepare_source_splits
from temporal import FrozenLSTMExtractor, build_sequences


def main() -> None:
    config = yaml.safe_load(
        (EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8")
    )
    temporal_config = config["temporal_encoder"]
    selected_name = "alternative_without_smap_x_year"
    features = temporal_feature_variants()[selected_name]
    encoder_dir = ARTIFACT_DIR / "hybrid" / "encoders" / selected_name
    output_dir = ARTIFACT_DIR / "hybrid" / "hidden_trace"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, combined, _ = prepare_source_splits()
    preprocessor = joblib.load(encoder_dir / "preprocessor.joblib")
    scaled = preprocessor.transform(combined, features)
    test_data = build_sequences(
        scaled,
        features,
        combined["_split"] == "test",
        sequence_length=temporal_config["sequence_length"],
        max_span_factor=temporal_config["max_window_span_factor"],
    )

    checkpoint = torch.load(encoder_dir / "best_model.pt", map_location="cpu")
    model = FrozenLSTMExtractor(
        n_features=len(features),
        projection_size=temporal_config["projection_size"],
        hidden_size=temporal_config["hidden_size"],
        embedding_size=temporal_config["embedding_size"],
        dropout=temporal_config["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    candidates = np.flatnonzero(
        test_data.metadata["station_id"].to_numpy() == "Touchet_WA_824"
    )
    index = int(candidates[len(candidates) // 2])
    sequence = torch.from_numpy(test_data.X[index : index + 1])
    with torch.no_grad():
        trace = model.forward_with_trace(sequence)

    input_values = trace["input"].squeeze(0).numpy()
    projected = trace["projected"].squeeze(0).numpy()
    recurrent = trace["recurrent"].squeeze(0).numpy()
    embedding = trace["embedding"].squeeze(0).numpy()
    head_weight = model.head.weight.squeeze(0).detach().numpy()
    contributions = embedding * head_weight

    np.savez_compressed(
        output_dir / "touchet_hidden_chain_example.npz",
        input=input_values,
        projected=projected,
        recurrent=recurrent,
        final_hidden=trace["final_hidden"].squeeze(0).numpy(),
        embedding=embedding,
        head_weight=head_weight,
        head_contribution=contributions,
        prediction=trace["prediction"].numpy(),
    )
    pd.DataFrame(input_values, columns=features).to_csv(
        output_dir / "touchet_scaled_input_window.csv", index=False
    )
    pd.DataFrame(projected).to_csv(
        output_dir / "touchet_projected_vectors.csv", index=False
    )
    pd.DataFrame(recurrent).to_csv(
        output_dir / "touchet_recurrent_vectors.csv", index=False
    )
    contribution_frame = pd.DataFrame(
        {
            "dimension": np.arange(len(embedding)),
            "embedding": embedding,
            "head_weight": head_weight,
            "contribution": contributions,
            "absolute_contribution": np.abs(contributions),
        }
    ).sort_values("absolute_contribution", ascending=False)
    contribution_frame.to_csv(
        output_dir / "touchet_embedding_head_contributions.csv", index=False
    )

    figure, axes = plt.subplots(4, 1, figsize=(14, 14))
    input_image = axes[0].imshow(input_values.T, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    axes[0].set_yticks(np.arange(len(features)))
    axes[0].set_yticklabels(features, fontsize=7)
    axes[0].set_ylabel("Input feature")
    axes[0].set_title("1. Standardized temporal input window")
    figure.colorbar(input_image, ax=axes[0], fraction=0.02)

    projection_image = axes[1].imshow(projected.T, aspect="auto", cmap="viridis")
    axes[1].set_ylabel("Projection dimension")
    axes[1].set_title("2. Per-timestep projected vectors (48 dimensions)")
    figure.colorbar(projection_image, ax=axes[1], fraction=0.02)

    recurrent_image = axes[2].imshow(recurrent.T, aspect="auto", cmap="viridis")
    axes[2].set_ylabel("Hidden dimension")
    axes[2].set_title("3. LSTM hidden-state sequence (64 dimensions)")
    figure.colorbar(recurrent_image, ax=axes[2], fraction=0.02)

    axes[3].bar(np.arange(len(contributions)), contributions, color="tab:purple")
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_xlabel("Embedding dimension")
    axes[3].set_ylabel("z_i × head_weight_i")
    axes[3].set_title(
        "4. Embedding contributions to temporary dense head "
        f"(prediction={trace['prediction'].item():.3f})"
    )
    figure.suptitle(
        "Touchet hidden-vector chain, target "
        f"{test_data.metadata.iloc[index]['date'].date()}, true={test_data.y[index]:.3f}"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "touchet_hidden_vector_chain_heatmap.png", dpi=170)
    plt.close(figure)

    top = contribution_frame.head(15).sort_values("contribution")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        [f"z_{int(value):02d}" for value in top["dimension"]],
        top["contribution"],
        color=["tab:red" if value < 0 else "tab:blue" for value in top["contribution"]],
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Contribution to temporary scalar head")
    axis.set_title("Largest Touchet embedding-to-head contributions")
    axis.grid(axis="x", alpha=0.20)
    figure.tight_layout()
    figure.savefig(output_dir / "touchet_top_embedding_contributions.png", dpi=160)
    plt.close(figure)

    print(
        f"[trace] Touchet {test_data.metadata.iloc[index]['date'].date()} "
        f"true={test_data.y[index]:.4f} pred={trace['prediction'].item():.4f}"
    )
    print(contribution_frame.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

