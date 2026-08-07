from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 경로 설정
# ============================================================

# 이 파일이 codes/ 폴더 안에 있다고 가정
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd()
)

RESULT_DIR = PROJECT_ROOT / "results" / "mnist"

PLOT_DIR = PROJECT_ROOT / "plots" / "mnist"
PLOT_RUNS_DIR = PLOT_DIR / "runs"
PLOT_COMPARISON_DIR = PLOT_DIR / "comparison"

PLOT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CSV 파일 경로
# ============================================================

epoch_history_path = RESULT_DIR / "epoch_history.csv"
run_summary_path = RESULT_DIR / "run_summary.csv"


if not epoch_history_path.exists():
    raise FileNotFoundError(
        f"epoch_history.csv를 찾을 수 없습니다: {epoch_history_path}"
    )

if not run_summary_path.exists():
    raise FileNotFoundError(
        f"run_summary.csv를 찾을 수 없습니다: {run_summary_path}"
    )


# ============================================================
# 3. CSV 불러오기
# ============================================================

epoch_history_df = pd.read_csv(epoch_history_path)
run_summary_df = pd.read_csv(run_summary_path)


# ============================================================
# 4. 비교할 실험 정의
# ============================================================

experiment_order = [
    "MLP_SGD_lr0.01_seed42",
    "MLP_Adam_lr0.001_seed42",
    "CNN_SGD_lr0.01_seed42",
    "CNN_Adam_lr0.001_seed42",
]


# 그래프에 표시할 이름
experiment_labels = {
    "MLP_SGD_lr0.01_seed42": "MLP + SGD (lr=0.01)",
    "MLP_Adam_lr0.001_seed42": "MLP + Adam (lr=0.001)",
    "CNN_SGD_lr0.01_seed42": "CNN + SGD (lr=0.01)",
    "CNN_Adam_lr0.001_seed42": "CNN + Adam (lr=0.001)",
}


# ============================================================
# 5. 필요한 실험만 선택
# ============================================================

epoch_history_df = epoch_history_df[
    epoch_history_df["experiment_id"].isin(experiment_order)
].copy()

run_summary_df = run_summary_df[
    run_summary_df["experiment_id"].isin(experiment_order)
].copy()


# 같은 실험을 실수로 여러 번 실행한 경우
# 동일한 experiment_id와 epoch 중 가장 마지막 행 사용
epoch_history_df = epoch_history_df.drop_duplicates(
    subset=["experiment_id", "epoch"],
    keep="last"
)

run_summary_df = run_summary_df.drop_duplicates(
    subset=["experiment_id"],
    keep="last"
)


# ============================================================
# 6. 네 실험이 모두 존재하는지 확인
# ============================================================

history_experiments = set(epoch_history_df["experiment_id"])
summary_experiments = set(run_summary_df["experiment_id"])

missing_history = set(experiment_order) - history_experiments
missing_summary = set(experiment_order) - summary_experiments


if missing_history:
    raise ValueError(
        "epoch_history.csv에 다음 실험이 없습니다: "
        f"{sorted(missing_history)}"
    )

if missing_summary:
    raise ValueError(
        "run_summary.csv에 다음 실험이 없습니다: "
        f"{sorted(missing_summary)}"
    )


# 각 실험에 epoch 1~20이 모두 있는지 확인
expected_epochs = set(range(1, 21))

for experiment_id in experiment_order:

    experiment_epochs = set(
        epoch_history_df.loc[
            epoch_history_df["experiment_id"] == experiment_id,
            "epoch"
        ]
    )

    missing_epochs = expected_epochs - experiment_epochs

    if missing_epochs:
        raise ValueError(
            f"{experiment_id}에 누락된 epoch가 있습니다: "
            f"{sorted(missing_epochs)}"
        )


# ============================================================
# 7. 개별 실험 그래프 생성
# ============================================================

for experiment_id in experiment_order:

    experiment_df = epoch_history_df[
        epoch_history_df["experiment_id"] == experiment_id
    ].sort_values("epoch")

    epochs = experiment_df["epoch"]


    # --------------------------------------------------------
    # 7-1. Train Loss + Validation Loss
    # --------------------------------------------------------

    loss_plot_path = (
        PLOT_RUNS_DIR / f"{experiment_id}_loss.png"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        experiment_df["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        experiment_df["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{experiment_labels[experiment_id]} Loss")
    plt.xticks(range(1, 21))
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(
        loss_plot_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


    # --------------------------------------------------------
    # 7-2. Train Accuracy + Validation Accuracy
    # --------------------------------------------------------

    accuracy_plot_path = (
        PLOT_RUNS_DIR / f"{experiment_id}_accuracy.png"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        experiment_df["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        experiment_df["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{experiment_labels[experiment_id]} Accuracy")
    plt.xticks(range(1, 21))
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(
        accuracy_plot_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 8. Validation Accuracy 비교 그래프
# ============================================================

validation_accuracy_comparison_path = (
    PLOT_COMPARISON_DIR
    / "validation_accuracy_comparison.png"
)

plt.figure(figsize=(9, 6))

for experiment_id in experiment_order:

    experiment_df = epoch_history_df[
        epoch_history_df["experiment_id"] == experiment_id
    ].sort_values("epoch")

    plt.plot(
        experiment_df["epoch"],
        experiment_df["val_accuracy"],
        label=experiment_labels[experiment_id]
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("MNIST Validation Accuracy Comparison")
plt.xticks(range(1, 21))
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    validation_accuracy_comparison_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 9. Test Accuracy 막대그래프
# ============================================================

# 그래프의 막대 순서를 experiment_order와 동일하게 정렬
ordered_summary_df = (
    run_summary_df
    .set_index("experiment_id")
    .loc[experiment_order]
    .reset_index()
)

bar_labels = [
    experiment_labels[experiment_id]
    for experiment_id in ordered_summary_df["experiment_id"]
]

test_accuracies = ordered_summary_df["test_accuracy"]


test_accuracy_comparison_path = (
    PLOT_COMPARISON_DIR
    / "test_accuracy_comparison.png"
)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    bar_labels,
    test_accuracies
)

plt.xlabel("Experiment")
plt.ylabel("Test Accuracy (%)")
plt.title("MNIST Test Accuracy Comparison")

# 막대그래프의 기준선을 0으로 설정
plt.ylim(0, 100)

plt.xticks(rotation=10)
plt.grid(axis="y")


# 각 막대 위에 정확한 accuracy 표시
for bar, accuracy in zip(bars, test_accuracies):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{accuracy:.2f}%",
        ha="center",
        va="bottom"
    )


plt.tight_layout()
plt.savefig(
    test_accuracy_comparison_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 10. 저장 결과 출력
# ============================================================

print("모든 그래프 저장 완료")
print()
print(f"개별 실험 그래프 폴더: {PLOT_RUNS_DIR}")
print(f"비교 그래프 폴더: {PLOT_COMPARISON_DIR}")
print()
print(
    "Validation Accuracy 비교 그래프:",
    validation_accuracy_comparison_path
)
print(
    "Test Accuracy 비교 그래프:",
    test_accuracy_comparison_path
)
