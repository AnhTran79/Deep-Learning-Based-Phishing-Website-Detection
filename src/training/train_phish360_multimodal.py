from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.multimodal_dataset_loader import MultimodalRecord, load_multimodal_records
from src.evaluation.metrics import evaluate_binary_classification, save_metric_figures, write_result_row
from src.models.dual_branch_cnn import DualBranchCnnClassifier
from src.models.html_cnn import HtmlCnnClassifier
from src.models.screenshot_cnn import ScreenshotCnnClassifier
from src.models.tri_branch_cnn import TriBranchCnnClassifier
from src.models.url_cnn import UrlCnnClassifier
from src.models.url_lstm import UrlLstmClassifier
from src.preprocessing.image_transforms import load_image_tensor
from src.preprocessing.text_cleaning import html_to_visible_text, normalize_url_text
from src.preprocessing.tokenizers import CharTokenizerConfig, encode_char_sequence


PHISH360_MODELS = [
    "url_cnn",
    "url_lstm",
    "html_cnn",
    "screenshot_cnn",
    "dual_branch_cnn",
    "tri_branch_cnn",
]


class Phish360TorchDataset(Dataset):
    def __init__(
        self,
        records: list[MultimodalRecord],
        model_name: str,
        url_config: CharTokenizerConfig,
        html_config: CharTokenizerConfig,
        image_size: int,
    ) -> None:
        self.records = records
        self.model_name = model_name
        self.url_config = url_config
        self.html_config = html_config
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        label = torch.tensor(record.label, dtype=torch.float32)
        url_ids = torch.tensor(encode_char_sequence(normalize_url_text(record.url), self.url_config), dtype=torch.long)
        html_ids = torch.tensor(
            encode_char_sequence(html_to_visible_text(record.html), self.html_config),
            dtype=torch.long,
        )
        if self.model_name in {"url_cnn", "url_lstm"}:
            return url_ids, label
        if self.model_name == "html_cnn":
            return html_ids, label

        image = load_image_tensor(record.screenshot_file, image_size=self.image_size)
        if self.model_name == "screenshot_cnn":
            return image, label
        if self.model_name == "dual_branch_cnn":
            return url_ids, html_ids, label
        return url_ids, html_ids, image, label


def split_train_val(
    records: list[MultimodalRecord],
    val_size: float,
    random_state: int,
) -> tuple[list[MultimodalRecord], list[MultimodalRecord]]:
    labels = [record.label for record in records]
    train_indices, val_indices = train_test_split(
        list(range(len(records))),
        test_size=val_size,
        stratify=labels,
        random_state=random_state,
    )
    return [records[index] for index in train_indices], [records[index] for index in val_indices]


def limit_records_balanced(
    records: list[MultimodalRecord],
    max_rows: int,
    random_state: int,
) -> list[MultimodalRecord]:
    if max_rows <= 0 or len(records) <= max_rows:
        return records
    labels = [record.label for record in records]
    selected_indices, _ = train_test_split(
        list(range(len(records))),
        train_size=max_rows,
        stratify=labels,
        random_state=random_state,
    )
    selected_indices.sort()
    return [records[index] for index in selected_indices]


def build_model(
    model_name: str,
    url_vocab_size: int,
    html_vocab_size: int,
    embedding_dim: int,
    dropout_rate: float,
    image_feature_dim: int,
):
    if model_name == "url_cnn":
        return UrlCnnClassifier(vocab_size=url_vocab_size, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    if model_name == "url_lstm":
        return UrlLstmClassifier(vocab_size=url_vocab_size, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    if model_name == "html_cnn":
        return HtmlCnnClassifier(vocab_size=html_vocab_size, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    if model_name == "screenshot_cnn":
        return ScreenshotCnnClassifier(dropout_rate=dropout_rate, feature_dim=image_feature_dim)
    if model_name == "dual_branch_cnn":
        return DualBranchCnnClassifier(
            url_vocab_size=url_vocab_size,
            html_vocab_size=html_vocab_size,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )
    if model_name == "tri_branch_cnn":
        return TriBranchCnnClassifier(
            url_vocab_size=url_vocab_size,
            html_vocab_size=html_vocab_size,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            image_feature_dim=image_feature_dim,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _forward(model, batch, model_name: str, device: torch.device):
    if model_name in {"url_cnn", "url_lstm", "html_cnn", "screenshot_cnn"}:
        inputs, labels = batch
        return model(inputs.to(device)), labels.to(device)
    if model_name == "dual_branch_cnn":
        url_ids, html_ids, labels = batch
        return model(url_ids.to(device), html_ids.to(device)), labels.to(device)
    url_ids, html_ids, images, labels = batch
    return model(url_ids.to(device), html_ids.to(device), images.to(device)), labels.to(device)


def evaluate_model(model, loader: DataLoader, model_name: str, device: torch.device, threshold: float) -> tuple[dict, list[int], list[float]]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits, batch_labels = _forward(model, batch, model_name, device)
            probabilities.extend(float(value) for value in torch.sigmoid(logits).cpu().tolist())
            labels.extend(int(value) for value in batch_labels.cpu().tolist())
    return evaluate_binary_classification(labels, probabilities, threshold=threshold), labels, probabilities


def train_one_model(
    model_name: str,
    train: list[MultimodalRecord],
    val: list[MultimodalRecord],
    test: list[MultimodalRecord],
    args: argparse.Namespace,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    url_config = CharTokenizerConfig(max_len=args.max_url_len, vocab_size=args.url_vocab_size)
    html_config = CharTokenizerConfig(max_len=args.max_html_len, vocab_size=args.html_vocab_size)
    train_loader = DataLoader(
        Phish360TorchDataset(train, model_name, url_config, html_config, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Phish360TorchDataset(val, model_name, url_config, html_config, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        Phish360TorchDataset(test, model_name, url_config, html_config, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = build_model(
        model_name,
        args.url_vocab_size,
        args.html_vocab_size,
        args.embedding_dim,
        args.dropout_rate,
        args.image_feature_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_val_f1 = -1.0
    patience_remaining = args.early_stopping_patience

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, labels = _forward(model, batch, model_name, device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        val_metrics, _, _ = evaluate_model(model, val_loader, model_name, device, args.threshold)
        print(f"phish360_{model_name} epoch={epoch + 1} loss={total_loss:.4f} val_f1={val_metrics['f1']:.4f}")
        if float(val_metrics["f1"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_state = deepcopy(model.state_dict())
            patience_remaining = args.early_stopping_patience
        else:
            patience_remaining -= 1
            if patience_remaining <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    output_name = f"phish360_{model_name}"
    test_metrics, test_labels, test_probabilities = evaluate_model(model, test_loader, model_name, device, args.threshold)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = out_dir / f"{output_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_name,
            "url_tokenizer": url_config.__dict__,
            "html_tokenizer": html_config.__dict__,
            "model_config": {
                "embedding_dim": args.embedding_dim,
                "dropout_rate": args.dropout_rate,
                "url_vocab_size": args.url_vocab_size,
                "html_vocab_size": args.html_vocab_size,
                "image_size": args.image_size,
                "image_feature_dim": args.image_feature_dim,
            },
            "threshold": args.threshold,
            "source": "phish360",
        },
        model_file,
    )
    save_metric_figures(output_name, test_labels, test_probabilities, Path(args.figures_dir), threshold=args.threshold)
    row = {
        "model": output_name,
        "input": {
            "url_cnn": "URL",
            "url_lstm": "URL",
            "html_cnn": "HTML",
            "screenshot_cnn": "Screenshot",
            "dual_branch_cnn": "URL + HTML",
            "tri_branch_cnn": "URL + HTML + Screenshot",
        }[model_name],
        **test_metrics,
        "model_file": str(model_file),
    }
    write_result_row(Path(args.results_dir) / "phish360_model_comparison.csv", row)
    (out_dir / f"{output_name}_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"{output_name} test_f1={test_metrics['f1']:.4f} recall={test_metrics['recall']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phish360 URL + HTML + screenshot models.")
    parser.add_argument("--data", default="data/processed/phish360_url_html_screenshot.csv")
    parser.add_argument(
        "--model",
        default="all",
        choices=[*PHISH360_MODELS, "all"],
    )
    parser.add_argument("--out-dir", default="models/saved/phish360")
    parser.add_argument("--results-dir", default="reports/results/phish360")
    parser.add_argument("--figures-dir", default="reports/figures/phish360")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--html-max-chars", type=int, default=2000)
    parser.add_argument("--max-url-len", type=int, default=200)
    parser.add_argument("--max-html-len", type=int, default=2000)
    parser.add_argument("--url-vocab-size", type=int, default=128)
    parser.add_argument("--html-vocab-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--image-feature-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainval_records = load_multimodal_records(
        Path(args.data),
        split="trainval",
        max_rows=0,
        html_max_chars=args.html_max_chars,
    )
    test_records = load_multimodal_records(
        Path(args.data),
        split="test",
        max_rows=0,
        html_max_chars=args.html_max_chars,
    )
    trainval_records = limit_records_balanced(trainval_records, args.max_rows, args.random_state)
    test_records = limit_records_balanced(test_records, args.max_rows, args.random_state)
    train_records, val_records = split_train_val(trainval_records, args.val_size, args.random_state)
    models = PHISH360_MODELS if args.model == "all" else [args.model]
    print(f"Phish360 train={len(train_records)} val={len(val_records)} test={len(test_records)}")
    for model_name in models:
        train_one_model(model_name, train_records, val_records, test_records, args)


if __name__ == "__main__":
    main()
