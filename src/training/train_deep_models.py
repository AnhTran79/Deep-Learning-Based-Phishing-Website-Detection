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
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset_loader import UrlHtmlRecord, load_url_html_records
from src.data.splitting import split_train_val_test as stratified_split_train_val_test
from src.evaluation.metrics import evaluate_binary_classification, save_metric_figures, write_result_row
from src.models.dual_branch_cnn import DualBranchCnnClassifier
from src.models.html_cnn import HtmlCnnClassifier
from src.models.url_cnn import UrlCnnClassifier
from src.models.url_lstm import UrlLstmClassifier
from src.preprocessing.text_cleaning import html_to_visible_text, normalize_url_text
from src.preprocessing.tokenizers import CharTokenizerConfig, encode_char_sequence


class UrlHtmlTorchDataset(Dataset):
    def __init__(
        self,
        records: list[UrlHtmlRecord],
        model_name: str,
        url_config: CharTokenizerConfig,
        html_config: CharTokenizerConfig,
    ) -> None:
        self.records = records
        self.model_name = model_name
        self.url_config = url_config
        self.html_config = html_config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        url_ids = torch.tensor(encode_char_sequence(normalize_url_text(record.url), self.url_config), dtype=torch.long)
        html_text = html_to_visible_text(record.html)
        html_ids = torch.tensor(encode_char_sequence(html_text, self.html_config), dtype=torch.long)
        label = torch.tensor(record.label, dtype=torch.float32)
        if self.model_name in {"url_cnn", "url_lstm"}:
            return url_ids, label
        if self.model_name == "html_cnn":
            return html_ids, label
        return url_ids, html_ids, label


def split_train_val_test(
    records: list[UrlHtmlRecord],
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[list[UrlHtmlRecord], list[UrlHtmlRecord], list[UrlHtmlRecord]]:
    labels = [record.label for record in records]
    return stratified_split_train_val_test(records, labels, test_size, val_size, random_state)


def build_model(model_name: str, url_vocab_size: int, html_vocab_size: int, embedding_dim: int, dropout_rate: float):
    if model_name == "url_cnn":
        return UrlCnnClassifier(vocab_size=url_vocab_size, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    if model_name == "url_lstm":
        return UrlLstmClassifier(vocab_size=url_vocab_size, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    if model_name == "html_cnn":
        return HtmlCnnClassifier(vocab_size=html_vocab_size, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    if model_name == "dual_branch_cnn":
        return DualBranchCnnClassifier(
            url_vocab_size=url_vocab_size,
            html_vocab_size=html_vocab_size,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _forward(model, batch, model_name: str, device: torch.device):
    if model_name in {"url_cnn", "url_lstm", "html_cnn"}:
        inputs, labels = batch
        return model(inputs.to(device)), labels.to(device)
    url_ids, html_ids, labels = batch
    return model(url_ids.to(device), html_ids.to(device)), labels.to(device)


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
    train: list[UrlHtmlRecord],
    val: list[UrlHtmlRecord],
    test: list[UrlHtmlRecord],
    args: argparse.Namespace,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    url_config = CharTokenizerConfig(max_len=args.max_url_len, vocab_size=args.url_vocab_size)
    html_config = CharTokenizerConfig(max_len=args.max_html_len, vocab_size=args.html_vocab_size)
    train_loader = DataLoader(
        UrlHtmlTorchDataset(train, model_name, url_config, html_config),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        UrlHtmlTorchDataset(val, model_name, url_config, html_config),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        UrlHtmlTorchDataset(test, model_name, url_config, html_config),
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = build_model(model_name, args.url_vocab_size, args.html_vocab_size, args.embedding_dim, args.dropout_rate).to(device)
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
        print(f"{model_name} epoch={epoch + 1} loss={total_loss:.4f} val_f1={val_metrics['f1']:.4f}")
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
    test_metrics, test_labels, test_probabilities = evaluate_model(model, test_loader, model_name, device, args.threshold)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = out_dir / f"{model_name}.pt"
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
            },
            "threshold": args.threshold,
        },
        model_file,
    )
    save_metric_figures(model_name, test_labels, test_probabilities, Path(args.figures_dir), threshold=args.threshold)
    row = {
        "model": model_name,
        "input": {
            "url_cnn": "URL",
            "url_lstm": "URL",
            "html_cnn": "HTML",
            "dual_branch_cnn": "URL + HTML",
        }[model_name],
        **test_metrics,
        "model_file": str(model_file),
    }
    write_result_row(Path(args.results_dir) / "model_comparison.csv", row)
    (out_dir / f"{model_name}_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"{model_name} test_f1={test_metrics['f1']:.4f} recall={test_metrics['recall']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyTorch deep models for URL + HTML phishing detection.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--html-root", default="")
    parser.add_argument("--model", default="dual_branch_cnn", choices=["url_cnn", "url_lstm", "html_cnn", "dual_branch_cnn", "all"])
    parser.add_argument("--out-dir", default="models/saved")
    parser.add_argument("--results-dir", default="reports/results")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--html-max-chars", type=int, default=2000)
    parser.add_argument("--max-url-len", type=int, default=200)
    parser.add_argument("--max-html-len", type=int, default=2000)
    parser.add_argument("--url-vocab-size", type=int, default=128)
    parser.add_argument("--html-vocab-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_root = Path(args.html_root) if args.html_root else None
    records = load_url_html_records(
        Path(args.data),
        html_root=html_root,
        max_rows=args.max_rows,
        html_max_chars=args.html_max_chars,
    )
    train, val, test = split_train_val_test(records, args.test_size, args.val_size, args.random_state)
    models = ["url_cnn", "url_lstm", "html_cnn", "dual_branch_cnn"] if args.model == "all" else [args.model]
    for model_name in models:
        train_one_model(model_name, train, val, test, args)


if __name__ == "__main__":
    main()
