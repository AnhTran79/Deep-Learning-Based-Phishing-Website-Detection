# Phish360 Model Comparison

The project trains six supervised deep-learning models on the same Phish360
split so each input modality can be compared fairly.

| Model | Input | Role |
|---|---|---|
| `phish360_url_cnn` | URL | URL ablation |
| `phish360_url_lstm` | URL | Sequence baseline |
| `phish360_html_cnn` | HTML | HTML ablation |
| `phish360_screenshot_cnn` | Screenshot | Visual ablation |
| `phish360_dual_branch_cnn` | URL + HTML | Fusion baseline |
| `phish360_tri_branch_cnn` | URL + HTML + Screenshot | Main model |

Results:

```text
reports/results/phish360/phish360_model_comparison.csv
reports/figures/phish360/phish360_model_comparison_metrics.png
```
