# Reproduce best.yaml

Канонический воспроизводимый запуск best run (`20260427_171719_hybrid_weighted_score_minmax`).

## Окружение

- Python 3.10.11
- см. requirements.txt + manifest of best run.

## Шаги

1. `python -m src.cli evaluate --config configs/best.yaml`
2. Метрики в `reports/runs/<run_id>/metrics.json`.

## Программная загрузка модели

```python
from src.models.hybrid import HybridModel

model = HybridModel.load("artifacts/hybrid/20260427_171719_hybrid_weighted_score_minmax")
ids, scores, lats = model.predict_topk(["обращение текст..."], k=10)
```

## AUD-20260427-02 (resolved, 2026-04-28, TASK-011)

`configs/best.yaml` — канонический пост-hoc конфиг winner-runа,
зафиксирован DECISION-2026-04-27-029. Поле
`metrics.json.config_path` лучшего grid runа указывает на
runtime grid YAML (`configs/_grid/hybrid_30_minmax.yaml`) —
это исторический артефакт grid-search, сохраняемый для
воспроизводимости конкретного запуска. Для production reload
используется `HybridModel.load(artifacts_dir)`, которая читает
`fusion_config.json` из `artifacts/`, а не runtime config.
