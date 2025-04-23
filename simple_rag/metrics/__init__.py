from collections import defaultdict
from typing import Tuple

# metrics.py
from sqlalchemy.orm import Session
from .kbase_metric import KBaseMetric

def increment_metric(db: Session, endpoint: str, model_name: str):
    """Increase count for the specified endpoint and model_name."""
    metric = db.query(KBaseMetric).filter_by(endpoint=endpoint, model_name=model_name).first()
    if metric:
        metric.count += 1
    else:
        metric = KBaseMetric(endpoint=endpoint, model_name=model_name, count=1)
        db.add(metric)
    db.commit()

def get_metrics(db: Session) -> Tuple[dict[str, dict[str, int]], int]:
    """Get all metrics from the database."""
    metrics = db.query(KBaseMetric).all()
    report = defaultdict(dict)

    for metric in metrics:
        report[metric.endpoint][metric.model_name] = metric.count

    totals = sum(metric.count for metric in metrics)
    return report, totals