from collections import Counter
import numpy as np


def compute_quality_metrics(id_entrance):
    counters, trk_lens, pred_matched = [], [], []
    gt_trk_ids = []

    print(f'id_entrance: {id_entrance}')

    for gt_trk, pred_trk_ids in id_entrance.items():
        cnt = Counter(pred_trk_ids)
        counters.append(cnt)
        trk_lens.append(len(pred_trk_ids))
        pred_matched.append(cnt.most_common(1)[0][0])
        gt_trk_ids.append(gt_trk)

    diff_match_val = len(set(pred_matched)) / len(gt_trk_ids)
    loss_trk = np.array([
        cnt[None] / trk_len for cnt, trk_len in zip(counters, trk_lens)
    ]).mean()

    recall = np.array([
        cnt[pred_match] / trk_len for cnt, pred_match, trk_len in zip(counters, pred_matched, trk_lens)
    ]).mean()

    all_preds = []
    for pred_trk_ids in id_entrance.values():
        all_preds.extend(pred_trk_ids)
    overall_cnt = Counter(all_preds)

    print(f'overall_cnt: {overall_cnt}')
    print(f'pred_matched: {pred_matched}')

    precision = np.array([
        cnt[pred_match] / overall_cnt[pred_match] for cnt, pred_match in zip(counters, pred_matched)
    ]).mean()

    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    ret_metrics = {
        "diff_match_val": diff_match_val,
        "recall": recall,
        "precision": precision,
        "f1-score": f1,
        "none_trk_freq": loss_trk, 
    }

    return ret_metrics


def print_quality_metrics(id_entrance):
    metrics_info = compute_quality_metrics(id_entrance)

    print("\nMetrics:\n")

    for metric_label, val in metrics_info.items():
        print(f'{metric_label}: {val:.2%}')

    print()
