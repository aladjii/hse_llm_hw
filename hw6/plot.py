import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run import SLA_S, valid, within_sla


def read_results(root: Path) -> list[dict]:
    timeline = {"tune": 0, "spec": 1, "load": 2}
    rows = [
        json.loads(path.read_text())
        for path in (root / "metrics").glob("*/*/result.json")
    ]
    return sorted(rows, key=lambda row: (timeline[row["phase"]], row["rpm"], row["config"]))


def table(rows: list[dict], output: Path) -> None:
    baseline = next(
        (row["mean_s"] for row in rows if row["phase"] == "tune"
         and row["config"] == "default" and valid(row)),
        None,
    )
    full = []
    for row in rows:
        fields = dict(row)
        if row["phase"] == "tune" and baseline is not None and valid(row):
            fields["speedup"] = baseline / row["mean_s"]
        if row["phase"] == "load":
            fields["sla_pass"] = within_sla(row)
        full.append(fields)
    columns = list(dict.fromkeys(
        ["phase", "config", "rpm", "gpu"] + [key for row in full for key in row]
    ))
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(full)


def speculative(rows: list[dict], output: Path) -> None:
    points = [row for row in rows if row["phase"] == "spec"]
    if not points:
        return
    candidate = next(row["config"] for row in points if row["config"] != "default")
    points += [
        row for row in rows if row["phase"] == "tune"
        and row["config"] in ("default", candidate)
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), layout="constrained")
    for config, legend in [
        ("default", "Без спекдека"),
        (candidate, f"Со спекдеком ({candidate})"),
    ]:
        series = sorted(
            (row for row in points if row["config"] == config),
            key=lambda row: row["rpm"],
        )
        ax.plot(
            [row["rpm"] for row in series],
            [row["p90_s"] if valid(row) else None for row in series],
            marker="o", label=legend,
        )
    ax.set(
        title="Спекулятивное декодирование: Qwen3-8B на H200",
        xlabel="Нагрузка, запросов/мин",
        ylabel="P90 E2E latency, с",
        xticks=sorted({row["rpm"] for row in points}),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def load(rows: list[dict], output: Path) -> None:
    points = [
        row for row in rows if row["phase"] == "load" and row.get("p90_s") is not None
    ]
    if not points:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.2), layout="constrained")
    ax.plot(
        [row["rpm"] for row in points], [row["p90_s"] for row in points],
        marker="o", color="tab:blue", label="Замеры по 300 с",
    )
    errors = [row for row in points if row["errors"]]
    if errors:
        ax.scatter(
            [row["rpm"] for row in errors], [row["p90_s"] for row in errors],
            marker="x", color="0.25", zorder=4, label="Были ошибки",
        )
    ax.axhline(SLA_S, color="0.45", linestyle="--", label=f"SLA: P90 < {SLA_S} с")
    passed = max((row for row in points if within_sla(row)), key=lambda r: r["rpm"], default=None)
    failed = next(
        (row for row in points if not within_sla(row)
         and (passed is None or row["rpm"] > passed["rpm"])),
        None,
    )
    for row, color, label, offset in (
        (passed, "tab:green", "Последний проход", (-18, -17)),
        (failed, "tab:red", "Первая точка выше предела", (3, 14)),
    ):
        if row is not None:
            ax.scatter(row["rpm"], row["p90_s"], color=color, zorder=5,
                       label=f"{label}: {row['rpm']} RPM")
            ax.annotate(
                f"{row['p90_s']:.2f} с",
                (row["rpm"], row["p90_s"]), xytext=offset,
                textcoords="offset points", color=color,
            )
    ax.set(
        title=f"Предельная нагрузка: {points[0]['config']}, одна H200",
        xlabel="Нагрузка, запросов/мин",
        ylabel="P90 E2E latency, с",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    rows = read_results(root)
    if not rows:
        parser.error("нет замеров в каталоге `metrics`")
    output = root / "results"
    output.mkdir()
    table(rows, output / "measurements.csv")
    speculative(rows, output / "speculative.png")
    load(rows, output / "load.png")


if __name__ == "__main__":
    main()
