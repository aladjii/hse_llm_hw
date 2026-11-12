import argparse
import hashlib
import html
import json
import math
import textwrap
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent
RUNS = ["base", "grpo_exact", "grpo_shaped", "dr_grpo_shaped"]
NAMES = ["Исходная модель", "GRPO, точная", "GRPO, частичная", "Dr. GRPO, частичная"]
COLORS = ["#84929f", "#397ba5", "#148776", "#bc7148"]
GROUPS = {
    "simple": ["simple"],
    "multiple": ["multiple", "live_multiple"],
    "parallel": ["parallel"],
    "parallel_multiple": ["parallel_multiple"],
    "irrelevance": ["irrelevance", "live_irrelevance"],
}
EVALUATOR_SHA256 = "62411a6f1bd790bd54f27fb09e567075b5b27510d126ebe145e72b8319abc7b6"
BENCHMARK_SHA256 = "b257ffe2c539897123706ed2211f9ad0e68b8f8fbad7b523ba5a3fcca2a457b2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_result(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics, rows = (
        read_json(directory / "metrics.json"),
        read_rows(directory / "predictions.jsonl"),
    )
    benchmark = read_rows(ROOT / "bfcl.jsonl")
    if len(rows) != 350 or Counter(row["id"] for row in rows) != Counter(
        row["id"] for row in benchmark
    ):
        raise ValueError(
            f"в `{directory}` должны быть все 350 уникальных примеров BFCL"
        )
    categories = {row["id"]: row["category"] for row in benchmark}
    for row in rows:
        if (
            row["category"] != categories[row["id"]]
            or type(row["check"]["valid"]) is not bool
        ):
            raise ValueError(f"некорректная категория или оценка в `{directory}`")
    for group, names in GROUPS.items():
        accuracies = []
        for name in names:
            flags = [row["check"]["valid"] for row in rows if row["category"] == name]
            recorded = metrics["categories"][name]
            if recorded["correct"] != sum(flags) or recorded["total"] != len(flags):
                raise ValueError(f"не совпадают счётчики `{name}` в `{directory}`")
            if not math.isclose(recorded["accuracy"], mean(flags), abs_tol=1e-12):
                raise ValueError(f"не совпадает точность `{name}` в `{directory}`")
            accuracies.append(mean(flags))
        if not math.isclose(metrics["groups"][group], mean(accuracies), abs_tol=1e-12):
            raise ValueError(f"не совпадает оценка группы `{group}` в `{directory}`")
    if not math.isclose(
        metrics["overall"], mean(metrics["groups"].values()), abs_tol=1e-12
    ):
        raise ValueError(f"не совпадает `overall` в `{directory}`")
    if metrics["benchmark_sha256"] != BENCHMARK_SHA256:
        raise ValueError(f"изменён набор BFCL в `{directory}`")
    if metrics["decoding"] != {
        "temperature": 0,
        "max_new_tokens": 1024,
        "enable_thinking": False,
        "seed": 42,
    }:
        raise ValueError(f"изменены параметры итоговой генерации в `{directory}`")
    return metrics, rows


def check_artifacts() -> None:
    if (
        file_hash(ROOT / "evaluate.py") != EVALUATOR_SHA256
        or file_hash(ROOT / "bfcl.jsonl") != BENCHMARK_SHA256
    ):
        raise ValueError("изменены исходный оценщик или BFCL")
    for name in RUNS:
        validate_result(ROOT / "results" / name)
    validation_ids = []
    for name in RUNS[1:]:
        folder = ROOT / "runs" / name
        run, selection = (
            read_json(folder / "run.json"),
            read_json(folder / "selection.json"),
        )
        history = read_rows(folder / "history.jsonl")
        training = [row for row in history if "loss" in row]
        evaluation = [row for row in history if "eval_reward" in row]
        config = read_json(ROOT / "configs" / f"{name}.json")
        if run["status"] != "complete" or [row["step"] for row in training] != list(
            range(1, 301)
        ):
            raise ValueError(f"запуск `{name}` не завершил 300 шагов")
        if (
            config["training"] != run["config"]["training"]
            or config["lora"] != run["config"]["lora"]
            or config["reward"] != run["config"]["reward"]
        ):
            raise ValueError(f"конфигурация `{name}` отличается от использованной")
        if [row["step"] for row in evaluation] != [50, 100, 150, 200, 250, 300]:
            raise ValueError(f"неполная история проверки в `{name}`")
        selected = max(evaluation, key=lambda row: row["eval_reward"])
        if int(Path(selection["best_checkpoint"]).name.split("-")[-1]) != selected[
            "step"
        ] or not math.isclose(selection["best_metric"], selected["eval_reward"]):
            raise ValueError(f"не совпадает выбранный шаг в `{name}`")
        export = read_json(folder / "export.json")
        if not export["merged"] or export["adapter_config_sha256"] != file_hash(
            folder / "adapter_config.json"
        ):
            raise ValueError(f"не подтверждено объединение LoRA в `{name}`")
        for row in training:
            if not all(
                math.isfinite(row[key])
                for key in ["loss", "reward", "tool_exact", "grad_norm"]
            ):
                raise ValueError(f"нечисловая метрика в `{name}`")
        validation_ids.append(read_json(folder / "validation_ids.json"))
    if (
        len(validation_ids[0]) != 128
        or not validation_ids[0] == validation_ids[1] == validation_ids[2]
    ):
        raise ValueError(
            "в основных запусках должны совпадать 128 проверочных запросов"
        )


def make_plots() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#273745",
            "axes.titleweight": "medium",
            "grid.alpha": 0.2,
            "savefig.facecolor": "white",
        }
    )
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)

    def save(figure: Any, name: str) -> None:
        figure.savefig(folder / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

    def smooth(values: list[float]) -> list[float]:
        return [
            mean(values[max(0, index - 14) : index + 1]) for index in range(len(values))
        ]

    figure, axes = plt.subplots(
        3, 1, figsize=(8.8, 4.8), sharex=True, constrained_layout=True
    )
    reward_figure, reward_axes = plt.subplots(
        1, 3, figsize=(10, 3.3), sharey=True, constrained_layout=True
    )
    diagnostic, diagnostic_axes = plt.subplots(
        1, 2, figsize=(9.5, 3.1), constrained_layout=True
    )
    for index, name in enumerate(RUNS[1:], 1):
        history = read_rows(ROOT / "runs" / name / "history.jsonl")
        train = [row for row in history if "loss" in row]
        validation = [row for row in history if "eval_reward" in row]
        selected = int(
            Path(
                read_json(ROOT / "runs" / name / "selection.json")["best_checkpoint"]
            ).name.split("-")[-1]
        )
        x = [row["step"] for row in train]
        axis = axes[index - 1]
        axis.plot(
            x,
            [row["loss"] for row in train],
            color=COLORS[index],
            alpha=0.28,
            linewidth=0.8,
        )
        axis.plot(
            x,
            smooth([row["loss"] for row in train]),
            color=COLORS[index],
            linewidth=1.7,
        )
        axis.axvline(selected, color="#77828b", linestyle="--", linewidth=1)
        axis.set_title(NAMES[index], loc="left", fontsize=10)
        axis.set_ylabel("Потери")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axis.grid()
        axis = reward_axes[index - 1]
        axis.plot(
            x,
            [row["reward"] for row in train],
            color=COLORS[index],
            alpha=0.12,
            linewidth=0.7,
        )
        axis.plot(
            x,
            smooth([row["reward"] for row in train]),
            color=COLORS[index],
            label="Обучение, среднее за 15 шагов",
        )
        axis.plot(
            [row["step"] for row in validation],
            [row["eval_reward"] for row in validation],
            "o--",
            color="#253746",
            markersize=3.5,
            linewidth=1.1,
            label="Проверка",
        )
        axis.axvline(selected, color="#77828b", linestyle="--", linewidth=1)
        axis.set_title(NAMES[index].replace(", ", "\n"), fontsize=10)
        axis.set_xlabel("Шаг")
        axis.set_ylim(0, 1.03)
        axis.grid()
        for diagnostic_axis, key in zip(
            diagnostic_axes,
            ["eval_tool_exact", "eval_frac_reward_zero_std"],
            strict=True,
        ):
            diagnostic_axis.plot(
                [row["step"] for row in validation],
                [100 * row[key] for row in validation],
                "o-",
                color=COLORS[index],
                markersize=3.5,
                label=NAMES[index],
            )
    axes[-1].set_xlabel("Шаг оптимизатора")
    reward_axes[0].set_ylabel("Награда")
    handles, labels = reward_axes[0].get_legend_handles_labels()
    reward_figure.legend(
        handles, labels, loc="outside lower center", ncols=2, frameon=False, fontsize=8
    )
    for axis, title in zip(
        diagnostic_axes,
        ["Точное совпадение на проверке", "Группы с одинаковой наградой"],
        strict=True,
    ):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Шаг")
        axis.set_ylabel("Доля, %")
        axis.grid()
    handles, labels = diagnostic_axes[0].get_legend_handles_labels()
    diagnostic.legend(
        handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=8
    )
    save(figure, "loss")
    save(reward_figure, "reward")
    save(diagnostic, "validation")


def make_pdf() -> None:
    from markdown_it import MarkdownIt
    from matplotlib import get_data_path
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    fonts = Path(get_data_path()) / "fonts/ttf"
    for name, filename in [
        ("Body", "DejaVuSans.ttf"),
        ("BodyBold", "DejaVuSans-Bold.ttf"),
        ("Mono", "DejaVuSansMono.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold"
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TextRU",
            fontName="Body",
            fontSize=9.7,
            leading=13.3,
            spaceAfter=5,
            textColor=colors.HexColor("#23323f"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitleRU",
            parent=styles["TextRU"],
            fontName="BodyBold",
            fontSize=23,
            leading=28,
            spaceAfter=14,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2RU",
            parent=styles["TextRU"],
            fontName="BodyBold",
            fontSize=15,
            leading=19,
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3RU",
            parent=styles["TextRU"],
            fontName="BodyBold",
            fontSize=11,
            leading=15,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CaptionRU",
            parent=styles["TextRU"],
            fontSize=8.4,
            leading=11,
            textColor=colors.HexColor("#53636e"),
            spaceAfter=11,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CellRU",
            parent=styles["TextRU"],
            fontSize=8.1,
            leading=10.5,
            spaceAfter=0,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeRU",
            fontName="Mono",
            fontSize=7.4,
            leading=10.2,
            backColor=colors.HexColor("#f0f4f6"),
            borderPadding=7,
            spaceBefore=4,
            spaceAfter=10,
        )
    )
    width = A4[0] - 38 * mm

    def inline(token: Any) -> str:
        output = []
        for child in token.children or []:
            text = html.escape(child.content)
            output.append(
                {
                    "strong_open": "<b>",
                    "strong_close": "</b>",
                    "em_open": "<i>",
                    "em_close": "</i>",
                    "softbreak": " ",
                    "hardbreak": "<br/>",
                    "link_open": "",
                    "link_close": "",
                }.get(
                    child.type,
                    f'<font name="Mono" size="8">{text}</font>'
                    if child.type == "code_inline"
                    else text,
                )
            )
        return "".join(output)

    tokens = MarkdownIt().enable("table").parse((ROOT / "README.md").read_text())
    story, index = [], 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == "heading_open":
            style = {"h1": "TitleRU", "h2": "H2RU", "h3": "H3RU"}.get(token.tag, "H3RU")
            story.append(Paragraph(inline(tokens[index + 1]), styles[style]))
            index += 3
            continue
        if token.type == "table_open":
            rows, row = [], []
            index += 1
            while tokens[index].type != "table_close":
                current = tokens[index]
                if current.type == "tr_open":
                    row = []
                elif current.type == "inline":
                    row.append(Paragraph(inline(current), styles["CellRU"]))
                elif current.type == "tr_close":
                    rows.append(row)
                index += 1
            columns = len(rows[0])
            if columns == 7:
                widths = [width * 0.26] + [width * 0.74 / 6] * 6
            elif columns == 2:
                widths = [width * 0.34, width * 0.66]
            else:
                widths = [width / columns] * columns
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce8ed")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f3f6f8")],
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#a0bac5")),
                    ]
                )
            )
            story.extend([table, Spacer(1, 9)])
        elif token.type == "paragraph_open":
            content = tokens[index + 1]
            pictures = [
                child for child in content.children or [] if child.type == "image"
            ]
            if pictures:
                picture = pictures[0]
                path = ROOT / picture.attrGet("src")
                from PIL import Image as PillowImage

                with PillowImage.open(path) as image:
                    ratio = image.height / image.width
                image_width = min(width, 190 * mm / ratio)
                block = [
                    Image(str(path), width=image_width, height=image_width * ratio)
                ]
                if (
                    index + 4 < len(tokens)
                    and tokens[index + 3].type == "paragraph_open"
                ):
                    block.extend(
                        [
                            Spacer(1, 3),
                            Paragraph(inline(tokens[index + 4]), styles["CaptionRU"]),
                        ]
                    )
                    index += 3
                story.append(KeepTogether(block))
            else:
                follows_image = index >= 2 and any(
                    child.type == "image" for child in tokens[index - 2].children or []
                )
                style = "CaptionRU" if follows_image else "TextRU"
                story.append(Paragraph(inline(content), styles[style]))
            index += 3
            continue
        elif token.type == "fence":
            wrapped = []
            for line in token.content.rstrip().splitlines():
                wrapped.extend(
                    textwrap.wrap(
                        line,
                        width=101,
                        subsequent_indent="    ",
                        break_long_words=False,
                        break_on_hyphens=False,
                        replace_whitespace=False,
                    )
                    or [""]
                )
            story.append(Preformatted("\n".join(wrapped), styles["CodeRU"]))
        elif token.type == "html_block" and "pagebreak" in token.content:
            story.append(PageBreak())
        index += 1

    def decorate(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setFont("Body", 8)
        canvas.setFillColor(colors.HexColor("#667887"))
        canvas.drawString(19 * mm, A4[1] - 12 * mm, "ДЗ-5  |  Обучение вызову функций")
        canvas.drawRightString(A4[0] - 19 * mm, 11 * mm, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(ROOT / "REPORT.pdf"),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=21 * mm,
        bottomMargin=19 * mm,
        title="Обучение Qwen3.5-2B вызову функций",
        author="Отчёт по ДЗ-5",
    )
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)


def render_pdf(directory: Path) -> None:
    import pypdfium2 as pdfium

    directory.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(ROOT / "REPORT.pdf")
    for index, page in enumerate(document):
        page.render(scale=1.6).to_pil().save(directory / f"page-{index + 1:02d}.png")
    print(f"страницы для проверки: {directory}, всего {len(document)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверка результатов, графики и PDF-отчёт без запуска модели"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="только сверить результаты и выбранные шаги",
    )
    parser.add_argument(
        "--render-dir", type=Path, help="сохранить страницы PDF в PNG для просмотра"
    )
    args = parser.parse_args()
    check_artifacts()
    print("проверены 1400 предсказаний и три полных RL-запуска")
    if args.check:
        return
    make_plots()
    make_pdf()
    if args.render_dir:
        render_pdf(args.render_dir)
    print(ROOT / "REPORT.pdf")


if __name__ == "__main__":
    main()
