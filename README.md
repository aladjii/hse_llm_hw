# Домашние задания по LLM

Репозиторий с домашними заданиями по курсу про большие языковые модели.

- `hw1/`: предобучение языковой модели (960M параметров) с нуля на ru Wikipedia. Сравнение 10 конфигураций обучения (scheduler, LR, microbatch, torch.compile) в бюджете 15 минут на A100. Код в `your_solution.py`, эксперименты и выводы в `experiment.ipynb`.
- `hw2/`: ускорение обучения той же модели на одной GPU и кольцевые коммуникации. Исследовано 8 механизмов (батч, checkpointing, FA2, liger, torch.compile, packing, padding-free, offloading); итог: packing + FlashAttention-2 + torch.compile, **S = 5.2×** по полезным токенам/с (75036 против 14369 у бейзлайна за 300 с). Отчёт: `hw2/REPORT.md`, эксперименты в `hw2/results/`, кольцевые reduce-scatter/all-gather/all-reduce на gloo в `hw2/collectives.py` + `communications.ipynb`.
