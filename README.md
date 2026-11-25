# Домашние задания по LLM

Репозиторий с домашними заданиями по курсу про большие языковые модели.

- `hw1/`: предобучение языковой модели (960M параметров) с нуля на ru Wikipedia. Сравнение 10 конфигураций обучения (scheduler, LR, microbatch, torch.compile) в бюджете 15 минут на A100. Код в `your_solution.py`, эксперименты и выводы в `experiment.ipynb`.
- `hw2/`: ускорение обучения той же модели на одной GPU и кольцевые коммуникации. Исследовано 8 механизмов (батч, checkpointing, FA2, liger, torch.compile, packing, padding-free, offloading); итог: packing + FlashAttention-2 + torch.compile, **S = 5.2×** по полезным токенам/с (75036 против 14369 у бейзлайна за 300 с). Отчёт: `hw2/REPORT.md`, эксперименты в `hw2/results/`, кольцевые reduce-scatter/all-gather/all-reduce на gloo в `hw2/collectives.py` + `communications.ipynb`.
- `hw5/`: RL-обучение Qwen3.5-2B вызову функций на ToolACE. Три конфигурации; лучший результат BFCL 75,33% против исходных 68,93%. Отчёт и команды воспроизведения в `hw5/README.md`, PDF в `hw5/REPORT.pdf`.
- `hw6/` — бенчмарк инференса Qwen3-8B на vLLM и H200. При 120 запросах/мин n-gram-декодирование снизило среднюю задержку на 5,3% и увеличило P90 на 3,1%. Максимальная проверенная нагрузка при P90 < 15 с и без ошибок составила 810 запросов/мин. [Отчёт и команды воспроизведения](hw6/REPORT.md).
- `hw7/`: квантизация и дистилляция Qwen3 на MASSIVE ru-RU. [Ноутбук](hw7/homework_compression_student.ipynb) и [отчёт](hw7/REPORT.md).
