# Black-box Optimization Library

Библиотека для оптимизации функций нулевого порядка, полностью совместимая с бенчмарком WIND.

## Реализованные алгоритмы
- Нелдер-Мид, SPSA, CMA-ES, ZO-SGD и др. (см. `zero_order_algorithns.py`).

## Интеграция с WIND
Реализован паттерн "Адаптер": все алгоритмы наследуются от `BaseOptimizer` и принимают `Observation` объект.

## Запуск экспериментов
1. **Тесты:** `pytest test_algorithms.py`
2. **Эксперименты (тепловая карта):** `python run_lyapunov_experiment.py`

## Команды для запуска
1. **Запуск тестов** `pytest --cov=. --cov-report=term-missing test_algorithms.py`
2. **Построение тепловой карты** `python3 run_and_plot_lyapunov.py`
