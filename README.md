# Настройка репозитория

0. Создать виртуальное окружения
`python3.11 -m venv venv`
`source venv/bin/activate`

1. Установить зависимости
`pip install --upgrade pip setuptools wheel`
`pip install .`

# Запуск тестов
`pytest -v`

## Сценарии

0. Создание сетки из файла (модуль mispy_extract_mesh) - проверка корректности создания сетки

# Сборка
`python3.11 -m build`

# Локальная отладка
При отладке модулей локально необходимо подгрузить в editable-режиме:
`pip install -e .`
Запуск .py файлов выполняется командой:
`PYTHONPATH=src python3 -m mispy.<имя модуля>.<имя функции>`, mispy - имя пакета
