# Настройка репозитория

0. Создать виртуальное окружения
`python3.11 -m venv venv`
`source venv/bin/activate`

1. Установить зависимости
`pip install --upgrade pip setuptools wheel`
`pip install .`

2. Установка pygraphviz
```bash
  pip install pygraphviz \
  --no-cache-dir \
  --config-settings="--global-option=build_ext" \
  --config-settings="--global-option=-I/opt/homebrew/opt/graphviz/include/" \
  --config-settings="--global-option=-L/opt/homebrew/opt/graphviz/lib/"
```


# Запуск тестов
`pytest -v`

## Сценарии

0. Создание сетки из файла (модуль mispy_extract_mesh)
1. Отрисовка сетки из файла (модуль mispy_mesh_plottly)
2. Вывод статистики сетки из файла (модуль mispy_mesh_statistics)

# Сборка
`python3.11 -m build`

# Локальная отладка
При отладке модулей локально необходимо подгрузить в editable-режиме:
`pip install -e .`
Запуск .py файлов выполняется командой:
`PYTHONPATH=src python3 -m mispy.<имя модуля>.<имя функции>`, mispy - имя пакета
