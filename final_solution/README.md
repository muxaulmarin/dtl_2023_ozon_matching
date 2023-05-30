# Перед запуском
Установить poetry (https://python-poetry.org/docs/#installation)
```bash
curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.2.2 python3 -
```

Склонировать репозиторий
```bash
git clone https://github.com/muxaulmarin/dtl_2023_ozon_matching.git
cd dtl_2023_ozon_matching
```

Установить зависимости и активировать окружение poetry
```bash
poetry install
poetry shell
```

Положить исходные данные в корень проекта в директорию `data/raw`
```bash
mkdir data
mkdir data/raw
cp "YOUR PATH HERE" data/raw/
```

# Инференс
## Предобработка
```bash
python -m ozon_matching.andrey_solution preprocess \
    --pairs-path data/raw/test_pairs_wo_target.parquet \
    --products-path data/raw/test_data.parquet
```

## Генерация признаков
### Статистики
```bash
python -m ozon_matching.andrey_solution generate-features \
    --feature-type categories \
    --feature-type characteristics \
    --feature-type colors \
    --feature-type names \
    --feature-type pictures \
    --feature-type variants \
    --pairs-path data/preprocessed/test_pairs_wo_target.parquet \
    --products-path data/preprocessed/test_data.parquet
```
В результате должна появиться директория `data/features`. 
Внутри лежат поддиректории с названиями `$--feature-type`.
Внутри поддиректорий лежат файлы `test.parquet`

### tf-idf
Если предварительно не запускалось обучение, то надо запустить шаги из обучения для построения tf-idf матриц или скачать их и распаковать:
```bash
bla-bla-bla
```

```bash
python -m ozon_matching.andrey_solution create-tfidf-similarity-features \
    --col-name characteristics_attributes \
    --col-name characteristics \
    --col-name name_norm_tokens \
    --col-name name_tokens \
    --col-name color_parsed \
    --pairs-path data/preprocessed/test_pairs_wo_target.parquet \
    --output-file test.parquet
```
В результате внутри директории `data/features` должны появиться поддиректории с названиями `tfidf-{$--col-name}`.
Внутри поддиректорий лежат файлы `test.parquet`

### Еще статистики
```bash
bash ozon_matching/kopatych_solution/workflows/v11/dag.sh
```
В результате внутри директории `data/features` должны появиться поддиректории с названиями 
`characteristics-extra`, `colors-extra`, `compatible-devices`, `titles`, `brands`.
Внутри поддиректорий лежат файлы `test.parquet`

### Фичи из сеток
Если предварительно не запускалось обучение, то надо запустить шаги из обучения для дообучения бертов или скачать их и распаковать (10Gb)
предполагается, что данные лежат в папке `hackathon_files_for_participants_ozon` на уровне запуска скриптов, а результат появится в папке nn в виде двух паркетов train и test с фичами  `chstic`, `mbert`, `colorbert`:
```bash
wget https://storage.yandexcloud.net/lcr/models_weights.tar?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=YCAJE3aFMqzbzkXdc3Q23murc%2F20230530%2Fru-central1%2Fs3%2Faws4_request&X-Amz-Date=20230530T193615Z&X-Amz-Expires=864000&X-Amz-Signature=0449107A1F498B238C9CDA2DEB42225B54F3567D8F61B2AF6221EF3C77DE9B4A&X-Amz-SignedHeaders=host
tar -xf models_weights.tarmodels_weights.tar
mkdir nn
python multibert.py && python multibert_chstic.py && python multibert_colors.py
```

#### Инференс
```bash
mkdir nn
python multibert_sub.py && python multibert_chstic_sub.py && python multibert_colors_sub.py
```


## Подготовка датасета для инференса
```bash
python -m ozon_matching.andrey_solution join-features \
    --features-path data/features/ \
    --pairs-path data/preprocessed/test_pairs_wo_target.parquet \
    --output-file test.parquet
```
В результате должна появиться директория `data/dataset`, в ней файл `test.parquet`

## Подготовка сабмита
Если предварительно не запускалось обучение, то надо запустить шаги из обучения для обучения моделей или скачать их и распаковать:
```bash
bla-bla-bla
```

```bash
python -m ozon_matching.andrey_solution prepare-submission \
    --test-path data/dataset/test.parquet \
    --experiment-path experiments/final
```
В результате должен появиться файл `experiments/final/submission.csv` с предиктами на тестовой выборке, готовый для загрузки на dsworks

# Обучение
## Предобработка
```bash
python -m ozon_matching.andrey_solution preprocess \
    --pairs-path data/raw/train_pairs.parquet \
    --products-path data/raw/train_data.parquet
```
В результате должны появиться файлы с таким же названием в директории `data/preprocessed`

## Генерация признаков
### Статистики
```bash
python -m ozon_matching.andrey_solution generate-features \
    --feature-type categories \
    --feature-type characteristics \
    --feature-type colors \
    --feature-type names \
    --feature-type pictures \
    --feature-type variants \
    --pairs-path data/preprocessed/train_pairs.parquet \
    --products-path data/preprocessed/train_data.parquet
```
В результате должна появиться директория `data/features`. 
Внутри лежат поддиректории с названиями `$--feature-type`.
Внутри поддиректорий лежат файлы `train.parquet`

### tf-idf
#### fit
```bash
python -m ozon_matching.andrey_solution create-tfidf-matrix \
    --col-name characteristics_attributes \
    --col-name characteristics \
    --col-name name_norm_tokens \
    --col-name name_tokens \
    --col-name color_parsed \
    --test-products-path data/preprocessed/test_data.parquet \
    --train-products-path data/preprocessed/train_data.parquet
```
В результате должна появиться директория `data/tfidf`.
Внутри лежат поддиректории с названиями `$--col-name`.
Внутри поддиректорий лежат файлы `matrix.npz`, `value_to_index.jbl`, `varaint_id_to_index.jbl`

#### transform
```bash
python -m ozon_matching.andrey_solution create-tfidf-similarity-features \
    --col-name characteristics_attributes \
    --col-name characteristics \
    --col-name name_norm_tokens \
    --col-name name_tokens \
    --col-name color_parsed \
    --pairs-path data/preprocessed/train_pairs.parquet \
    --output-file train.parquet
```
В результате внутри директории `data/features` должны появиться поддиректории с названиями `tfidf-{$--col-name}`.
Внутри поддиректорий лежат файлы `train.parquet`

### Еще статистики
```bash
bash ozon_matching/kopatych_solution/workflows/v11/dag.sh
```
В результате внутри директории `data/features` должны появиться поддиректории с названиями 
`characteristics-extra`, `colors-extra`, `compatible-devices`, `titles`.
Внутри поддиректорий лежат файлы `train.parquet`

### Фичи из сеток
#### fit
Осторожно: занимает много времени и требует наличия gpu
```bash
bla-bla-bla
```

#### transform
```bash
bla-bla-bla
```
В результате внутри директории `data/features` должны появиться поддиректории с названиями `chstic`, `mbert`, `colorbert`.
Внутри поддерикторий лежат файлы `train.parquet`

## Подготовка датасетов для обучения
```bash
python -m ozon_matching.andrey_solution join-features \
    --features-path data/features/ \
    --pairs-path data/preprocessed/train_pairs.parquet \
    --output-file train.parquet
```
В результате должна появиться директория `data/dataset`, в ней файл `train.parquet`

## Генерация сплитов
```bash
bla-bla-bla
```
В результате должен появиться файл `data/cv_pivot.parquet`

## Обучение модели
```bash
python -m ozon_matching.andrey_solution fit-catboost \
    --train-path data/dataset/train.parquet \
    --experiment-path experiments/final \
    --folds-path data/cv_pivot.parquet
```
В результате должна появиться директория `experiments/final`.
Внутри лежат поддиректории с названиями `--col-name`.
Внутри поддерикторий лежат файлы:
* `fold-0.cbm`, `fold-1.cbm`, `fold-2.cbm`, `fold-3.cbm`, `fold-4.cbm` – бинари моделей catboost по фолдам
* `metrics.json` – oof-метрики по фолдам
* `oof.parquet` – oof предикты
