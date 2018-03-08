wovle
==============================

update language model via word2vec

## 使い方

1. ソフトウェアのインストール
    1. 以下のソフトウェアをインストール
        `python, nkf, wget, unar, curl, mecab, libmecab-dev, mecab-ipadic-utf8`
    1. `./make requirements` （バージョンは固定してしまってるので適宜いい感じにする）
1. データのダウンロード
    1. 本を開く（`cd ./notebook && jupyter notebook`）
    1. jupyterで`00-download-data.ipynb`を実行する
1. 前処理
    - `./make data`

    ...で実行させたかったけど疲れたので無し  
    jupyterで`10,20,30`を順番に実行する。
1. 音声認識
    - `./make recognition no=[音声No]`
1. 特徴量抽出
    - `./make build-features no=[音声No]`
1. クラスタリング
    - `./make clustering no=[音声No]`
    - `clustering_grid.pyでグリッドサーチ（未整備）`
1. 言語モデル更新
    - `./make update no=[音声No] param=[パラメータ文字列]`
1. 再認識
    - `./make re-recognition no=[音声No] param=[パラメータ文字列]`
1. 認識率表示
    - `./make score no=[音声No] param=[パラメータ文字列]`
    - `calc_score_meisi.shで名詞のみの認識率（未整備）`
    - `（lxmlにがんばって日本語を通すと任意の単語が認識できたかがわかるので、`
      `品詞の割合などが読み取れる（実装コードはここにはない））`

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
