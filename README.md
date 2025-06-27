# MuvFDE
A library to generate fixed dimensional embeddings for multi-dimensional vectors based on [Google Muvera](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/).

![Version](https://img.shields.io/pypi/v/muvfde?style=flat-square)
![UnitTests](https://img.shields.io/github/actions/workflow/status/viig99/muvfde/pip.yml?branch=main&style=flat-square)
![Downloads](https://img.shields.io/pypi/dm/muvfde?style=flat-square)
![License](https://img.shields.io/github/license/viig99/muvfde?style=flat-square)


## Installation
```shell script
pip install --upgrade muvfde
```

## Integeration Benchmark
```shell script
uv venv --python 3.12
uv pip install -e .
uv pip install -r integeration_test/requirements.txt
python integeration_test/benchmark.py
```

## Results
| Metric                       | ColBERT v2   | Jina ColBERT v2   | FDE          |
|:-----------------------------|:-------------|:------------------|:-------------|
| ndcg_at_1                    | 0.46904      | 0.4613            | 0.13158      |
| ndcg_at_3                    | 0.4147       | 0.42327           | 0.10996      |
| ndcg_at_5                    | 0.38653      | 0.39435           | 0.1115       |
| ndcg_at_10                   | 0.34516      | 0.35649           | 0.10484      |
| ndcg_at_20                   | 0.31771      | 0.33264           | 0.10073      |
| ndcg_at_100                  | 0.30333      | 0.323             | 0.11111      |
| ndcg_at_1000                 | 0.37864      | 0.4085            | 0.20864      |
| map_at_1                     | 0.06055      | 0.06108           | 0.01077      |
| map_at_3                     | 0.10182      | 0.10379           | 0.0149       |
| map_at_5                     | 0.11673      | 0.12105           | 0.01868      |
| map_at_10                    | 0.13162      | 0.1384            | 0.02358      |
| map_at_20                    | 0.14415      | 0.15275           | 0.02748      |
| map_at_100                   | 0.15982      | 0.17192           | 0.03314      |
| map_at_1000                  | 0.17163      | 0.1853            | 0.04149      |
| recall_at_1                  | 0.06055      | 0.06108           | 0.01077      |
| recall_at_3                  | 0.11042      | 0.11434           | 0.01729      |
| recall_at_5                  | 0.13402      | 0.14125           | 0.02697      |
| recall_at_10                 | 0.1598       | 0.1744            | 0.04515      |
| recall_at_20                 | 0.19205      | 0.20963           | 0.06869      |
| recall_at_100                | 0.28145      | 0.30856           | 0.14218      |
| recall_at_1000               | 0.552        | 0.63163           | 0.47019      |
| precision_at_1               | 0.48607      | 0.47988           | 0.14241      |
| precision_at_3               | 0.38596      | 0.40041           | 0.10423      |
| precision_at_5               | 0.33003      | 0.33746           | 0.10341      |
| precision_at_10              | 0.24582      | 0.25604           | 0.08607      |
| precision_at_20              | 0.17786      | 0.19102           | 0.06625      |
| precision_at_100             | 0.07372      | 0.08241           | 0.03384      |
| precision_at_1000            | 0.01889      | 0.0206            | 0.01569      |
| mrr_at_1                     | 0.486068     | 0.479876          | 0.142415     |
| mrr_at_3                     | 0.550052     | 0.55418           | 0.175439     |
| mrr_at_5                     | 0.559804     | 0.564551          | 0.193705     |
| mrr_at_10                    | 0.565006     | 0.570379          | 0.206144     |
| mrr_at_20                    | 0.567171     | 0.573896          | 0.212625     |
| mrr_at_100                   | 0.569528     | 0.575261          | 0.217622     |
| mrr_at_1000                  | 0.569996     | 0.575833          | 0.218832     |
| nauc_ndcg_at_1_max           | 0.440418     | 0.414784          | 0.268555     |
| nauc_ndcg_at_1_std           | 0.213547     | 0.166462          | 0.136842     |
| nauc_ndcg_at_1_diff1         | 0.318255     | 0.407501          | 0.053946     |
| nauc_ndcg_at_3_max           | 0.490577     | 0.434876          | 0.271036     |
| nauc_ndcg_at_3_std           | 0.236718     | 0.184312          | 0.19998      |
| nauc_ndcg_at_3_diff1         | 0.223141     | 0.296838          | 0.062946     |
| nauc_ndcg_at_5_max           | 0.500952     | 0.421916          | 0.267371     |
| nauc_ndcg_at_5_std           | 0.279453     | 0.198252          | 0.182263     |
| nauc_ndcg_at_5_diff1         | 0.176046     | 0.247189          | 0.068921     |
| nauc_ndcg_at_10_max          | 0.480425     | 0.399418          | 0.303262     |
| nauc_ndcg_at_10_std          | 0.314832     | 0.212016          | 0.136579     |
| nauc_ndcg_at_10_diff1        | 0.159597     | 0.222947          | 0.072795     |
| nauc_ndcg_at_20_max          | 0.443027     | 0.380975          | 0.321634     |
| nauc_ndcg_at_20_std          | 0.280552     | 0.211846          | 0.101298     |
| nauc_ndcg_at_20_diff1        | 0.168327     | 0.219576          | 0.066569     |
| nauc_ndcg_at_100_max         | 0.420804     | 0.404721          | 0.405331     |
| nauc_ndcg_at_100_std         | 0.22803      | 0.216776          | 0.093032     |
| nauc_ndcg_at_100_diff1       | 0.194883     | 0.257504          | 0.04643      |
| nauc_ndcg_at_1000_max        | 0.459663     | 0.460067          | 0.427858     |
| nauc_ndcg_at_1000_std        | 0.264512     | 0.268762          | 0.127968     |
| nauc_ndcg_at_1000_diff1      | 0.197681     | 0.271556          | 0.050067     |
| nauc_map_at_1_max            | 0.134952     | 0.133128          | 0.467432     |
| nauc_map_at_1_std            | -0.224168    | -0.202274         | -0.269798    |
| nauc_map_at_1_diff1          | 0.520463     | 0.468012          | 0.05543      |
| nauc_map_at_3_max            | 0.235025     | 0.206459          | 0.44363      |
| nauc_map_at_3_std            | -0.175923    | -0.177065         | -0.148846    |
| nauc_map_at_3_diff1          | 0.40453      | 0.393757          | 0.03092      |
| nauc_map_at_5_max            | 0.279919     | 0.231995          | 0.429922     |
| nauc_map_at_5_std            | -0.130096    | -0.162264         | -0.090674    |
| nauc_map_at_5_diff1          | 0.348037     | 0.364103          | 0.051462     |
| nauc_map_at_10_max           | 0.325895     | 0.27353           | 0.43387      |
| nauc_map_at_10_std           | -0.066359    | -0.119012         | -0.069416    |
| nauc_map_at_10_diff1         | 0.305329     | 0.33252           | 0.032106     |
| nauc_map_at_20_max           | 0.350215     | 0.305735          | 0.43255      |
| nauc_map_at_20_std           | -0.021044    | -0.071707         | -0.046785    |
| nauc_map_at_20_diff1         | 0.272759     | 0.306941          | 0.039246     |
| nauc_map_at_100_max          | 0.362594     | 0.325282          | 0.442787     |
| nauc_map_at_100_std          | 0.045316     | 0.009331          | 0.010934     |
| nauc_map_at_100_diff1        | 0.250008     | 0.290768          | 0.03672      |
| nauc_map_at_1000_max         | 0.368975     | 0.32697           | 0.414434     |
| nauc_map_at_1000_std         | 0.088677     | 0.047439          | 0.057809     |
| nauc_map_at_1000_diff1       | 0.240806     | 0.282087          | 0.04931      |
| nauc_recall_at_1_max         | 0.134952     | 0.133128          | 0.467432     |
| nauc_recall_at_1_std         | -0.224168    | -0.202274         | -0.269798    |
| nauc_recall_at_1_diff1       | 0.520463     | 0.468012          | 0.05543      |
| nauc_recall_at_3_max         | 0.230551     | 0.211642          | 0.401272     |
| nauc_recall_at_3_std         | -0.174305    | -0.162588         | -0.139841    |
| nauc_recall_at_3_diff1       | 0.395117     | 0.367957          | -0.00939     |
| nauc_recall_at_5_max         | 0.264562     | 0.222997          | 0.363865     |
| nauc_recall_at_5_std         | -0.130317    | -0.149047         | -0.101748    |
| nauc_recall_at_5_diff1       | 0.310632     | 0.304356          | 0.067209     |
| nauc_recall_at_10_max        | 0.304145     | 0.246904          | 0.402029     |
| nauc_recall_at_10_std        | -0.057703    | -0.10716          | -0.089381    |
| nauc_recall_at_10_diff1      | 0.254648     | 0.270364          | 0.018997     |
| nauc_recall_at_20_max        | 0.298737     | 0.270968          | 0.327109     |
| nauc_recall_at_20_std        | -0.01793     | -0.036123         | -0.04656     |
| nauc_recall_at_20_diff1      | 0.206511     | 0.222335          | 0.00928      |
| nauc_recall_at_100_max       | 0.283664     | 0.295975          | 0.304568     |
| nauc_recall_at_100_std       | 0.102043     | 0.149719          | 0.022056     |
| nauc_recall_at_100_diff1     | 0.121299     | 0.186879          | 0.02251      |
| nauc_recall_at_1000_max      | 0.23539      | 0.196962          | 0.264563     |
| nauc_recall_at_1000_std      | 0.178421     | 0.176776          | 0.114291     |
| nauc_recall_at_1000_diff1    | 0.035984     | 0.059028          | 0.025409     |
| nauc_precision_at_1_max      | 0.431441     | 0.418561          | 0.307331     |
| nauc_precision_at_1_std      | 0.209316     | 0.198142          | 0.101121     |
| nauc_precision_at_1_diff1    | 0.320835     | 0.40996           | 0.0702       |
| nauc_precision_at_3_max      | 0.509605     | 0.429727          | 0.27091      |
| nauc_precision_at_3_std      | 0.300477     | 0.243193          | 0.228439     |
| nauc_precision_at_3_diff1    | 0.116036     | 0.191677          | 0.05983      |
| nauc_precision_at_5_max      | 0.515557     | 0.404031          | 0.253536     |
| nauc_precision_at_5_std      | 0.401979     | 0.286749          | 0.223356     |
| nauc_precision_at_5_diff1    | 0.011756     | 0.097248          | 0.059779     |
| nauc_precision_at_10_max     | 0.474925     | 0.361102          | 0.256876     |
| nauc_precision_at_10_std     | 0.518904     | 0.37201           | 0.211409     |
| nauc_precision_at_10_diff1   | -0.064326    | 0.012417          | 0.081217     |
| nauc_precision_at_20_max     | 0.400418     | 0.30358           | 0.222033     |
| nauc_precision_at_20_std     | 0.546197     | 0.439675          | 0.211123     |
| nauc_precision_at_20_diff1   | -0.100292    | -0.040087         | 0.094631     |
| nauc_precision_at_100_max    | 0.250123     | 0.140257          | 0.133771     |
| nauc_precision_at_100_std    | 0.640884     | 0.527872          | 0.271889     |
| nauc_precision_at_100_diff1  | -0.137625    | -0.080946         | 0.072067     |
| nauc_precision_at_1000_max   | 0.153336     | -0.015312         | -0.075492    |
| nauc_precision_at_1000_std   | 0.58537      | 0.446167          | 0.239417     |
| nauc_precision_at_1000_diff1 | -0.109024    | -0.116597         | 0.090136     |
| nauc_mrr_at_1_max            | 0.431441     | 0.418561          | 0.307331     |
| nauc_mrr_at_1_std            | 0.209316     | 0.198142          | 0.101121     |
| nauc_mrr_at_1_diff1          | 0.320835     | 0.40996           | 0.0702       |
| nauc_mrr_at_3_max            | 0.479451     | 0.469744          | 0.308737     |
| nauc_mrr_at_3_std            | 0.236395     | 0.240663          | 0.119734     |
| nauc_mrr_at_3_diff1          | 0.336093     | 0.392852          | 0.043912     |
| nauc_mrr_at_5_max            | 0.484796     | 0.476115          | 0.315624     |
| nauc_mrr_at_5_std            | 0.243275     | 0.24252           | 0.10594      |
| nauc_mrr_at_5_diff1          | 0.32899      | 0.390362          | 0.050617     |
| nauc_mrr_at_10_max           | 0.489404     | 0.472745          | 0.326788     |
| nauc_mrr_at_10_std           | 0.252878     | 0.247655          | 0.098184     |
| nauc_mrr_at_10_diff1         | 0.325208     | 0.387673          | 0.057138     |
| nauc_mrr_at_20_max           | 0.488438     | 0.47402           | 0.323381     |
| nauc_mrr_at_20_std           | 0.252903     | 0.252912          | 0.101504     |
| nauc_mrr_at_20_diff1         | 0.323893     | 0.388055          | 0.056116     |
| nauc_mrr_at_100_max          | 0.489666     | 0.475351          | 0.323934     |
| nauc_mrr_at_100_std          | 0.254377     | 0.254073          | 0.103622     |
| nauc_mrr_at_100_diff1        | 0.322078     | 0.388537          | 0.053497     |
| nauc_mrr_at_1000_max         | 0.489313     | 0.474873          | 0.324247     |
| nauc_mrr_at_1000_std         | 0.253888     | 0.253582          | 0.103707     |
| nauc_mrr_at_1000_diff1       | 0.322043     | 0.388485          | 0.053168     |
| main_score                   | 0.34516      | 0.35649           | 0.10484      |
| hf_subset                    | default      | default           | default      |
| languages                    | ['eng-Latn'] | ['eng-Latn']      | ['eng-Latn'] |
## References
* [Google Muvera Blog](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/)
* [MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings](https://arxiv.org/abs/2405.19504)
* [Reference Fixed Dimensional embeddings code](https://github.com/google/graph-mining/tree/main/sketching/point_cloud)