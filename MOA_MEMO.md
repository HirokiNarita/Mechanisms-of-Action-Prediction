# MEMO

## Domain
- 推測されるデータの前処理過程[(URL)](https://www.kaggle.com/c/lish-moa/discussion/184005#1034211)
    * LEVEL1 : 実験から得られた生データ
    * LEVEL2 : レベル1のデータから各遺伝子にピーク値を割り当てる
    * LEVEL3 : LEVEL2のn郡間に対してQuantile Normalization(分位正規化)を行う.
    * LEVEL4 : LEVEL3に対して（x-中央値）/（中央絶対偏差* 1.4826）
    * LEVEL5 : 実験環境を変えた同一薬品での重み付け平均？（よくわからないたすけて）

- g-xの値の意味[(URL)](https://www.kaggle.com/c/lish-moa/discussion/180390#1000307)
    * g-1、g-2、....、g-772は、サンプル中の各遺伝子1、2、...、772のmRNAレベルの正規化された値を表す。
    * g-xの高い絶対値(>2または<2)は、薬剤または摂動が遺伝子xに「有意な」効果を持っていたことを示し、ゼロに近い値は、その遺伝子に対する効果が測定不能であったことを意味する。

## EDA
- 相関が高いtaget columnがあることがわかる [(URL)](https://www.kaggle.com/amiiiney/drugs-classification-mechanisms-of-action#4-Targets-(MoA))

- ctl_vehicleのラベルは全てゼロなので、predictはしない。

## TIPS
- Multi Label問題であるが、Multi Class問題として捉えることもできる[(URL)](https://www.kaggle.com/c/lish-moa/discussion/180500)
    * MoAのコンテキストでは、ちょうど2つのMoAが活性化されている1538個のsig_idがあり、これらは96の異なる組み合わせで(たった？)現れます。
    * 206 + 96 = 302クラスを使ってマルチクラスを行うことができます。(ただし、trainにない組み合わせになると、トリッキーになる。)

- MoA: Multi Input ResNet Model[(URL)](https://www.kaggle.com/rahulsd91/moa-multi-input-resnet-model)
    * input1,input2と分けて片方は1層目でweightを計算、片方は2層目で結合してあげる（skip connection）ような学習をしている。
    * (1層目で蒸留された情報と組み合わせることで精度があがっている（？）)

## TECNICAL
- iterative-stratification(sklearn likeな層化抽出ライブラリ)[(URL)](https://github.com/trent-b/iterative-stratification)
    * MultilabelStratifiedKFoldというMulti Label用の層化抽出法がある