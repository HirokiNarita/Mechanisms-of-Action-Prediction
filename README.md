# Mechanisms of Action (MoA) Prediction
## Final submission deadline.
- November 30, 2020

## Data
In this competition, you will be predicting multiple targets of the Mechanism of Action (MoA) response(s) of different samples (sig_id), given various inputs such as gene expression data and cell viability data.

Two notes:

the training data has an additional (optional) set of MoA labels that are not included in the test data and not used for scoring.
the re-run dataset has approximately 4x the number of examples seen in the Public test.
Files
train_features.csv - Features for the training set. Features g- signify gene expression data, and c- signify cell viability data. cp_type indicates samples treated with a compound (cp_vehicle) or with a control perturbation (ctrl_vehicle); control perturbations have no MoAs; cp_time and cp_dose indicate treatment duration (24, 48, 72 hours) and dose (high or low).
train_targets_scored.csv - The binary MoA targets that are scored.
train_targets_nonscored.csv - Additional (optional) binary MoA responses for the training data. These are not predicted nor scored.
test_features.csv - Features for the test data. You must predict the probability of each scored MoA for each row in the test data.
sample_submission.csv - A submission file in the correct format.
### 日本語訳
本コンテストでは、遺伝子発現データや細胞生存率データなどの様々なインプットを与えられた異なるサンプル（sig_id）の作用機序（MoA）応答の複数のターゲットを予測

2つの注意点がある

トレーニングデータには、テストデータには含まれず、スコアリングには使用されないMoAラベルの追加（オプション）セットがある
再実行データセットは、パブリックテストで見られる例の約4倍の数を持つ
### ファイル
- train_features.csv - train_features.csv - 訓練セットの特徴量．g-は遺伝子発現データ，c-は細胞生存率データを示す． cp_typeは化合物（cp_vehicle）または対照摂動（ctrl_vehicle）で処理されたサンプルを示す；対照摂動はMoAsを持たない；cp_timeとcp_doseは処理時間（24, 48, 72時間）と投与量（高または低）を示す．
    * g-接頭辞を持つ特徴量は遺伝子発現特徴量であり、その数は772個（g-0からg-771まで）ある
    * c-接頭辞を持つ特徴量は細胞生存率の特徴量であり、その数は100個（c-0からc-99まで）ある
    * cp_typeは，サンプルが化合物で処理されたか，対照摂動（rt_cpまたはctl_vehicle）で処理されたかを示す2値のカテゴリ特徴量
    * cp_timeは，治療期間（24時間，48時間，72時間）を示す分類的特徴量
    * cp_doseは，投与量が低いか高いかを示す2値のカテゴリ特徴量である(D1またはD2)．
- train_targets_scored.csv - スコアされるバイナリMoAターゲット
    * Number of Scored Target Features: 206
- train_targets_nonscored.csv - 訓練データの追加の（オプションの）バイナリMoA反応。これらは予測もスコア化もされない
    * Number of Non-scored Target Features: 402
- test_features.csv - テストデータの特徴量．テストデータの各行のスコアされたMoAの確率を予測する必要がある
- sample_submission.csv - 正しい形式の提出ファイル

## 評価指標
各カラムにおけるバイナリクロスエントロピーを計算しその平均値を、すべてのサンプルで平均した値

## ToDo
- [ ] ノンスコアのターゲットを予測し、その後のモデルのメタ特徴として使用
- [ ] ノンスコアのターゲットも含めたモデルで学習する
- [ ] カテゴリ変数を埋め込み特徴量として学習
- [ ] AEでデノイズor中間層を特徴量に追加(Nakayamaさんこれ好きなイメージ)
- [ ] メトリックラーニングをAEに適応して、他のモデルの特徴量にする (クラスタリングの重心を特徴量に加えるイメージ)
- [ ] Multi Label問題であるが、Multi Class問題として捉える[URL](https://www.kaggle.com/c/lish-moa/discussion/180500)
- [https://www.kaggle.com/c/lish-moa/discussion/184005](https://www.kaggle.com/c/lish-moa/discussion/184005)
* データの前処理
- [ ] label smooth
- [ ] トレーニングデータからコントロールグループを削除するとCV上がるらしい.LBは下がるけどブレンドするとLBもup
- [ ] pretictのlowerとupperをclip
- [ ] コントロールの出力を全て確認。もしかしたらすべて0かも
- [ ] Pseudolabeling
- [ ] バランシングの適応 (優先順位低め)
- [ ] アップサンプリング[https://www.kaggle.com/c/lish-moa/discussion/187419](https://www.kaggle.com/c/lish-moa/discussion/187419)
    * [ノートブック](https://www.kaggle.com/tolgadincer/upsampling-multilabel-data-with-mlsmote)
    * [CVの方法のノートブック](https://www.kaggle.com/tolgadincer/mlsmote) (優先順位低め)
- [ ] ImbalancedDataSampler[pytorchの実装github]

## On Going
- [ ] ノンスコアのターゲットを予測し、その後のモデルのメタ特徴として使用
- [ ] ノンスコアのターゲットも含めたモデルで学習する
## Done
- [x] baseline MLP
    * CV : 0.015842093180158456
    * Public : Public : 0.01979
- [x] Maxout MLP
    * CV : 0.015528392642207555
    * Public : 0.01930
## my task
- [ ] カテゴリ変数を埋め込み特徴量として学習
- [ ] AEでデノイズor中間層を特徴量に追加(Nakayamaさんこれ好きなイメージ)
- [ ] メトリックラーニングをAEに適応して、他のモデルの特徴量にする (クラスタリングの重心を特徴量に加えるイメージ)
