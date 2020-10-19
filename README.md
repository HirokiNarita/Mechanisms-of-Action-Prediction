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
- Multi Label問題であるが、Multi Class問題として捉える[URL](https://www.kaggle.com/c/lish-moa/discussion/180500)
- 次元圧縮したときにスコアがどの程度変わるか確認（接頭辞が同じものについては、相関が非常に高いため）
- scored,unscoredのlabelを全て使って学習、transfer-learning or fine-tuning
## On Going
- 
## Done
- 

## memo

## diary
- 10/14(水)
    - 今日やったこと
        * 初回キックオフMTG
        * タスクの確認と目的の設定
        * 毎週火曜日 22:00　にMTGを決定
    - 次やること
        * MLPのベースライン提出
        * discussionの確認
- 10/15(木)
    - 今日やったこと
        * 実験環境とベースライン作成(途中)
        * ディスカッションよんだ
- 10/18(日)
    - 今日やったこと
        * 実験環境とベースライン作成(実行まだしてない)
        * ディスカッションよんだ
- 10/19(月)
    - 今日やったこと
        * ベースラインと提出用ノートブック作成（完成）