class IO_CFG:
    input_root =  '/media/hiroki/working/kaggle/Mechanisms-of-Action-Prediction/datasets'
    output_root =  '/media/hiroki/working/kaggle/Mechanisms-of-Action-Prediction/result/MLP/baseline'

class MODEL_CFG:
    max_grad_norm=1000
    gradient_accumulation_steps=1
    hidden_size=512
    dropout=0.5
    lr=1e-2
    weight_decay=1e-6
    batch_size=32
    epochs=20
    #total_cate_size=5
    #emb_size=4
    num_features=872
    #cat_features=cat_features
    target_cols=206