# 配置文件
conf = {
    
    "dataroot": "./data",

    # 数据类型，tabular, image
    "data_type": "tabular",

    # 选择模型 mlp, simple-cnn, vgg
    "model_name": "mlp",

    # 全局 epoch
    "global_epochs": 100,

    # 本地 epoch
    "local_epochs": 3,

    # 狄利克雷参数
    "beta": 0.5,

    # FedMix 超参数
    "mean_batch": 5,
    "lambda": 0.05,

    "batch_size": 64,

    "weight_decay": 1e-5,

    # 学习速率
    "lr": 0.001,
    "momentum": 0.9,

    # 节点数
    "diff_clients": False,
    "num_parties": 5,

    # 模型聚合权值
    "is_init_avg": True,

    # 模型保存目录
    "model_dir": "./save_model/",

    # 选择数据集
    "which_dataset": "clinical",
    # "which_dataset": "credit",
    # "which_dataset": "tb",
    # "which_dataset": "covtype",
    # "which_dataset": "intrusion",
    # "which_dataset": "body",

    # 分类
    "num_classes": {
        "clinical": 2,
        "credit": 2,
        "tb": 2,
        "covtype": 7,
        "intrusion": 10,
        "body": 4,
    },

    # 离散(categorial)列名
    "discrete_columns": {
        "adult": ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country'],
        "intrusion": ['protocol_type', 'service', 'flag'],
        "credit": [],
        "covtype": ['Wilderness_Area4', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Soil_Type40', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39'],
        "clinical": ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"],
        "tb": [],
        "body": ['gender'],
    },

    # 标签列名
    "label_column": {
        "clinical": "label",
        "credit": "Class",
        "covtype": "Cover_Type",
        "intrusion": "label",
        "tb": "Condition",
        "body": "class",
    },
}
