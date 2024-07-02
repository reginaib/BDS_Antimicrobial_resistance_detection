from finetuning import start_sweep


sweep_config = {
        "method": "random",
        "metric": {
            "goal": "minimize",
            "name": "Validation/val_loss"
        },
        "parameters": {
            "input_dim": {"value": 6000},
            "n_hidden_layers": {"values": [1, 2, 3, 4]},
            "hidden_dim": {"values": [16, 32, 64, 128, 256, 512, 768, 1024]},
            "lr": {"values": [0.00001, 0.00003, 0.0001, 0.0003, 0.001]},
            "dr": {'values': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]},
            "accelerator": {"value": "gpu"},
            "max_epochs": {"value": 100},
            "train_dataset": {"value": './bds_project/analysis/train_dataset.pt'},
            "valid_dataset": {"value": './bds_project/analysis/valid_dataset.pt'},
            "test_dataset": {"value": './bds_project/analysis/test_dataset.pt'},
            "save_preds": {"value": False},
        }
}

start_sweep(config=sweep_config, project_name='BDS_task_comb', num_config=100)