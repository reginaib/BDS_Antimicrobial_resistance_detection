from finetuning import start_training


class config:
    input_dim = 2
    n_hidden_layers = 4
    hidden_dim = 128
    lr = 0.00003
    dr = 0.1
    patience = 15
    accelerator = 'gpu'
    max_epochs = 100
    train_dataset =  './bds_project/analysis/train_dataset_lda.pt'
    valid_dataset = './bds_project/analysis/valid_dataset_lda.pt'
    test_dataset = './bds_project/analysis/test_dataset_lda.pt'
    save_preds = False


start_training(config=config, project_name='BDS_task_comb_train')