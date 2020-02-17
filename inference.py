import os
import pydoc
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict

# pytorch
import torch
from torch.utils.data import DataLoader

# transformers
from transformers import AutoTokenizer

# catalyst
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import InferCallback, CheckpointCallback

from dataset import TokenDataset
from utils.metrics import spearman_correlation
from utils.callbacks import SpearmanCorrelationMetricCallback
from utils.preprocessing import compute_input_arrays, compute_output_arrays
from utils.metrics import spearman_correlation, spearman_correlation_columnwise
from utils.helper_functions import get_config, seed_everything, get_dir_to_save_model, target_cols


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--thres', type=float, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = get_config(args.config)
    thres = args.thres

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_to_load_model = config['train'].get('dir_to_load_model')
    if dir_to_load_model is None:
        dir_to_load_model = os.path.join(get_dir_to_save_model(config), f'fold_{args.fold}' if args.fold is not None else '')
    print('Loading weights from:', dir_to_load_model)

    # read data
    train_df = pd.read_csv(config['data']['path_to_train_csv'])
    valid_df = pd.read_csv(config['data']['path_to_valid_csv'])
    test_df = pd.read_csv(config['data']['path_to_test_csv'])
    input_categories = list(train_df.columns[[1, 2, 5]])

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['train']['model_params']['pretrained_model_name'])
    print(len(tokenizer))

    if config['data'].get('additional_special_tokens') is not None:
        config['train']['model_params']['num_additional_special_tokens'] = len(
            config['data']['additional_special_tokens']
        )
        tokenizer.add_special_tokens(dict(additional_special_tokens=config['data']['additional_special_tokens']))
        print(config['data']['additional_special_tokens'])
    print(len(tokenizer))

    # texts to bert format
    print(config['data']['preprocessing_params'])
    inputs_valid = compute_input_arrays(valid_df, input_categories, tokenizer, **config['data']['preprocessing_params'])
    targets_valid = compute_output_arrays(valid_df, columns=target_cols)
    # dataset
    valid_dataset = TokenDataset(inputs_valid, targets_valid, config['data']['bert_inputs'])
    # catalyst dataloaders
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=64,
                              shuffle=False)

    model = pydoc.locate(config['train']['model'])(**config['train']['model_params'])
    # catalyst runner
    runner = SupervisedRunner(
        input_key=config['data']['bert_inputs']
    )
    # model inference
    runner.infer(
        model=model,
        loaders=OrderedDict(loader=valid_loader),
        callbacks=[
            CheckpointCallback(
                resume=f"{dir_to_load_model}/checkpoints/best.pth"
            ),
            InferCallback(),
            SpearmanCorrelationMetricCallback()
        ],
        verbose=True
    )
    predicted_logits = runner.callbacks[0].predictions['logits']
    valid_preds = torch.sigmoid(torch.tensor(predicted_logits)).numpy()
    spearm_corr = spearman_correlation(targets_valid.numpy(), valid_preds)
    print('Spearman correlation:', runner.callbacks[1].metric_value, spearm_corr)

    # test inference
    inputs_test = compute_input_arrays(test_df, input_categories, tokenizer, **config['data']['preprocessing_params'])
    test_dataset = TokenDataset(inputs_test, bert_inputs=config['data']['bert_inputs'])
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=64,
                             shuffle=False)
    # model inference
    runner.infer(
        model=model,
        loaders=OrderedDict(loader=test_loader),
        callbacks=[
            CheckpointCallback(
                resume=f"{dir_to_load_model}/checkpoints/best.pth"
            ),
            InferCallback(),
        ],
        verbose=True
    )
    predicted_logits = runner.callbacks[0].predictions['logits']
    test_preds = torch.sigmoid(torch.tensor(predicted_logits)).numpy()
    print(test_preds.shape)
    submission = pd.read_csv('data/sample_submission.csv')
    submission.loc[:, 'question_asker_intent_understanding':] = test_preds
    dir_to_save_submission = os.path.join('submissions', dir_to_load_model.split('/')[-2])
    print(dir_to_save_submission)
    if not os.path.exists(dir_to_save_submission):
        os.makedirs(dir_to_save_submission)
    submission.to_csv(f'{dir_to_save_submission}/submission.csv', index=False)
    print(submission.head())

    # Save valid and test preds
    dir_to_save_predictions = os.path.join('predictions', dir_to_load_model.split('/')[-2])
    print(dir_to_save_predictions)
    if not os.path.exists(dir_to_save_predictions):
        os.makedirs(dir_to_save_predictions)
    with open(os.path.join(dir_to_save_predictions, 'valid.pkl'), 'wb') as f:
        pickle.dump(valid_preds, f)
    with open(os.path.join(dir_to_save_predictions, 'test.pkl'), 'wb') as f:
        pickle.dump(test_preds, f)
