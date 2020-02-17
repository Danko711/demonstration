import os
import pydoc
import argparse
from collections import OrderedDict
import pandas as pd

# pytorch
import torch
from torch.utils.data import DataLoader

# transformers
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

# catalyst
from catalyst.dl.runner import SupervisedRunner
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.dl.callbacks import EarlyStoppingCallback, CheckpointCallback, OptimizerCallback, \
    CriterionCallback, CriterionAggregatorCallback

from dataset import TokenDataset
from utils.callbacks import SpearmanCorrelationMetricCallback
from utils.preprocessing import compute_input_arrays, compute_output_arrays, rescale_targets
from utils.helper_functions import get_config, get_dir_to_save_model, save_config, target_cols


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = get_config(args.config)
    if args.lr is not None:
        config['train']['lr'] = args.lr
        if 'OneCycleLRWithWarmup' in config['train']['scheduler']:
            # config['train']['scheduler_params']['lr_range'] = [args.lr * 3, args.lr * 1.5, args.lr]
            config['train']['scheduler_params']['lr_range'] = [args.lr * 2.5, args.lr / 1.01, args.lr / 1.5]
            config['train']['scheduler_params']['init_lr'] = args.lr
    print(config['train']['scheduler_params'])

    set_global_seed(args.seed)  # reproducibility
    prepare_cudnn(deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_to_save_model = os.path.join(get_dir_to_save_model(config),
                                     f'fold_{args.fold}' if args.fold is not None else '')
    print('Saving weights to:', dir_to_save_model)
    if not os.path.exists(dir_to_save_model):
        os.makedirs(dir_to_save_model)
    save_config(os.path.join(
        dir_to_save_model,
        f'config_fold_{args.fold}.yml' if args.fold is not None else 'config.yml'),
        config)

    # read data
    train_df = pd.read_csv(config['data']['path_to_train_csv'])
    # always use global validation
    if args.fold is not None:
        train_df = train_df[train_df['fold'] != args.fold].reset_index(drop=True)
    valid_df = pd.read_csv(config['data']['path_to_valid_csv'])
    input_categories = list(train_df.columns[[1, 2, 5, 9]])

    if args.debug:
        train_df = train_df.head(128)
        valid_df = valid_df.head(128)

    print(train_df.shape, valid_df.shape)
    all_df = pd.concat((train_df, valid_df))
    all_df = rescale_targets(all_df, target_cols)
    print(all_df.head())
    train_df, valid_df = all_df.iloc[:train_df.shape[0], :], all_df.iloc[train_df.shape[0]:, :]
    print(train_df.shape, valid_df.shape)

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
    inputs_train = compute_input_arrays(train_df, input_categories, tokenizer, **config['data']['preprocessing_params'])
    targets_train = compute_output_arrays(train_df, columns=target_cols)

    inputs_valid = compute_input_arrays(valid_df, input_categories, tokenizer, **config['data']['preprocessing_params'])
    targets_valid = compute_output_arrays(valid_df, columns=target_cols)

    # dataset
    train_dataset = TokenDataset(inputs_train, targets_train, config['data']['bert_inputs'])
    valid_dataset = TokenDataset(inputs_valid, targets_valid, config['data']['bert_inputs'])

    # catalyst dataloaders
    train_val_loaders = OrderedDict(
        train=DataLoader(dataset=train_dataset,
                         batch_size=config['train']['batch_size'],
                         shuffle=True),
        valid=DataLoader(dataset=valid_dataset,
                         batch_size=config['train']['batch_size'],
                         shuffle=False)
    )

    model = pydoc.locate(config['train']['model'])(**config['train']['model_params'])
    criterion_callbacks = None
    if isinstance(config['train']['loss'], str):
        criterion = pydoc.locate(config['train']['loss'])(**config['train']['loss_params'])
        criterion.to(device)
        output_keys = 'logits'
    elif isinstance(config['train']['loss'], dict):
        criterion_callbacks = []
        criterion = {}
        loss_keys, output_keys = [], []
        for criterion_name, criterion_dict in config['train']['loss'].items():
            criterion.update({
                criterion_name: pydoc.locate(criterion_dict['function'])(**criterion_dict['loss_params'])
            })
            criterion_callbacks.append(
                CriterionCallback(
                    criterion_key=criterion_name,
                    **criterion_dict['callback_keys']
                )
            )
            loss_keys.append(criterion_dict['callback_keys']['prefix'])
            output_keys.append(criterion_dict['callback_keys']['output_key'])
        criterion_callbacks.append(
            CriterionAggregatorCallback(
                prefix="loss",
                loss_keys=loss_keys,
                loss_aggregate_fn="mean"
            )
        )
    else:
        raise ValueError('Criterion must be str or dict')
    print(criterion)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config['train']['lr'], eps=4e-5)

    if 'schedule_with_warmup' in config['train']['scheduler']:
        num_training_steps = int(len(train_dataset) / \
                                 (config['train']['accumulation_steps'] * config['train']['batch_size']))
        config['train']['scheduler_params']['num_training_steps'] = num_training_steps
        print(config['train']['scheduler_params'])
    scheduler = pydoc.locate(config['train']['scheduler'])(
        optimizer=optimizer,
        **config['train']['scheduler_params'])
    print(scheduler)

    # catalyst runner
    print(config['data']['bert_inputs'])
    print(output_keys)
    runner = SupervisedRunner(
        input_key=config['data']['bert_inputs'],
        output_key=output_keys
    )
    # train callbacks
    callbacks = [
        OptimizerCallback(accumulation_steps=config['train']['accumulation_steps']),
        # EarlyStoppingCallback(patience=config['train']['patience'], metric='spearman', minimize=False),
        CheckpointCallback(save_n_best=1)
    ]
    if criterion_callbacks is not None:
        callbacks.extend(criterion_callbacks)
        callbacks.extend([
            SpearmanCorrelationMetricCallback(output_key='cls_logits', prefix='cls_spearman'),
            SpearmanCorrelationMetricCallback(output_key='mse_logits', prefix='mse_spearman', activation='none')
        ])
    else:
        callbacks.append(SpearmanCorrelationMetricCallback())
    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=train_val_loaders,
        callbacks=callbacks,
        fp16=dict(opt_level=config['train']['fp16']),
        logdir=dir_to_save_model,
        num_epochs=config['train']['num_epochs'],
        main_metric='spearman' if criterion_callbacks is None else 'cls_spearman',
        minimize_metric=False,
        verbose=True
    )
