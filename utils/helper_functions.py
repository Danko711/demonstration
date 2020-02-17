import os
import yaml
import torch
import random
import calendar
import numpy as np
import os.path as osp
from datetime import datetime


# TODO: add cudnn.benchmark and deterministic
# TODO: set torch.backends params in config
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def save_config(path, config):
    with open(path, 'w') as stream:
        yaml.dump(config, stream, default_flow_style=False)


def get_date():
    today = datetime.today()
    month_name = calendar.month_abbr[today.month].lower()
    return month_name + str(today.day)


def get_dir_to_save_model(config):
    current_date = get_date()
    model_name = config['train']['model_params']['pretrained_model_name']
    batch_size = config['train']['batch_size']
    accumulation_steps = config['train']['accumulation_steps']
    lr = config['train']['lr']
    experiment_name = config['train']['experiment_name']
    return osp.join(
        config['train']['dir_to_save_model'],
        f'{current_date}_{model_name}_bs{batch_size}_as{accumulation_steps}_lr{lr}_{experiment_name}'
    )


target_cols = ['question_asker_intent_understanding', 'question_body_critical',
               'question_conversational', 'question_expect_short_answer',
               'question_fact_seeking', 'question_has_commonly_accepted_answer',
               'question_interestingness_others', 'question_interestingness_self',
               'question_multi_intent', 'question_not_really_a_question',
               'question_opinion_seeking', 'question_type_choice',
               'question_type_compare', 'question_type_consequence',
               'question_type_definition', 'question_type_entity',
               'question_type_instructions', 'question_type_procedure',
               'question_type_reason_explanation', 'question_type_spelling',
               'question_well_written', 'answer_helpful',
               'answer_level_of_information', 'answer_plausible',
               'answer_relevance', 'answer_satisfaction',
               'answer_type_instructions', 'answer_type_procedure',
               'answer_type_reason_explanation', 'answer_well_written']


question_cols = ['question_asker_intent_understanding', 'question_body_critical',
                 'question_conversational', 'question_expect_short_answer',
                 'question_fact_seeking', 'question_has_commonly_accepted_answer',
                 'question_interestingness_others', 'question_interestingness_self',
                 'question_multi_intent', 'question_not_really_a_question',
                 'question_opinion_seeking', 'question_type_choice',
                 'question_type_compare', 'question_type_consequence',
                 'question_type_definition', 'question_type_entity',
                 'question_type_instructions', 'question_type_procedure',
                 'question_type_reason_explanation', 'question_type_spelling',
                 'question_well_written']