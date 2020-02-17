# Preprocessing for input sequences.
# Before learning is started, texts are converted to BERT input tokens.
# Format input texts to bert inputs from [CLS] + title + [SEP] + question + [SEP] + answer
# TODO: Refactor and make more readable

import numpy as np
from tqdm import tqdm
from math import floor, ceil
from collections import OrderedDict
import torch

from utils.text_cleaning import clean_test


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


# TODO: Check this shit
def _trim_input(tokenizer, title, question, answer, max_sequence_length,
                t_max_len=30, q_max_len=239, a_max_len=239, head_tail=False):
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len + q_len + a_len + 5) > max_sequence_length:

        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
            q_max_len = q_max_len + ceil((t_max_len - t_len) / 2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        if t_new_len + a_new_len + q_new_len + 5 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d"
                             % (max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))

        t = t[:t_new_len]
        if head_tail:
            # Head+Tail method
            q_len_head = round(q_new_len / 2)
            q_len_tail = -1 * (q_new_len - q_len_head)
            a_len_head = round(a_new_len / 2)
            a_len_tail = -1 * (a_new_len - a_len_head)
            q = q[:q_len_head] + q[q_len_tail:]
            a = a[:a_len_head] + a[a_len_tail:]
        else:
            # No Head+Tail, usual processing
            q = q[:q_new_len]
            a = a[:a_new_len]

    return t, q, a


def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    # 27 jan 0.42 cv - with title-question sep
    # stoken = ["[Q_CLS]"] + ["[A_CLS]"] + ["[REGR]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]
    stoken = ["[Q_CLS]"] + ["[A_CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]
    # stoken = ["[CLS]"] + title + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def _convert_to_bart_inputs(title, question, answer):

    tokens = bart.encode(['<s>']+title+['</s>']+question, answer)

    return tokens


# TODO: Check this
def compute_input_arrays(df, columns, tokenizer, max_sequence_length, t_max_len, q_max_len, a_max_len, head_tail,
                         do_preproc=False, two_inputs=False):
    if two_inputs:
        return compute_input_arrays_for_two_inputs(df, columns, tokenizer, max_sequence_length)
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows(), total=df.shape[0]):
        t, q, a = instance.question_title, instance.question_body, instance.answer
        if do_preproc:
            t = clean_test(t)
            q = clean_test(q)
            a = clean_test(a)
        t, q, a = _trim_input(tokenizer, t, q, a, max_sequence_length=max_sequence_length,
                              t_max_len=t_max_len, q_max_len=q_max_len, a_max_len=a_max_len, head_tail=head_tail)

        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    # TODO: Add another bert inputs
    return OrderedDict(
        features=torch.from_numpy(np.asarray(input_ids, dtype=np.long)),
        attention_mask=torch.from_numpy(np.asarray(input_masks, dtype=np.long)),
        token_type_ids=torch.from_numpy(np.asarray(input_segments, dtype=np.long))
    )


def _convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(str1, str2, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1, str2,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy)

        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids_q, input_masks_q, input_segments_q = return_id(
        title + ' ' + question, None, 'longest_first', max_sequence_length)

    input_ids_a, input_masks_a, input_segments_a = return_id(
        answer, None, 'longest_first', max_sequence_length)

    return [input_ids_q, input_masks_q, input_segments_q,
            input_ids_a, input_masks_a, input_segments_a]


def compute_input_arrays_for_two_inputs(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows(), total=df.shape[0]):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        ids_q, masks_q, segments_q, ids_a, masks_a, segments_a = \
            _convert_to_transformer_inputs(t, q, a, tokenizer, max_sequence_length)

        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

        input_ids_a.append(ids_a)
        input_masks_a.append(masks_a)
        input_segments_a.append(segments_a)

    return OrderedDict(
        features_question=torch.from_numpy(np.asarray(input_ids_q, dtype=np.long)),
        attention_mask_question=torch.from_numpy(np.asarray(input_masks_q, dtype=np.long)),
        token_type_ids_question=torch.from_numpy(np.asarray(input_segments_q, dtype=np.long)),
        features_answer=torch.from_numpy(np.asarray(input_ids_a, dtype=np.long)),
        attention_mask_answer=torch.from_numpy(np.asarray(input_masks_a, dtype=np.long)),
        token_type_ids_answer=torch.from_numpy(np.asarray(input_segments_a, dtype=np.long))
    )


def compute_output_arrays(df, columns, columns_question=None):
    if columns_question is None:
        return torch.from_numpy(df[columns].values)
    else:
        return torch.from_numpy(np.hstack((df[columns].values, df[columns_question].values)))


def rescale_targets(data, target_cols):
    for col in target_cols:
        n = data[col].nunique()
        values = data[col].unique()
        new_values = np.linspace(start=0, stop=1, num=n, endpoint=True)
        mapping = dict(zip(sorted(values), new_values))
        data[col] = data[col].map(mapping)
    return data

def code_reduce(text):
    lines = text.split('\n')
    red_lines = []
    for i in lines:
        if set(i).intersection(set([';', '{', '}'])):
            red_lines.append('CODE_LINE')
        else:
            red_lines.append(i)
    return ' \n '.join(red_lines)
