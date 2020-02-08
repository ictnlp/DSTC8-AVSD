# coding: utf-8
# author: noctli
import json
import pickle
import logging
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from itertools import chain
# from train import SPECIAL_TOKENS, MODEL_INPUTS, PADDED_INPUTS
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>","<cap>", "<video>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>"], 'pad_token': "<pad>"}
MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]


def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)    

def get_dataset(tokenizer, data_file, feature_path=None, undisclosed_only=False, n_history=3):
    
    dialog_data = json.load(open(data_file, 'r'))
    dialog_list = []
    vid_set = set()
    for dialog in dialog_data['dialogs']:
        caption = [tokenize(dialog['caption'],tokenizer)] + [tokenize(dialog['summary'],tokenizer)]
        questions = [tokenize(d['question'],tokenizer) for d in dialog['dialog']]
        answers = [tokenize(d['answer'],tokenizer) for d in dialog['dialog']]
        vid = dialog["image_id"]
        vid_set.add(vid)
        if undisclosed_only:
            it = range(len(questions) - 1, len(questions))
        else:
            it = range(len(questions))
        qalist=[]
        history = []
        if undisclosed_only:
            for n in range(len(questions)-1):
                qalist.append(questions[n])
                qalist.append(answers[n])
            history=qalist[max(-len(qalist),-n_history*2):]
        for n in it:
            if undisclosed_only:
                assert dialog['dialog'][n]['answer'] == '__UNDISCLOSED__'
            question = questions[n]
            answer = answers[n]
            history.append(question)
            if n_history == 0:
                item = {'vid': vid, 'history': [question], 'answer': answer, 'caption': caption}
            else:
                item = {'vid': vid, 'history': history, 'answer': answer, 'caption': caption}
            dialog_list.append(item)
            qalist.append(question)
            qalist.append(answer)
            history=qalist[max(-len(qalist),-n_history*2):]

    all_features = {}
    if feature_path is not None:
        fea_types = ['vggish', 'i3d_flow', 'i3d_rgb']
        dataname = '<FeaType>/<ImageID>.npy'
        for ftype in fea_types:
            if undisclosed_only:
                basename = dataname.replace('<FeaType>', ftype+'_testset')
            else:
                basename = dataname.replace('<FeaType>', ftype)
            features = {}
            for vid in vid_set:
                filename = basename.replace('<ImageID>', vid)
                filepath = feature_path + filename
                features[vid] = (filepath, filepath)
            all_features[ftype] = features
        return dialog_list, all_features

    return dialog_list


class AVSDDataSet(Dataset):
    def __init__(self, dialogs, tokenizer, features=None, drop_rate=0.5, train=True):
        self.dialogs = dialogs
        self.features = features
        self.tokenizer = tokenizer
        self.drop_rate = drop_rate
        self.train = train

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):
        dialog = self.dialogs[index]
        vid = dialog['vid']
        his = self.dialogs[index]['history']
        cap = self.dialogs[index]['caption']
        ans = self.dialogs[index]['answer']
        
        if np.random.rand() < self.drop_rate:
            instance, _ = build_input_from_segments(cap, his, ans, self.tokenizer, video=False, drop_caption=True, train=self.train)
        else:
            instance, _ = build_input_from_segments(cap, his, ans, self.tokenizer, video=False, drop_caption=False, train=self.train)
        input_ids = torch.Tensor(instance["input_ids"]).long()
        token_type_ids = torch.Tensor(instance["token_type_ids"]).long()
        lm_labels = torch.Tensor(instance["lm_labels"]).long()

        if self.features is not None:
            try:
                vgg = np.load(self.features[0]["vggish"][vid][0])
                i3d_flow = np.load(self.features[0]["i3d_flow"][vid][0])
                i3d_rgb = np.load(self.features[0]["i3d_rgb"][vid][0])
            except KeyError:
                vgg = np.load(self.features[1]["vggish"][vid][0])
                i3d_flow = np.load(self.features[1]["i3d_flow"][vid][0])
                i3d_rgb = np.load(self.features[1]["i3d_rgb"][vid][0])

            sample_i3d_flow = i3d_flow[range(1, i3d_flow.shape[0], 1)]
            sample_i3d_rgb = i3d_rgb[range(1, i3d_rgb.shape[0], 1)]

            vgg = torch.from_numpy(vgg).float()
            i3d_flow = torch.from_numpy(sample_i3d_flow).float()
            i3d_rgb = torch.from_numpy(sample_i3d_rgb).float()
            min_length = min([i3d_flow.size(0), i3d_rgb.size(0), vgg.size(0)])
            i3d = torch.cat([i3d_flow[:min_length], i3d_rgb[:min_length], vgg[:min_length]], dim=1)

            return input_ids, token_type_ids, lm_labels, i3d
        else:
            return input_ids, token_type_ids, lm_labels


def collate_fn(batch, pad_token, features=None):
    def padding(seq, pad_token):
        max_len = max([i.size(0) for i in seq])
        if len(seq[0].size()) == 1:
            result = torch.ones((len(seq), max_len)).long() * pad_token
        else:
            result = torch.ones((len(seq), max_len, seq[0].size(-1))).float()
        for i in range(len(seq)):
            result[i, :seq[i].size(0)] = seq[i]
        return result

    input_ids_list, token_type_ids_list, lm_labels_list, i3d_list = [], [], [], []
    for i in batch:
        input_ids_list.append(i[0])
        token_type_ids_list.append(i[1])
        lm_labels_list.append(i[2])
        if features is not None:
            i3d_list.append(i[3])

    input_ids = padding(input_ids_list, pad_token)
    token_type_ids = padding(token_type_ids_list, pad_token)
    lm_labels = padding(lm_labels_list, -1)
    input_mask = input_ids != pad_token
    if features is not None:
        i3d = padding(i3d_list, pad_token)
        i3d_mask = torch.sum(i3d != 1, dim=2) != 0
        input_mask = torch.cat([i3d_mask, input_mask], dim=1)
        i3d_labels = torch.ones((i3d.size(0), i3d.size(1))).long() * -1
        video_mask = torch.cat([torch.zeros((i3d.size(0), i3d.size(1))), torch.ones(lm_labels.size())], 1)
        reply_mask = torch.zeros(video_mask.size())
        lm_labels = torch.cat([i3d_labels, lm_labels], dim=1)
        return input_ids, token_type_ids, lm_labels, input_mask, i3d, video_mask, reply_mask
    else:
        return input_ids, token_type_ids, lm_labels, input_mask


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def build_input_from_segments(caption, history, reply, tokenizer, with_eos=True, video=False, drop_caption=False, train=True):
    """ Build a sequence of input from 3 segments: caption(caption+summary) history and last reply """
    bos, eos, speaker1, speaker2, cap = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-2])
    if not drop_caption:
        instance = {}
        sequence = [[bos] + list(chain(*caption))] + history + [reply + ([eos] if with_eos else [])]
        sequence = [[cap] + sequence[0] + [eos]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [cap] * len(sequence[0]) + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
        if video and train:
            #instance["lm_labels"] = sequence[0] + ([-1]*sum(len(s) for s in sequence[1:-1])) + sequence[-1]
            instance["lm_labels"] = sequence[0] + ([-1]*sum(len(s) for s in sequence[1:-1])) + sequence[-1]
        else:
            instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]
    else:
        instance = {}
        sequence = history + [reply + ([eos] if with_eos else [])]
        sequence = [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence)]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        if video:
            instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]
        else:
            instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]

    return instance, sequence


