import torch
import torch.optim as optim
from tqdm import tqdm

import time
import json
from dataset.dataset import get_time_dif, convert_onehot
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from model.prompthate import *
from model.clip import *
from model.vit_roberta import *
from model.MHKE import *
from model.EAR import compute_negative_entropy


def train(config, model, train_iter, dev_iter):

    model_name = 'sub_step_{}_Lr-{}_Len-{}_entity-{}_race-{}_ear-{}_att-{}_tem-{}_mm-cot'.format(
        config.model_name, config.learning_rate, config.pad_size, config.if_entity, config.if_race, config.if_ear, config.if_att, config.template)
    print(model_name)

    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': config.weight_decay,},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0},
    # ]
    # model_optimizer = optim.AdamW(optimizer_grouped_parameters,
    #                         lr=config.learning_rate, eps=config.eps)

    model_optimizer = optim.AdamW(model.parameters(),
                                  lr=config.learning_rate, eps=config.eps)

    # loss_fn = nn.BCEWithLogitsLoss()
    max_score = 0

    for epoch in range(config.num_epochs):
        model.train()
        start_time = time.time()
        print("Model is training in epoch {}".format(epoch))

        # preds = []
        # labels = []
        loss_all = 0.
        step = 0
        f = open(
            '{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
        f.write(' ==================================================  Epoch: {}  ==================================================\n'.format(epoch))

        for batch in tqdm(train_iter, desc='Training', colour='MAGENTA'):
            model.zero_grad()
            output = model(**batch)
            loss = output["loss"].cpu()

            # logit = model.evaluate(**batch)
            # label = batch['label']
            # pred = get_preds(config, logit)

            if config.if_ear:
                negative_entropy = compute_negative_entropy(
                    output["encoder_attentions"], batch["prompt_attention_mask"]
                )
                reg_loss = config.ear_reg_strength * negative_entropy.cpu()
                loss += reg_loss

            # preds.extend(pred)
            # labels.extend(label.detach().numpy())

            loss_all += loss.item()
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            step += 1

        end_time = time.time()
        print(" took: {:.1f} min".format((end_time - start_time)/60.))
        print("TRAINED for {} epochs".format(epoch))

        # 验证
        if epoch >= config.num_warm:
            # print("training loss: loss={}".format(loss_all/len(data)))
            # trn_scores = get_scores(preds, labels, data_name="TRAIN")
            dev_scores, preds = eval(config, model, dev_iter, data_name='DEV')
            f.write('TrainLoss: \n{}\n'.format(
                json.dumps({"loss": loss_all/len(train_iter)})))
            f.write('EvalScore: \n{}\n'.format(json.dumps(dev_scores)))
            max_score = save_best(config, epoch, model_name,
                                  model, dev_scores, max_score, preds)
        print("ALLTRAINED for {} epochs".format(epoch))


def eval(config, model, dev_iter, data_name='DEV'):

    preds = []
    labels = []
    model.eval()
    
    for batch in tqdm(dev_iter, desc='Evaling', colour='CYAN'):
        with torch.no_grad():
            pred = model.generate(**batch)
            pred = get_preds(config, pred)
            label = batch['label'].detach().numpy()
            label = get_labels(config, label.tolist())
            preds.extend(pred)
            labels.extend(label)

    dev_scores = get_scores(preds, labels, data_name=data_name)
    return dev_scores, preds


# def test(model, dev_iter):

#     preds = []
#     labels = []

#     for batch in tqdm(dev_iter, desc='Testing', colour='CYAN'):
#         with torch.no_grad():
#             pred = model.evaluate(**batch)
#             label = batch['label']
#             preds.extend(pred)
#             labels.extend(label.detach().numpy())

#         df = pd.DataFrame({'new_pred': preds})
#         output_file = 'preds.csv'
#         df.to_csv(output_file, index=False)

#     return preds


def get_preds(config, preds):
    new_preds = []
    labels = list(config.label_list.values())
    for pred in preds:
        if pred not in labels:
            new_preds.append(labels[0])
        else:
            new_preds.append(pred)
    return new_preds


def get_labels(config, labels):
    new_labels = [config.label_list[label[1]] for label in labels]
    return new_labels


def output_preds(logit):
    results = torch.max(logit.data, 1)[1].cpu().numpy()
    new_results = []
    for result in results:
        new_results.append(result)
    return new_results


def get_scores(all_preds, all_lebels, data_name):
    score_dict = dict()
    f1 = f1_score(all_lebels, all_preds, average='macro')
    acc = accuracy_score(all_lebels, all_preds)
    all_f1 = f1_score(all_lebels, all_preds, average=None)
    pre = precision_score(all_lebels, all_preds, average='macro')
    recall = recall_score(all_lebels, all_preds, average='macro')

    score_dict['F1'] = f1
    score_dict['accuracy'] = acc
    score_dict['all_f1'] = all_f1.tolist()
    score_dict['precision'] = pre
    score_dict['recall'] = recall

    # score_dict['all_loss'] = loss_all/len
    print("Evaling on \"{}\" data".format(data_name))
    for s_name, s_val in score_dict.items():
        print("{}: {}".format(s_name, s_val))
    return score_dict


def save_best(config, epoch, model_name, model, score, max_score, preds):
    score_key = config.score_key
    curr_score = score[score_key]
    print('The epoch_{} {}: {}\nCurrent max {}: {}'.format(
        epoch, score_key, curr_score, score_key, max_score))

    if curr_score > max_score or epoch == 0:
        torch.save({
            'epoch': config.num_epochs,
            'model_state_dict': model.state_dict(),
        }, '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST'))
        
        save_path = '{}/{}.txt'.format(config.prediction_path, model_name)
        with open(save_path, 'w', encoding='utf-8') as file:
            for pred in preds:
                file.write(f"{pred}\n")        
        return curr_score
    else:
        return max_score
