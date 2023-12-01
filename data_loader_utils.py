import numpy as np
def story_preprocess_function(examples, tokenizer):
    """This preprocess function combines all the sentences together and outputs the story that is plausible"""
    story_1 = [' '.join(stories[0]['sentences']) for stories in examples['stories']]
    story_2 = [' '.join(stories[1]['sentences']) for stories in examples['stories']]
    num_data = len(examples['stories'])
    tokenized_story = tokenizer(story_1 + story_2)
    return {k: [[v[i], v[i+num_data]] for i in range(num_data)] for k, v in tokenized_story.items()}


def sentence_preprocess_function(examples, tokenizer):
    """This preprocess function combines all the sentences together and outputs the story that is plausible"""
    ##  ----------------1st part: tokenize the sentences-------------------------------
    # unflatten
    story_1 = [stories[0]['sentences'] for stories in examples['stories']]
    story_2 = [stories[1]['sentences'] for stories in examples['stories']]
    num_sentences_1 = [len(this_story) for this_story in story_1]
    num_sentences_2 = [len(this_story) for this_story in story_2]
    story_1 = sum(story_1, [])
    story_2 = sum(story_2, [])
    num_data = len(examples['stories'])
    total_num_sentences_1 = len(story_1)
    tokenized_story = tokenizer(story_1 + story_2)

    # flatten        
    tokenized_results = {}
    for key in tokenized_story.keys():
        ctr_1 = 0
        ctr_2 = 0
        values = []
        for num_1, num_2 in zip(num_sentences_1, num_sentences_2):
            values.append([tokenized_story[key][ctr_1: ctr_1+num_1], \
            tokenized_story[key][total_num_sentences_1+ctr_2: total_num_sentences_1+ctr_2+num_2]])
            ctr_1 += num_1
            ctr_2 += num_2

        tokenized_results[key] = values
    # ------------------------2nd part: reorganize the data-----------------------------------
    story_labels = examples['label']
    confl_pairs = examples['confl_pairs']
    arranged_feature = arrange_feature(tokenized_results)
    # fine the label
    sentence_labels = []
    # flattened_feature_for_comparison = [{k: self.flatten_sentences(v) for k, v in feature.items()} for feature in features]
    for i, (story_label, confl_pair, tokenized_input_ids, arranged_input_ids) \
    in enumerate(zip(story_labels, confl_pairs, tokenized_results['input_ids'], arranged_feature['input_ids'])):
        confl_sent_pair = tokenized_input_ids[story_label][confl_pair[0][0]] + tokenized_input_ids[story_label][confl_pair[0][1]]
        for j in range(len(arranged_input_ids)):
            if len(arranged_input_ids[j]) == len(confl_sent_pair):
                if (np.array(arranged_input_ids[j]) == np.array(confl_sent_pair)).all():
                    if story_label == 0:
                        assert j < 10
                    else:
                        assert j >= 10
                    sentence_labels.append(j)
    tokenized_results['arranged_confl_labels'] = sentence_labels
    if 'token_type_ids' in tokenized_results.keys():
        tokenized_results['token_type_ids'] = arranged_feature['token_type_ids']
    tokenized_results['input_ids'] = arranged_feature['input_ids']
    tokenized_results['attention_mask'] = arranged_feature['attention_mask']
    return tokenized_results
def pair2label(pair, num_sentences):
    # returns the index of the conflict pair
    assert pair[0] < pair[1]
    num = 0
    for i in range(1, pair[0] + 1):
        num += (num_sentences - i)
    num += (pair[1] - pair[0] - 1) 
    return num
# def arrange_feature(input_ids, token_type_ids, attention_mask):
#     # keys = feature.keys()
#     input_ids_list = []
#     token_type_ids_list = []
#     attention_mask_list = []

#     # input_ids, attention_mask = feature['input_ids'], feature['attention_mask']
#     for i in range(2): # loop over stories
#         assert len(input_ids[i]) == 5
#         for j in range(len(input_ids[i])): # find the [i,j] sentence pair
#             for k in range(j+1, len(input_ids[i])):
#                 input_ids_list.append(input_ids[i][j] + input_ids[i][k])
#                 attention_mask_list.append(attention_mask[i][j] + attention_mask[i][k])
#                 # results.append({'input_ids': input_ids[i][j] + input_ids[i][k], 'attention_mask': attention_mask[i][j] + attention_mask[i][k]})
#     results = {'input_ids': input_ids_list, 'attention_mask': attention_mask_list}    
#     return results
def arrange_feature(tokenized_results):
    # keys = feature.keys()
    results = {}
    for key in tokenized_results.keys():
        results[key] = []

    for data_idx in range(len(tokenized_results['input_ids'])):
        temp_dict = {}
        for key in tokenized_results.keys():
            temp_dict[key] = []
        for i in range(2): # loop over stories
            assert len(tokenized_results['input_ids'][data_idx][i]) == 5
            for j in range(5): # find the [i,j] sentence pair
                for k in range(j+1, 5):
                    for key in tokenized_results.keys():
                        temp_dict[key].append(tokenized_results[key][data_idx][i][j] + tokenized_results[key][data_idx][i][k])
        for key in tokenized_results.keys():
            results[key].append(temp_dict[key])

    return results