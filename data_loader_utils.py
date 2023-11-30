def story_preprocess_function(examples, tokenizer):
    """This preprocess function combines all the sentences together and outputs the story that is plausible"""
    story_1 = [' '.join(stories[0]['sentences']) for stories in examples['stories']]
    story_2 = [' '.join(stories[1]['sentences']) for stories in examples['stories']]
    num_data = len(examples['stories'])
    tokenized_story = tokenizer(story_1 + story_2)
    return {k: [[v[i], v[i+num_data]] for i in range(num_data)] for k, v in tokenized_story.items()}


# def sentence_preprocess_function(examples, tokenizer):
#     """This preprocess function combines all the sentences together and outputs the story that is plausible"""
#     # unflatten
#     story_1 = [stories[0]['sentences'] for stories in examples['stories']]
#     story_2 = [stories[1]['sentences'] for stories in examples['stories']]
#     num_sentences_1 = [len(this_story) for this_story in story_1]
#     num_sentences_2 = [len(this_story) for this_story in story_2]
#     story_1 = sum(story_1, [])
#     story_2 = sum(story_2, [])
#     num_data = len(examples['stories'])
#     total_num_sentences_1 = len(story_1)
#     tokenized_story = tokenizer(story_1 + story_2)

#     # flatten        
#     flattened_results = {}
#     for key in tokenized_story.keys():
#         ctr_1 = 0
#         ctr_2 = 0
#         values = []
#         for num_1, num_2 in zip(num_sentences_1, num_sentences_2):
#             values.append([tokenized_story[key][ctr_1: ctr_1+num_1], \
#             tokenized_story[key][total_num_sentences_1+ctr_2: total_num_sentences_1+ctr_2+num_2]])
#             ctr_1 += num_1
#             ctr_2 += num_2

#         flattened_results[key] = values
#     return flattened_results
# def pair2label(pair, num_sentences):
#     # returns the index of the conflict pair
#     assert pair[0] < pair[1]
#     num = 0
#     for i in range(1, pair[0] + 1):
#         num += (num_sentences - i)
#     num += (pair[1] - pair[0] - 1) 
#     return num
# def flatten_feature(feature):
#     keys = feature.keys()
#     results = []

#     input_ids, attention_mask = feature['input_ids'], feature['attention_mask']
#     for i in range(2): # loop over stories
#         for j in range(len(input_ids[i])): 
#             for k in range(j+1, len(input_ids[i])):
#                 results.append({'input_ids': input_ids[i][j] + input_ids[i][k], 'attention_mask': attention_mask[i][j] + attention_mask[i][k]})
    
#     return results
        # def flatten_sentences(self, sentences):
        #     result = []
        #     for story in sentences:
        #         for i, sentence in enumerate(story):
        #             for j in range(i, len(story)):
        #                 result.append([story[i] + story[j]])
        #     return result