# from transformers import pipeline, AutoTokenizer, AutoModel
# import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# from numpy import dot
# from numpy.linalg import norm

# def mycos(x,y):
#     return dot(x, y)/(norm(x)*norm(y))

# tokenizer = AutoTokenizer.from_pretrained("Shushant/nepaliBERT")
# # model = AutoModel.from_pretrained("Shushant/thesis_nepaliGPT")
# model = SentenceTransformer('Shushant/nepaliBERT')
# mypipe = pipeline('feature-extraction', model)

# embedding1 = mypipe("यो समान धेरै राम्रो छ ")
# embedding2 = mypipe("सारै खराब समान रहेछ ")
import numpy as np

model = SentenceTransformer('Shushant/nepaliBERT',  device='cpu')

def find_embedding(text):
      return model.encode(text)

# message_emb = np.array([find_embedding(prompt['content']) for prompt in messageDB]).reshape(-1)

input_str = 'पराजय'
input_emb = find_embedding(input_str)
another_emb = find_embedding("हार")


similarity = cosine_similarity([input_emb], [another_emb])[0][0]
print(similarity)
# print(mycos(embedding1,embedding2))
# embedding1 = model.encode("पराजय").reshape(1,-1)
# embedding2 = model.encode("हार").reshape(1,-1)
# print(embeddings)



# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model.resize_token_embeddings(len(tokenizer))

# pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

# data = pipe("राम्रो")


# def get_bert_embedding_sentence(input_sentence):
#     md = model
#     # tokenizer = tokenizer
# #     md = local_model
# #     tokenizer = local_tokenizers
#     marked_text = " [CLS] " + input_sentence + " [SEP] "
#     tokenized_text = tokenizer.tokenize(marked_text)

#     indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
#     segments_ids = [1] * len(indexed_tokens) 
    
#     # Convert inputs to Pytorch tensors
#     tokens_tensors = torch.tensor([indexed_tokens])
#     segments_tensors = torch.tensor([segments_ids])
    
#     with torch.no_grad():
#         outputs = md(tokens_tensors, segments_tensors)
#         # removing the first hidden state
#         # the first state is the input state 

#         hidden_states = outputs.hidden_states
# #         print(hidden_states[-2])
#         # second_hidden_states = outputs[2]
#     # hidden_states has shape [13 x 1 x 22 x 768]

#     # token_vecs is a tensor with shape [22 x 768]
# #     token_vecs = hidden_states[-2][0]
#     # get last four layers
# #     last_four_layers = [hidden_states[i] for i in (-1,-2, -3,-4)]
#     # cast layers to a tuple and concatenate over the last dimension
# #     cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
# #     print(cat_hidden_states.shape)
#     token_vecs = hidden_states[-2][0]
#     # take the mean of the concatenated vector over the token dimension
# #     sentence_embedding = torch.mean(cat_hidden_states, dim=0).squeeze()

#     # Calculate the average of all 22 token vectors.
#     sentence_embedding = torch.mean(token_vecs, dim=0)
# #     sentence_embedding = torch.mean(token_vecs, dim=1)
#     return sentence_embedding.numpy()

# get_bert_embedding_sentence("आज पानि परिरहेको")