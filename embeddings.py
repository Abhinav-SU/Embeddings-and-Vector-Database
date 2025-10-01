#Embeddings
'''
it is way to reprensent data such as word and 
text in numerical format embeddings are dense 
vectors that charecterize meaningful info about 
object they encode
'''
#Word Embeddings
'''
Captures semantic meaning of words 
words that are sematically similar in natural 
language have embeddings that are similar ot each other
'''

'''
Here we will use spacy for word vectors
we will use the en_core_web_md model of spacy that 
provide us with word embeddings
it includes 20,000 pretrained word embeddings and has 300 dim
'''


import spacy
from cosine_similarity import compute_cosine_similarity
nlp = spacy.load("en_core_web_md")
dog_embedding = nlp.vocab["dog"].vector
cat_embedding = nlp.vocab["cat"].vector
apple_embedding = nlp.vocab["apple"].vector
truck_embedding = nlp.vocab["truck"].vector
print(compute_cosine_similarity(dog_embedding,cat_embedding))
print(compute_cosine_similarity(dog_embedding,apple_embedding))
print(compute_cosine_similarity(dog_embedding,truck_embedding))
'''
print(type(dog_embedding))
print(dog_embedding.shape)
print(dog_embedding[0:10])
'''


'''
First we import spacy then we load medium English model
into object called nlp , we look embedding using 
nlp.vocab["dog"].vector and store in dog vector , 
then we check its type , dimension and first 10 values
'''

'''nlp.vocab object is used to find word embedding for any words'''


#Text Embeddings

'''
encode info about texts , allows to compare large body of text 
most efficient way to generate text embedding is to use pretrained model
Sentence transformers liabrary is one tool to use for this
'''

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
texts=[
    "The canine barked loudly.",
    "The dog made a noisy bark.",
    "He ate a lot of pizza",
    "He devoured a large quantity of pizza pie",
]

text_embeddings =model.encode(texts)

print(type(text_embeddings))
print(text_embeddings.shape)

text_embeddings_dict = dict(zip(texts,list(text_embeddings)))
dog_text_1 ="The canine barked loudly."
dog_text_2 ="The dog made a noisy bark."
print(compute_cosine_similarity(text_embeddings_dict[dog_text_1],
                          text_embeddings_dict[dog_text_2]))

pizza_text_1 ="He ate a lot of pizza"
pizza_text_2 ="He devoured a large quantity of pizza pie"
print(compute_cosine_similarity(text_embeddings_dict[pizza_text_1],
                                text_embeddings_dict[pizza_text_2]))

print(compute_cosine_similarity(text_embeddings_dict[dog_text_1],
                                text_embeddings_dict[pizza_text_2]))