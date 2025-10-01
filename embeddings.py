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