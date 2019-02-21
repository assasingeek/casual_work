# You will find that if a word whose probability is to be measured is not in the 
# traing data set then the model will fail to predict correctly.
# This only means that we need a much bigger data set for model to learn more spanish and english words
# The spanish sentences and phrases has been taken from :
# "https://www.iwillteachyoualanguage.com/learn/spanish/spanish-tips/common-spanish-phrases"
# Much of the code is from pytorch documentation

import logging as lg
lg.basicConfig(level=lg.INFO, format='%(levelname)-8s: %(message)s')
import torch
import torch.optim as optim

torch.manual_seed(1)

data = [("Buenos días Buenos".split(), "SPANISH"),
        ("Good morning".split(), "ENGLISH"),
        ("Cómo estás".split(), "SPANISH"),
        ("How are you".split(), "ENGLISH"),
        ("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH"),
        ("Cómo te va".split(), "SPANISH"),
        ("How’s it going".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH"),
             ("Buenos provecho".split(), "SPANISH"),
             ("Bon appetit".split(), "ENGLISH")]

# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


class BoW(torch.nn.Module):

    def __init__(self, num_labels, vocab_size):
        super().__init__()
        self.linear = torch.nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return torch.nn.functional.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    # print(vec.shape)
    # exit(0)
    return vec.view(1, -1)

def make_target(label, label_ix):
    return torch.LongTensor([label_ix[label]])


model = BoW(NUM_LABELS, VOCAB_SIZE)

# for param in model.parameters():
#     print(param)

label_ix = {"SPANISH": 0, "ENGLISH": 1}

loss_function = torch.nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


loss_collection = []
for epoch in range(100):
    for instance, label in data:

        model.zero_grad()

        # print(instance)
        # print(label)
        # exit(0)

        bow_vec = make_bow_vector(instance, word_to_ix)
        # print(bow_vec.shape)
        # exit(0)
        target = make_target(label, label_ix)
        # print(target)

        log_probs = model(bow_vec)
        print(log_probs)
        # exit(0)

        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        loss_collection.append(loss.item())
    # lg.info('Epoch:%02d loss:%8.4f',
    #     epoch, torch.tensor(loss_collection).mean().item())


# i=0
# for loss in losses:
#     print('Instance :',i,loss)
#     i=i+1


# with torch.no_grad():
#     for instance, label in test_data:
#         bow_vec = make_bow_vector(instance, word_to_ix)
#         log_probs = model(bow_vec)
#         print(log_probs)

#Testing the model
# Index corresponding to Spanish goes up, English goes down
# print('only first time')
print(next(model.parameters())[:, word_to_ix["Buenos"]])

# One can change the word - "" - in the quotes for the probability.