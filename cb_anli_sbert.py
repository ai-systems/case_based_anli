import msgpack
import json
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from case_based_modules.retrieve.sentence_transformer import SentenceEmbedding
from case_based_modules.utils import Utils
from case_based_modules.retrieve.relevance_score import RelevanceScore
from case_based_modules.reuse.explanatory_power import ExplanatoryPower
from case_based_modules.refine.multi_hop_inference import ExplanationGraph

# Load table-store
with open("data/cache/table_store.mpk", "rb") as f:
    ts_dataset = msgpack.unpackb(f.read(), raw=False)

# Load train and dev set
with open("data/cache/eb_train.mpk", "rb") as f:
    eb_dataset_train = msgpack.unpackb(f.read(), raw=False)

with open("data/cache/eb_dev.mpk", "rb") as f:
    eb_dataset_dev = msgpack.unpackb(f.read(), raw=False)

with open("data/cache/eb_test.mpk", "rb") as f:
    eb_dataset_test = msgpack.unpackb(f.read(), raw=False)

#load folds
with open('data/cache/fold_train.json') as f:
    fold_train = json.load(f)

with open('data/cache/fold_dev.json') as f:
    fold_dev = json.load(f)

with open('data/cache/fold_test.json') as f:
    fold_test = json.load(f)

#load hypotheses
with open('data/cache/hypothesis_train.json') as f:
    hypothesis_train = json.load(f)

with open('data/cache/hypothesis_dev.json') as f:
    hypothesis_dev = json.load(f)

with open('data/cache/hypothesis_test.json') as f:
    hypothesis_test = json.load(f)

# Parameters
K = 5000  # relevance facts limit
Q = 40 # similar questions limit
QK = 5000 # explanatory power facts limit
M = 45 #number of facts considered for grounding/abstraction
N = 40 #number of facts considered for central explanation
CF = 1 #number of central facts to consider for the final inference
weights = [0.97, 0.03] # relevance and explanatory power weigths
grounding_tables = ["KINDOF","SYNONYMY", "OPPOSITES","INTENSIVE-EXTENSIVE"]
not_central_tables = grounding_tables
eb_dataset = eb_dataset_test # test dataset
hypotheses_dataset = hypothesis_test # test hypotheses
fold_dataset = fold_test #test fold

facts_retriever = SentenceEmbedding()  # facts retrieval model
cases_retriever = facts_retriever # cases retrieval model
utils = Utils() #utils
utils.init_explanation_bank_lemmatizer() #lemmatizer

corpus = []
original_corpus = []
question_train = []
ids = []
q_ids = []
tables = []
entities_dict = {}  #entities dictionary for the facts
# fitting the models and extract entities (concepts)
for t_id, ts in tqdm(ts_dataset.items()):
    # facts lemmatization
    if "#" in ts["_sentence_explanation"][-1]:
        fact = ts["_sentence_explanation"][:-1]
    else:
        fact = ts["_sentence_explanation"]
    lemmatized_fact = []
    original_corpus.append(fact)
    for chunck in fact:
        temp = []
        for word in nltk.word_tokenize(
            chunck.replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(";", " ")
            .replace("-", " ")
        ):
            temp.append(word.lower())
        if len(temp) > 0:
            lemmatized_fact.append(" ".join(temp))
    corpus.append(lemmatized_fact)
    ids.append(t_id)
    #entities (concepts) extraction
    clean_fact = utils.clean_fact_for_overlaps(ts["_explanation"])
    entities_dict[t_id] = list(set(utils.recognize_entities(clean_fact)))

for q_id, exp in tqdm(eb_dataset_train.items()):
    # concatenate question with candidate answer as the hypothesis
    if exp["_answerKey"] in exp["_choices"]:
        question = hypothesis_train[q_id][exp["_answerKey"]]
    question_train.append(question)
    q_ids.append(q_id)

#setup the dense models
facts_retriever.fit(corpus, question_train, ids, q_ids, "sentence-transformers/bert-large-nli-stsb-mean-tokens")

RS = RelevanceScore(facts_retriever)
PW = ExplanatoryPower(cases_retriever, eb_dataset_train)

total = 0
correct = 0
total_easy = 0
correct_easy = 0
total_challenge = 0
correct_challenge = 0
accuracy_explanation = 0
precision_explanation = 0 
recall_explanation = 0
f1_explanation = 0
res = {}

#Case-Based Abductive NLI
for q_id, exp in tqdm(eb_dataset.items()):
    score_hypotheses= {}
    best_central_choice_id = {}
    best_central_choice_all_ids = {}
    best_central = {}
    best_abstractions = {}

    if not exp["_answerKey"] in exp["_choices"]:
        continue

    res[q_id] = {}
    ranking_correct_answer = {}

    # EXPLANATION GENERATION
    for choice in exp["_choices"]:
        # concatenate question with candidate answer to get the hypothesis
        hypothesis = hypotheses_dataset[q_id][choice]
        hypothesis_entities = list(set(utils.recognize_entities(hypothesis)))

        # RETRIEVE
        relevance_scores = RS.compute(hypothesis, K)
        explanatory_power_scores = PW.compute(q_id, hypothesis, Q, QK)

        # REUSE
        explanatory_relevance = {}
        for t_id, ts in ts_dataset.items():
            if t_id in relevance_scores.keys() and t_id in explanatory_power_scores.keys():
                explanatory_relevance[t_id] = weights[0] * relevance_scores[t_id] + weights[1] * explanatory_power_scores[t_id]
            elif t_id in relevance_scores.keys():
                explanatory_relevance[t_id] =  weights[0] * relevance_scores[t_id]
            elif t_id in explanatory_power_scores.keys():
                explanatory_relevance[t_id] = weights[1] * explanatory_power_scores[t_id]
            else:
                explanatory_relevance[t_id] = 0

        # REFINE
        explanation_graph = ExplanationGraph(grounding_tables, not_central_tables, entities_dict, ts_dataset)
        #perform abstraction step
        abstractions = explanation_graph.abstraction(hypothesis_entities, relevance_scores, M)
        #perform central explanation step
        best_central_ids, score_hypotheses[choice] = explanation_graph.central_explanation(abstractions, explanatory_relevance, N, CF)

        #compute the final hypothesis score and explanation and cache the results
        if best_central_ids != None:
            best_central[choice] = []
            best_central_choice_id[choice] = best_central_ids[0]
            index = 0
            res[q_id][choice] = []
            for central in best_central_ids:
                best_central[choice].append(utils.clean_fact(ts_dataset[central]["_explanation"]))
                index += 1
            res[q_id][choice] = best_central_ids
        else:
            best_central[choice] = []
            res[q_id][choice] = []

    # ABDUCTIVE INFERENCE (Select as an answer the hypothesis with the best score)
    answer = ""
    for choice_id in sorted(score_hypotheses, key=score_hypotheses.get, reverse=True):
        answer = choice_id
        break

    # EVALUATION
    #check if the answer is correct
    if answer == exp["_answerKey"]:
        correct += 1
        if fold_dataset[q_id] == "Easy":
            correct_easy += 1
        else:
            correct_challenge += 1
    #count easy and challenge questions
    if fold_dataset[q_id] == "Easy":
        total_easy += 1
    else:
        total_challenge += 1
    total += 1

    #compute the evaluation metrics
    correct_explanation = False
    if eb_dataset == eb_dataset_dev:
        curr_precision = 0
        curr_recall = 0
        if best_central_choice_id[answer] in eb_dataset[q_id]["_explanation"]:
            accuracy_explanation += 1
            correct_explanation = True
        for fact in res[q_id][answer]:
            if fact in eb_dataset[q_id]["_explanation"]:
                curr_precision +=1
                curr_recall +=1
        curr_precision = curr_precision/len(res[q_id][answer])
        if len(eb_dataset[q_id]["_explanation"]) > 0:
            curr_recall = curr_recall/len(eb_dataset[q_id]["_explanation"])
        else:
            curr_recall = 1
        precision_explanation += curr_precision
        recall_explanation += curr_recall
        f1_explanation += 0.5*curr_precision + 0.5*curr_recall

    print("accuracy", correct/total)

#print the final results    
print("accuracy", correct/total)
print("accuracy easy", correct_easy/total_easy)
print("accuracy challenge", correct_challenge/total_challenge)
print("best explanation accuracy:", accuracy_explanation/(total))
print("explanation precision:", precision_explanation/(total))
print("explanation recall:", recall_explanation/(total))
print("explanation f1:", f1_explanation/(total))

#save evidence
json.dump(res,  open('res.json', 'w'), indent = 3)
