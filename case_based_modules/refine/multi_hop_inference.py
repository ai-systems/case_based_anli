from case_based_modules.utils import Utils

class ExplanationGraph():
    
    def __init__(self, abstraction_tables:[str], not_central_tables:[str], entities_dict:dict, fact_KB):
        self.abstraction_tables = abstraction_tables
        self.not_central_tables = not_central_tables
        self.entities_dict = entities_dict
        self.FKB = fact_KB
        self.utils = Utils()
        self.abstractions_facts = {}

    def abstraction(self, hypothesis_entities, ranking:dict, limit:int):
        abstractions = {}
        self.abstractions_ranking = ranking
        for entity in hypothesis_entities:
            abstractions[entity] = [entity]
            self.abstractions_facts[entity] = []
        for fact_id in sorted(ranking, key=ranking.get, reverse=True)[:limit]:
            if ranking[fact_id] > 0:
                #perform abstraction step
                if self.FKB[fact_id]["_table_name"].replace("PROTO-","") in self.abstraction_tables:
                    for entity in set(self.entities_dict[fact_id]).intersection(set(hypothesis_entities)):
                        abstractions[entity] += self.entities_dict[fact_id]
                        abstractions[entity] = list(set(abstractions[entity]))
                        self.abstractions_facts[entity].append(fact_id)
        return abstractions

    def central_explanation(self, abstractions:dict, explanatory_relevance:dict, limit:int, num_central:int):
        central_score = {}
        #compute the scores of central explanation facts
        for fact_id in sorted(explanatory_relevance, key=explanatory_relevance.get, reverse=True)[:limit]:
            if explanatory_relevance[fact_id] > 0 and not self.FKB[fact_id]["_table_name"].replace("PROTO-","") in self.not_central_tables:
                #compute semantic plausibility
                paths = 0
                semantic_plausibility = 0
                for entity in abstractions:
                    if len(set(self.entities_dict[fact_id]).intersection(set(abstractions[entity]))) > 0:
                        paths += 1
                if paths > 0:
                    semantic_plausibility = paths/len(abstractions)
                central_score[fact_id] = semantic_plausibility + explanatory_relevance[fact_id] 

        num = 1
        final_score = 0
        explanation = []
        for fact_id in sorted(central_score, key=central_score.get, reverse=True):
            explanation.append(fact_id)
            final_score += central_score[fact_id]
            if num == num_central:
                return explanation, final_score
            num += 1
        return None, 0