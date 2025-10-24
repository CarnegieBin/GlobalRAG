import re
import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
from sentence_transformers import SentenceTransformer, util

# from flashrag.evaluator.metrics import BaseMetric, F1_Score

from verl.utils.reward_score.pse.flashrag.evaluator.metrics import BaseMetric, F1_Score



class Plan_Score(BaseMetric):
    """Token-level F1 score"""

    metric_name = "p"

    def __init__(self, config):
        super().__init__(config)
        model_path = "/Users/wanfan01/Downloads/e5"
        model_path = config['e5_model_path']
        self.sim_model = SentenceTransformer(model_path)

    def dict2graph(self, graph_dict):
        graph = nx.DiGraph()
        pattern = r"<A(\d+)>"

        for nid, node in graph_dict.items():
            graph.add_node(nid, template=node[0])
            template = node[0]
            placeholders = re.findall(pattern, template)
            for placeholder in placeholders:
                graph.add_edge(f"Q{placeholder}", nid)

        return graph
    
    def template2embedding(self, graph):
        pattern = r"<A(\d+)>"
        nids = [nid for nid in graph.nodes()]
        templates = [attr["template"] for _, attr in graph.nodes(data=True)]
        templates_ = []
        for template in templates:
            placeholders = re.findall(pattern, template)
            for placeholder in placeholders:
                template = template.replace(f"<A{placeholder}>", f"entity")
            templates_.append(template)
        embeddings = self.sim_model.encode(templates_)
        return nids, embeddings
    
    def graph_edit_distance(self, mapping, G_golden, G_pred, beta=0.1):

        G_golden_nodes_relabel = {}
        G_pred_nodes_relabel = {}
        common_nid_suffix = 1
        for key, value in mapping.items():
            G_golden_nodes_relabel[key] = f"C{common_nid_suffix}"
            G_pred_nodes_relabel[value] = f"C{common_nid_suffix}"

            common_nid_suffix += 1

        G_golden_nodes_relabel.update({node: f"Qg{idx+1}" for idx, node in enumerate(G_golden.nodes()) if node not in G_golden_nodes_relabel.keys()})
        G_pred_nodes_relabel.update({node: f"Qp{idx+1}" for idx, node in enumerate(G_pred.nodes()) if node not in G_pred_nodes_relabel.keys()})
        G_golden_relabel = nx.relabel_nodes(G_golden, G_golden_nodes_relabel)
        G_pred_relabel = nx.relabel_nodes(G_pred, G_pred_nodes_relabel)

        a = set(G_pred_relabel.nodes()).union(set(G_golden_relabel.nodes()))
        b = set(G_pred_relabel.nodes()).intersection(set(G_golden_relabel.nodes()))
        c = set(G_pred_relabel.edges()).union(set(G_golden_relabel.edges()))
        d = set(G_pred_relabel.edges()).intersection(set(G_golden_relabel.edges()))
        ged_score = len(a-b) + len(c-d)
        # print("GED Score:", ged_score)

        normalized_ged_score = np.exp(-beta * ged_score)
        return ged_score, round(normalized_ged_score, 6)
    
    def plan_score(self, average_graph_similarity, ged_score, alpha=0.8):
        return alpha * average_graph_similarity + (1 - alpha) * ged_score
    
    def get_predicted_subplan(self, G_pred, G_golden):
        matcher = DiGraphMatcher(G_pred, G_golden)
        H_pred_nodes = []
        for mapping in matcher.subgraph_isomorphisms_iter():
            H_pred_nodes += list(mapping.keys())
        H_pred_nodes = list(set(H_pred_nodes))
        # print("Subgraph nodes:", H_pred_nodes)
        H_pred = G_pred.subgraph(H_pred_nodes).copy()
        # print("Subgraph nodes with attributes:", H_pred.nodes(data=True))
        # print("Subgraph edges:", H_pred.edges(data=True))
        return H_pred

    def calculate_plan_score(self, predicted_plan, golden_plan, thereshold=0.7, alpha=0.9, beta=1.0):
        G_golden, G_pred = self.dict2graph(golden_plan), self.dict2graph(predicted_plan)
        # print("======")
        # print("Golden Plan Nodes with attributes:", G_golden.nodes(data=True))
        # print("Golden Plan Edges:", G_golden.edges(data=True), end="\n\n")
        # print("Predicted Plan Nodes with attributes:", G_pred.nodes(data=True))
        # print("Predicted Plan Edges:", G_pred.edges(data=True))
        # print("======", end="\n\n")


        mapping = {}
        average_graph_similarity = 0.0

        G_golden_zero_in_degree_nodes = [n for n, d in G_golden.in_degree() if d == 0]
        G_pred_zero_in_degree_nodes = [n for n, d in G_pred.in_degree() if d == 0]

        G_golden_nid, G_golden_embedding = self.template2embedding(G_golden)
        G_pred_nid, G_pred_embedding = self.template2embedding(G_pred)

        similarity = util.cos_sim(G_golden_embedding, G_pred_embedding).tolist()

        # print("G_golden Nodes with zero in-degree:", G_golden_zero_in_degree_nodes)
        # print("G_pred Nodes with zero in-degree:", G_pred_zero_in_degree_nodes)
        # print("Similarity matrix:", similarity)

        init_mapping = {}
        graph_similarity = []
        for node in G_golden_zero_in_degree_nodes:
            node_index = G_golden_nid.index(node)
       
            if max(similarity[node_index]) < thereshold:
                continue
            max_sim_index = similarity[node_index].index(max(similarity[node_index]))
            if G_pred_nid[max_sim_index] in G_pred_zero_in_degree_nodes:

                if G_pred_nid[max_sim_index] not in init_mapping.values():
                    init_mapping[node] = G_pred_nid[max_sim_index]
                    graph_similarity.append(similarity[node_index][max_sim_index])

        # print("Initial mapping based on zero in-degree nodes:", init_mapping)

        def map_subsequent_nodes(G_golden_node, G_pred_node):
            # print("Golden Node:", G_golden_node, "Predicted Node:", G_pred_node)

            for successor in G_golden.successors(G_golden_node):

                successor_index = G_golden_nid.index(successor)
                predicted_successors = list(G_pred.successors(G_pred_node))

                if len(predicted_successors) == 0:
                    continue
                predicted_successor_embeddings = [G_pred_embedding[G_pred_nid.index(n)] for n in predicted_successors]
                similarities = util.cos_sim(G_golden_embedding[successor_index], predicted_successor_embeddings).tolist()[0]

                if max(similarities) < thereshold:
                    continue

                max_sim_index = similarities.index(max(similarities))
                if len(list(G_golden.predecessors(successor))) != len(list(G_pred.predecessors(predicted_successors[max_sim_index]))):
                    continue

                if predicted_successors[max_sim_index] not in extra_mapping.values():
                    # print("successor:", successor, "predicted_successor:", predicted_successors[max_sim_index], end="\n\n")
                    extra_mapping[successor] = predicted_successors[max_sim_index]
                    graph_similarity.append(similarities[max_sim_index])
                    map_subsequent_nodes(successor, predicted_successors[max_sim_index])

        extra_mapping = {}
        for G_golden_node, G_pred_node in init_mapping.items():
            map_subsequent_nodes(G_golden_node, G_pred_node)

        mapping.update(init_mapping)
        mapping.update(extra_mapping)

        del_mapping_keys = []
        del_sim_indexes = []
        for G_golden_node, G_pred_node in mapping.items():
            predecessors = list(G_pred.predecessors(G_pred_node))
            if len(predecessors) > 1:
                for predecessor in predecessors:
                    if predecessor not in mapping.values():
                        del_mapping_keys.append(G_golden_node)
                        del_sim_indexes.append(list(mapping.keys()).index(G_golden_node))

        for key in del_mapping_keys:
            del mapping[key]

        graph_similarity = [round(score, 6) for idx, score in enumerate(graph_similarity) if idx not in del_sim_indexes]
        average_graph_similarity = sum(graph_similarity) / len(G_golden.nodes())
        ged_score, normalized_ged_score = self.graph_edit_distance(mapping, G_golden, G_pred, beta=beta)
        plan_score = round(self.plan_score(average_graph_similarity, normalized_ged_score, alpha=alpha), 6)


        return mapping, {
            "graph_similarity": graph_similarity, 
            "average_graph_similarity": average_graph_similarity,
            "ged_score": ged_score,
            "normalized_ged_score": normalized_ged_score,
            "plan_score": plan_score
            }
    

class Step_Score(BaseMetric):
    """Token-level F1 score"""

    metric_name = "p"

    def __init__(self, config):
        super().__init__(config)
        self.f1_evaluator = F1_Score(config)

    def calculate_max_f1(self, pred_step_answer, gold_step_answers):

        return max([self.f1_evaluator.token_level_scores(pred_step_answer, gold_step_answer)["f1"] for gold_step_answer in gold_step_answers])


    def calculate_step_score(self, final_graphs, incomplete_graphs, golden_graphs, mapping):

        if len(final_graphs) != 0:
            evaluate_graphs = [final_graph for final_graph, _ in final_graphs]
        elif len(incomplete_graphs) != 0:
            evaluate_graphs = incomplete_graphs
            mapping_ = {}
            for golden_node, predicted_node in mapping.items():
                if predicted_node in list(incomplete_graphs[0].keys()):
                    mapping_[golden_node] = predicted_node
            mapping = mapping_
        else:
            return {"step_score": 0.0}


        pred_eval_order = {}
        gold_eval_order = {}
        for golden_node, predicted_node in mapping.items():
            gold_eval_order[golden_node] = list(set([graph[golden_node]["answer"] for graph in golden_graphs]))
            pred_eval_order[predicted_node] = list(set([graph[predicted_node]["answer"] for graph in evaluate_graphs]))

        pred_step_scores = []
        for golden_node, predicted_node in mapping.items():
            gold_step_answers = gold_eval_order[golden_node]
            pred_step_answers = pred_eval_order[predicted_node]

            pred_step_score = [self.calculate_max_f1(pred_step_answer, gold_step_answers) for pred_step_answer in pred_step_answers]
            pred_step_score = sum(pred_step_score) / len(pred_step_score)
            pred_step_scores.append(pred_step_score)

        final_step_score = round(sum(pred_step_scores) / len(golden_graphs[0]), 6)
        return {"step_score": final_step_score}
