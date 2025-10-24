
import re
import numpy as np


from verl.utils.reward_score.pse.evaluate import Plan_Score, Step_Score
from verl.utils.reward_score.pse.flashrag.evaluator.metrics import BaseMetric
from verl.utils.reward_score.pse.flashrag.config.config import Config


# from evaluate import Plan_Score, Step_Score
# from flashrag.evaluator.metrics import BaseMetric
# from flashrag.config.config import Config



class PseScore(BaseMetric):

    def __init__(self, config):
        self.config = config
        self.plan_evaluator = Plan_Score(self.config)
        self.step_evaluator = Step_Score(self.config)

    def calculate_final_score(self, plan_score, step_score):
        return (2 * plan_score * step_score) / (plan_score + step_score + 1e-6)

    def calculate_pse_score(self, pred_plan, gold_plan, pred_graph, gold_graph):
        mapping, plan_scores = self.plan_evaluator.calculate_plan_score(pred_plan, gold_plan)
        pred_graph.append("")
        pred_graph = [pred_graph]
        step_score = {'step_score': 0}
        try:
            step_score = self.step_evaluator.calculate_step_score(pred_graph, [], gold_graph, mapping)
        except:
            print('calcuLate step score error, graph not follow plan......')
        # pse_g_score = self.calculate_final_score(plan_scores["plan_score"], step_score["step_score"])

        graph_plan_sim_score = plan_scores['average_graph_similarity']
        graph_plan_structure_score = 0 if plan_scores['ged_score'] >= 1 else 1
        return graph_plan_sim_score, graph_plan_structure_score, step_score['step_score']


if __name__ == "__main__":
    config_dict={'e5_model_path': '/Users/wanfan01/Downloads/e5'}
    # generation_params = {
    #     "do_sample": False,
    #     "max_tokens": 1024,
    #     "temperature": 0.0,
    # }
    # config_dict["generation_params"] = generation_params
    # config_dict["generator_max_input_len"] = 5000

    # multihopqa_params = {
    #     "use_relevant_expert": False,
    #     "use_rewriter": False,
    #     "use_hyde": True
    # }
    # config_dict["multihopqa_params"] = multihopqa_params

    config = Config(config_dict=config_dict)
    pse_score = PseScore(config)

    # pred_plan = {'Q1': ['Who is Samuel "Sam" Ervin Beam?', '<A1>'], 'Q2': ['On what album is <A1>\'s cover of "Such Great Heights" featured?', '<A2>']}
    # gold_plan = {'Q1': ["Who performed the cover of 'Such Great Heights'?", '<A1>'], 'Q2': ["On what album is <A1>'s cover of 'Such Great Heights' featured?", '<A2>']}
    #
    # pred_graph = [{'Q1': {'answer': 'Mountain West Conference'}, 'Q2': {'answer': '1999'}}]
    # gold_graph = [{'Q1': {'answer': 'Iron & Wine'}, 'Q2': {'answer': 'Give Up'}}]
    # plan_scores, step_score, pse_g_score = pse_score.calculate_pse_score(pred_plan, gold_plan, pred_graph, gold_graph)
    # print("plan_scores:", plan_scores, "step_score:", step_score, "pse_g_score:", pse_g_score)
    #

    # print("---" * 20)
    # pred_plan = {'Q1': ['In which conference did the 2005 Air Force Falcons participate?', '<A1>'], 'Q2': ['In what year did <A1> begin operations?', '<A2>']}
    # gold_plan = {'Q1': ['Which conference did the 2005 Air Force Falcons participate in?', '<A1>'], 'Q2': ['In what year did <A1> begin operations?', '<A2>']}
    #
    # pred_graph = [{'Q1': {'answer': 'Mountain West Conference'}, 'Q2': {'answer': '1999'}}]
    # gold_graph = [{'Q1': {'answer': 'Mountain West Conference'}, 'Q2': {'answer': '1999'}}]
    # plan_scores, step_score = pse_score.calculate_pse_score(pred_plan, gold_plan, pred_graph, gold_graph)
    # print("plan_scores:", plan_scores, "step_score:", step_score)

    from numpy import array
    print("---" * 20)
    print_info = {'predict_plan': {
                'Q1': ["Who is the performer from 'Changes Two'?", '<A1>'],
                'Q2': ['What state is <A1> from?', '<A2>'],
                'Q3': ['What is the largest city in state <A2>?', '<A3>'],
                'Q4': ['Who won the IndyCar race in city <A3>?', '<A4>'],
                },
                  # 'predict_graph': [{
                  #      'Q1': {'answer': 'Dionne Bray Mingus'},
                  #      'Q2': {'answer': 'Oklahoma'},
                  #      'Q3': {'answer': 'Oklahoma City'},
                  #      'Q4': {'answer': 'Mario City'}
                  #   }
                  # ],

                 'predict_graph': [{

                  }],

                  'golden_plan': {
                      'Q1': array(['Who is the performer of Changes Two?', '<A1>'], dtype=object),
                      'Q2': array(['What city is <A1> from?', '<A2>'], dtype=object),
                      'Q3': array(['What city is both the largest city and the state capital of <A2>?', '<A3>'], dtype=object),
                      'Q4': array(['Who won the Indy car race in <A3>?', '<A4>'], dtype=object)
                  },

                  'golden_graph': [
                      {'Q1': {'answer': 'Charles Mingus', 'previous': array([], dtype=object), 'query': 'Who is the performer of Changes Two?', 'supports': array([{'contents': array(['Changes Two is an album by Charles Mingus. It was recorded on 27, 28, and 30 December 1974 at Atlantic Studios in New York City—the same sessions which resulted in Mingus\' album "Changes One". Accordingly, Atlantic Records initially released the record. In 1993, it was issued on CD by Rhino Records.'],
             dtype=object), 'title': 'Changes Two'}                                                                                                                                                                                                                                                                                         ],
      dtype=object), 'tag': '<A1>', 'template': 'Who is the performer of Changes Two?'},
                       'Q2': {'answer': 'Nogales', 'previous': array(['Q1'], dtype=object), 'query': 'What city is Charles Mingus from?', 'supports': array([{'contents': array(["Charles Mingus was born in Nogales, Arizona. His father, Charles Mingus Sr., was a sergeant in the U.S. Army. Mingus was largely raised in the Watts area of Los Angeles. His maternal grandfather was a Chinese British subject from Hong Kong, and his maternal grandmother was an African-American from the southern United States. Mingus was the third great-grandson of the family's founding patriarch who was, by most accounts, a German immigrant. His ancestors included German American, African American, and Native American."],
             dtype=object), 'title': 'Charles Mingus'}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ],
      dtype=object), 'tag': '<A2>', 'template': 'What city is <A1> from?'},
                       'Q3': {'answer': 'Phoenix', 'previous': array(['Q2'], dtype=object), 'query': 'What city is both the largest city and the state capital of Nogales?', 'supports': array([{'contents': array(["Arizona ( (listen); Navajo: Hoozdo Hahoodzo Navajo pronunciation: [xòːztò xɑ̀xòːtsò]; O'odham: Alĭ ṣonak Uto-Aztecan pronunciation: [ˡaɺi ˡʂonak]) is a state in the southwestern region of the United States. It is also part of the Western and the Mountain states. It is the sixth largest and the 14th most populous of the 50 states. Its capital and largest city is Phoenix. Arizona shares the Four Corners region with Utah, Colorado, and New Mexico; its other neighboring states are Nevada and California to the west and the Mexican states of Sonora and Baja California to the south and southwest."],
             dtype=object), 'title': 'Arizona'}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ],
      dtype=object), 'tag': '<A3>', 'template': 'What city is both the largest city and the state capital of <A2>?'},
                       'Q4': {'answer': 'Mario Andretti', 'previous': array(['Q3'], dtype=object), 'query': 'Who won the Indy car race in Phoenix?', 'supports': array([{'contents': array(['After a hiatus of eleven years, the race was revived by the Verizon IndyCar Series in 2016. It was held on Saturday night under the lights. Long considered a popular Indy car track, Phoenix has a rich history of open wheel races, including a spectacular crash involving Johnny Rutherford (1980), and the final career victory for Indy legend Mario Andretti (1993).'],
             dtype=object), 'title': 'Desert Diamond West Valley Phoenix Grand Prix'}                                                                                                                                                                                                                                                                                                                     ],
      dtype=object), 'tag': '<A4>', 'template': 'Who won the Indy car race in <A3>?'}}
                  ]
    }

    def fix_label(golden_label):
        """  过滤掉goldenlabel中出现value为None的key """
        new_dict = {}
        for key, value in golden_label.items():
            if value is None:
                continue
            new_dict[key] = value
        return new_dict

    gold_plan = fix_label(print_info['golden_plan'])
    # print(gold_plan)
    gold_graph = [fix_label(print_info['golden_graph'][0])]
    # print(gold_graph)

    plan_sim_score, plan_structure_score, step_score = pse_score.calculate_pse_score(print_info['predict_plan'], gold_plan,
                                                            print_info['predict_graph'], gold_graph)
    print("plan_scores:", plan_sim_score, "step_score:", step_score, 'plan_structure_score:', plan_structure_score)
