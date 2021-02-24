from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

from typing import Dict, List

'''
Apply the adjustment formula
evidences is the state of the environment
'''
def causal_query(target:str, evidence:Dict, actions:List[str], model:BayesianModel):

    queries = {}
    #Pearl Adjustment formula
    for action in actions:
        z = set(model.get_parents(action))

        inference_engine = VariableElimination(model)
        base_query = inference_engine.query(variables=[target], evidence={**evidence, **{action:'True'}}, show_progress=False)

        if len(z) != 0:
            adjustment_query = inference_engine.query(variables=z, show_progress=False)
            query = base_query.product(adjustment_query, inplace=False)
        else:
            query = base_query

        query.normalize()
        query.marginalize(z)
        queries.update({action : query})

    return queries
