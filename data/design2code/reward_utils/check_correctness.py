import json
import dill
from visual_score import visual_eval_v3_multi
import sys

if __name__ == "__main__":

    js = json.load(open(f"_solution_to_grade_{sys.argv[1]}.json"))
    result = visual_eval_v3_multi(
        **js,
    )
    with open(f"_solution_output_{sys.argv[1]}.pkl", "wb") as f:
        dill.dump(result, f)