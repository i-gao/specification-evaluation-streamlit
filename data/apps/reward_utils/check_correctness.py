import json
from test_one_solution import check_correctness
import sys


if __name__ == "__main__":
    js = json.load(open(sys.argv[1]))
    passed, output = check_correctness(
        problem=js["problem"],
        generation=js["yhat"],
        timeout=60,
        debug=False,
    )
    print(">>>>>> UNIT TEST RESULTS <<<<<<")
    print(passed)
    print(">>>>>> UNIT TEST OUTPUT <<<<<<")
    print(output)
