import json
from pathlib import Path


def test_local_arithmetic_examples_and_reasoning_counterexamples():
    rows = json.loads(
        (Path(__file__).parents[1] / "examples/concept-local-drafts.json").read_text()
    )
    cells = {(r["concept_id"], r["ability"]): r for r in rows}
    assert len(cells) == 18
    assert {r["correct_choice_index"] for r in rows} == {0, 1, 2, 3}

    def answer(concept, ability):
        row = cells[(concept, ability)]
        return row["choices"][row["correct_choice_index"]].removeprefix(r"\(").removesuffix(r"\)")

    a, b = [[1, 2], [0, 1]], [[2, 0], [3, 1]]
    assert answer("matrix", "application") == str(sum(a[0][k] * b[k][0] for k in range(2)))
    u, v = (1, 2), (-1, 1)
    result = tuple(2 * x - 3 * y for x, y in zip(u, v, strict=True))
    assert answer("vector", "application") == str(result).replace(" ", "")
    x, y = 3, 2
    assert x + y == 5 and x - y == 1
    assert answer("linear system", "application") == str((x, y)).replace(" ", "")
    coordinates = (3, 1)
    assert (coordinates[0] + coordinates[1], coordinates[0] - coordinates[1]) == (4, 2)
    assert answer("basis", "application") == str(coordinates).replace(" ", "")
    k = int(answer("orthogonality", "application"))
    assert k - 2 == 0
    a, b = [[0, 1], [0, 0]], [[0, 0], [1, 0]]
    ab = [[sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    ba = [[sum(b[i][k] * a[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    assert ab == [[1, 0], [0, 0]] and ba == [[0, 0], [0, 1]] and ab != ba
    assert r"AB=\begin{bmatrix}1&0\\0&0\end{bmatrix}" in answer("matrix", "reasoning")
    assert r"BA=\begin{bmatrix}0&0\\0&1\end{bmatrix}" in answer("matrix", "reasoning")
    assert (2 * 1 + 3 * 0, 2 * 0 + 3 * 1, 2 * 1 + 3 * 1) == (2, 3, 5)
    assert tuple(a + b - c for a, b, c in zip((1, 0), (0, 1), (1, 1), strict=True)) == (0, 0)
