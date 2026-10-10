import ast
import operator


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def calculate(expression):
    """
    Safely evaluate basic arithmetic expressions.
    """

    tree = ast.parse(
        expression,
        mode="eval",
    )

    return _evaluate(tree.body)


def _evaluate(node):

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError(
            "Only numeric constants are allowed."
        )

    if isinstance(node, ast.BinOp):
        operation = OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError(
                "Unsupported operation."
            )

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand)

        if isinstance(node.op, ast.USub):
            return -value

        if isinstance(node.op, ast.UAdd):
            return value

    raise ValueError(
        "Invalid mathematical expression."
    )


if __name__ == "__main__":
    print(
        calculate(
            "(130.5 - 60.9) / 60.9 * 100"
        )
    )