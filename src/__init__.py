import ast
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import List, Union


def ast_replace(node: ast.AST, **changes) -> ast.AST:
    """
    Create a copy of an AST node with some fields replaced.

    Args:
        node (ast.AST): The original AST node.
        **changes: Field names and their new values to replace in the node.

    Returns:
        ast.AST: A new AST node with the specified fields replaced.
    """
    new_node = copy(node)
    for field, value in changes.items():
        setattr(new_node, field, value)
    return new_node


def parse_stmts(s: str) -> List[ast.stmt]:
    """
    Parse a string containing Python statements into a list of AST statement nodes.

    Args:
        s (str): The string containing Python statements.

    Returns:
        List[ast.stmt]: A list of AST statement nodes.
    """
    module_node = ast.parse(s)
    return module_node.body


def parse_expr(s: str) -> ast.expr:
    """
    Parse a string containing a Python expression into an AST expression node.

    Args:
        s (str): The string containing a Python expression.

    Returns:
        ast.expr: An AST expression node.
    """
    module_node = ast.parse(s, mode="eval")
    return module_node.body


class ASTRewrite(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the AST rewrite.

        Returns:
            str: The name of the AST rewrite.
        """
        raise NotImplementedError

    @abstractmethod
    def match(
        self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool
    ) -> bool:
        """
        Check if the given AST node matches the rewrite rule.

        Args:
            node (ast.AST): The AST node to check.
            parent (ast.AST): The parent AST node.
            field (str): The name of the field in the parent node that contains the child node.
            inside_list (bool): Whether the child node is inside a list.

        Returns:
            bool: True if the node matches the rewrite rule, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def rewrite(
        self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool
    ) -> Union[ast.AST, List[ast.AST], None]:
        """
        Rewrite the given AST node according to the rewrite rule.

        Args:
            node (ast.AST): The AST node to rewrite.
            parent (ast.AST): The parent AST node.
            field (str): The name of the field in the parent node that contains the child node.
            inside_list (bool): Whether the child node is inside a list.

        Returns:
            Union[ast.AST, List[ast.AST], None]: The rewritten AST node(s), or None to remove the node.
        """
        raise NotImplementedError


@dataclass
class GeneralRemove(ASTRewrite):
    name: str
    target_type: type[ast.AST]
    inside_list: bool = True
    replace_with: "ast.AST | None" = None

    @override
    def get_name(self) -> str:
        return self.name

    @override
    def match(
        self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool
    ) -> bool:
        return isinstance(node, self.target_type) and (
            not self.inside_list or inside_list
        )

    @override
    def rewrite(
        self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool
    ) -> None:
        return deepcopy


def expr_to_zeros(expr: ast.expr) -> ast.expr:
    """
    Convert an AST expression node to an AST node representing zeros of the same shape.

    Args:
        expr (ast.expr): The original AST expression node.

    Returns:
        ast.expr: An AST expression node representing zeros of the same shape.
    """
    zeros_call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="torch", ctx=ast.Load()),
            attr="zeros",
            ctx=ast.Load(),
        ),
        args=[expr],
        keywords=[],
    )
    return zeros_call


class CallFwdArgs1(ASTRewrite):
    @override
    def get_name(self) -> str:
        return "CallFwdArgs1"

    @override
    def match(
        self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool
    ) -> bool:
        return isinstance(node, ast.Call) and len(node.args) >= 1

    @override
    def rewrite(
        self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool
    ) -> ast.AST:
        first_arg = node.args[0]
        return deepcopy(first_arg)


class AttachFullFuncArgs(ASTRewrite):
    @override
    def get_name(self) -> str:
        return "AttachFullFuncArgs"

    @override
    def match(
        self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool
    ) -> bool:
        return isinstance(node, ast.Call)

    @override
    def rewrite(
        self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool
    ) -> ast.AST:
        args_list = ast.List(elts=node.args, ctx=ast.Load())
        new_call = ast.Call(
            func=ast.Attribute(
                value=deepcopy(node.func),
                attr="with_full_args",
                ctx=ast.Load(),
            ),
            args=[args_list],
            keywords=deepcopy(node.keywords),
        )
        return new_call


@dataclass
class IntConstApply(ASTRewrite):
    name: str
    matcher: str
    apply: int

    @override
    def get_name(self) -> str:
        return self.name

    @override
    def match(
        self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool
    ) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == self.target_func
            and len(node.args) >= 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, int)
        )

    @override
    def rewrite(
        self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool
    ) -> ast.AST:
        original_value = node.args[0].value
        new_value = original_value + self.int_const
        new_arg = ast.Constant(value=new_value)
        new_args = [new_arg] + deepcopy(node.args[1:])
        new_call = ast.Call(
            func=deepcopy(node.func),
            args=new_args,
            keywords=deepcopy(node.keywords),
        )
        return new_call


@dataclass
class BinOpFwdArgs(ASTRewrite):
    forward: Literal["left", "right"] = "left"

    @override
    def get_name(self) -> str:
        return f"binop-fwd-args-{self.forward}"

    @override
    def match(
        self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool
    ) -> bool:
        return isinstance(node, ast.BinOp)

    @override
    def rewrite(
        self, node: ast.AST, parent: ast.AST, field: str, inside_list: bool
    ) -> ast.AST:
        if self.forward == "left":
            return deepcopy(node.left)
        else:
            return deepcopy(node.right)


def _as_expr_placeholder(temp: ast.AST) -> "str | None":
    if isinstance(temp, ast.Name):
        return temp.value.id
    else:
        return None


def _as_stmt_placeholder(temp: ast.AST) -> "str | None":
    if isinstance(temp, ast.Expr) and isinstance(temp.value, ast.Name):
        return temp.value.id
    else:
        return None


def _ast_match(temp: ast.AST, node: ast.expr, placeholders: set[str]):
    _as_expr_placeholder(temp)
