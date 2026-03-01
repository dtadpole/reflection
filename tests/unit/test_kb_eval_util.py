"""Tests for kbEval core evaluation utilities (CPU only, no GPU required)."""

import pytest

from services.kb_eval.baseline.util import (
    _get_call_name,
    _is_triton_jit,
    load_custom_model,
    load_model_and_inputs,
    resolve_triton_code,
)


class TestLoadModelAndInputs:
    def test_valid_reference(self):
        code = """
class Model:
    def __init__(self):
        pass
    def forward(self, x):
        return x

def get_inputs():
    return [[1, 2, 3]]

def get_init_inputs():
    return []
"""
        context: dict = {"__builtins__": __builtins__}
        model_cls, get_inputs, get_init_inputs = load_model_and_inputs(code, context)
        assert model_cls is not None
        assert callable(get_inputs)
        assert callable(get_init_inputs)

    def test_missing_model(self):
        code = """
def get_inputs():
    return []
def get_init_inputs():
    return []
"""
        context: dict = {"__builtins__": __builtins__}
        with pytest.raises(ValueError, match="Model"):
            load_model_and_inputs(code, context)

    def test_missing_get_inputs(self):
        code = """
class Model:
    pass
def get_init_inputs():
    return []
"""
        context: dict = {"__builtins__": __builtins__}
        with pytest.raises(ValueError, match="get_inputs"):
            load_model_and_inputs(code, context)

    def test_missing_get_init_inputs(self):
        code = """
class Model:
    pass
def get_inputs():
    return []
"""
        context: dict = {"__builtins__": __builtins__}
        with pytest.raises(ValueError, match="get_init_inputs"):
            load_model_and_inputs(code, context)


class TestLoadCustomModel:
    def test_valid_generated(self):
        ref_code = """
class Model:
    pass
def get_inputs():
    return []
def get_init_inputs():
    return []
"""
        gen_code = """
class ModelNew:
    pass
"""
        context: dict = {"__builtins__": __builtins__}
        load_model_and_inputs(ref_code, context)
        new_cls = load_custom_model(gen_code, context)
        assert new_cls.__name__ == "ModelNew"

    def test_missing_model_new(self):
        gen_code = """
class SomeOtherModel:
    pass
"""
        context: dict = {"__builtins__": __builtins__, "Model": type("Model", (), {})}
        with pytest.raises(ValueError, match="ModelNew"):
            load_custom_model(gen_code, context)

    def test_clobbers_original_model(self):
        original = type("Model", (), {})
        gen_code = """
class Model:
    pass
class ModelNew:
    pass
"""
        context: dict = {"__builtins__": __builtins__, "Model": original}
        with pytest.raises(ValueError, match="must not modify"):
            load_custom_model(gen_code, context)


class TestResolveTritonCode:
    def test_valid_triton_code(self):
        code = """
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, y_ptr, n: tl.constexpr):
    pid = tl.program_id(0)

class ModelNew:
    def __init__(self):
        pass
    def forward(self, x):
        my_kernel(x, x, 128)
        return x
"""
        result = resolve_triton_code(code)
        assert "my_kernel" in result["jit_functions"]
        assert result["model_new_class"] == "ModelNew"
        assert "my_kernel" in result["forward_calls"]

    def test_no_jit_function(self):
        code = """
class ModelNew:
    def forward(self, x):
        return x
"""
        with pytest.raises(ValueError, match="No @triton.jit"):
            resolve_triton_code(code)

    def test_no_model_new(self):
        code = """
import triton

@triton.jit
def my_kernel():
    pass
"""
        with pytest.raises(ValueError, match="No 'ModelNew'"):
            resolve_triton_code(code)

    def test_forward_doesnt_call_jit(self):
        code = """
import triton

@triton.jit
def my_kernel():
    pass

class ModelNew:
    def forward(self, x):
        return x
"""
        with pytest.raises(ValueError, match="does not call any jit"):
            resolve_triton_code(code)

    def test_syntax_error(self):
        code = "def incomplete("
        with pytest.raises(ValueError, match="Syntax error"):
            resolve_triton_code(code)

    def test_multiple_jit_functions(self):
        code = """
import triton

@triton.jit
def kernel_a():
    pass

@triton.jit
def kernel_b():
    pass

class ModelNew:
    def forward(self, x):
        kernel_a()
        kernel_b()
        return x
"""
        result = resolve_triton_code(code)
        assert len(result["jit_functions"]) == 2
        assert "kernel_a" in result["jit_functions"]
        assert "kernel_b" in result["jit_functions"]


class TestASTHelpers:
    def test_is_triton_jit_attribute(self):
        import ast

        node = ast.parse("@triton.jit\ndef f(): pass").body[0]
        assert _is_triton_jit(node.decorator_list[0]) is True

    def test_is_not_triton_jit(self):
        import ast

        node = ast.parse("@other.jit\ndef f(): pass").body[0]
        assert _is_triton_jit(node.decorator_list[0]) is False

    def test_get_call_name_simple(self):
        import ast

        node = ast.parse("foo()").body[0].value
        assert _get_call_name(node) == "foo"

    def test_get_call_name_attribute(self):
        import ast

        node = ast.parse("self.foo()").body[0].value
        assert _get_call_name(node) == "foo"

    def test_get_call_name_subscript(self):
        """Triton kernel launch pattern: kernel[grid](...)."""
        import ast

        node = ast.parse("relu_kernel[grid](x, y, n)").body[0].value
        assert _get_call_name(node) == "relu_kernel"
