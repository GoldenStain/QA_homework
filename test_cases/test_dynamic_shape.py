from __future__ import annotations

import math
import unittest
from test_case_base import TestCaseBase, test_instruction_translator_cache_context
import paddle
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import allow_dynamic_shape_guard

DYNAMIC_DIM_START = 3
DYNAMIC_DIM_END = 7
IMAGE_SIZE = 224
BASE_DIMENSION = 4
TENSOR_RANK = 3

def process_dynamic_dimension(input_tensor):
    dimension_size = input_tensor.shape[0]
    return input_tensor + dimension_size

def reshape_with_dynamic_dim(input_tensor, dimension_size):
    reshaped = paddle.reshape(input_tensor, [dimension_size, -1])
    return (reshaped + dimension_size) * 2 - 1, (-dimension_size + 1) * 2 - 1, type(dimension_size) is int

def apply_dimension_constraint(input_tensor, dimension_size):
    return (input_tensor + dimension_size) * 2

def process_dynamic_index(input_tensor, index_dict):
    return input_tensor + index_dict[1]

def conditional_dimension_processing(input_tensor, dimension_size):
    if dimension_size < 4:
        return 1
    reshaped = paddle.reshape(input_tensor, [dimension_size, -1])
    return (reshaped + dimension_size) * 2 - 1, (-dimension_size + 1) * 2 - 1

def access_inner_dimension(input_tensor):
    adjusted_tensor = input_tensor + 1
    return adjusted_tensor.shape[0]

def reshape_with_dynamic_list(input_tensor, target_shape):
    return input_tensor.reshape(target_shape)

def process_dimension_calculation(input_value):
    adjusted_value = input_value * 0.5
    trigonometric_result = math.sin(adjusted_value)
    return trigonometric_result

class CustomStridedConv(paddle.nn.Conv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @paddle.jit.to_static(full_graph=False)
    def forward(self, x):
        modified_stride = [self._stride[0] + 1, self._stride[1]]
        return paddle.nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            modified_stride,
            self._padding,
            self._dilation,
            self._groups,
            self._data_format,
        )

def dynamic_pooling(input_tensor, kernel_size):
    return paddle.nn.functional.max_pool2d(input_tensor, kernel_size=kernel_size)

class DynamicShapeCacheTests(TestCaseBase):
    def test_integer_dimension_cache_hit_scenario_1(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            base_tensor = paddle.randn([BASE_DIMENSION, 5, 6])
            self.assert_results(reshape_with_dynamic_dim, base_tensor, 2)
            self.assertEqual(ctx.translate_count, 1)
            for dim in range(DYNAMIC_DIM_START, DYNAMIC_DIM_END):
                self.assert_results(reshape_with_dynamic_dim, base_tensor, dim)
                self.assertEqual(ctx.translate_count, 2)

    def test_dynamic_index_cache_behavior(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            base_tensor = paddle.randn([BASE_DIMENSION, 5, 6])
            self.assert_results(process_dynamic_index, base_tensor, {1: 2})
            self.assertEqual(ctx.translate_count, 1)
            for idx in range(DYNAMIC_DIM_START, DYNAMIC_DIM_END):
                self.assert_results(process_dynamic_index, base_tensor, {1: idx})
                self.assertEqual(ctx.translate_count, 2)

    def test_conditional_dimension_cache_behavior(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            for dim in range(0, 6):
                self.assert_results(conditional_dimension_processing, paddle.randn([4, 5, 6]), dim)
                self.assertEqual(ctx.translate_count, dim + 1)

    def test_dimension_access_cache_behavior(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            self.assert_results(process_dynamic_dimension, paddle.randn([2, 4, 5]))
            self.assertEqual(ctx.translate_count, 1)
            for dim in range(DYNAMIC_DIM_START, DYNAMIC_DIM_END):
                self.assert_results(process_dynamic_dimension, paddle.randn([dim, 4, 5]))
                self.assertEqual(ctx.translate_count, 2)

    def test_inner_dimension_access_cache(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            self.assert_results(access_inner_dimension, paddle.randn([2, 4, 5]))
            self.assertEqual(ctx.translate_count, 1)
            for dim in range(DYNAMIC_DIM_START, DYNAMIC_DIM_END):
                self.assert_results(access_inner_dimension, paddle.randn([dim, 4, 5]))
                self.assertEqual(ctx.translate_count, 2)

    def test_dimension_conversion_operations(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            bool_converter = check_no_breakgraph(lambda n: bool(n))
            int_converter = check_no_breakgraph(lambda n: int(n))
            float_converter = check_no_breakgraph(lambda n: float(n))
            for converter in [bool_converter, int_converter, float_converter]:
                self.assert_results(converter, 1)
                self.assert_results(converter, 2)

    def test_list_shape_processing_cache(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            self.assert_results(reshape_with_dynamic_list, paddle.randn([2, 2, 5]), [4, 5])
            self.assertEqual(ctx.translate_count, 1)
            for dim in range(DYNAMIC_DIM_START, DYNAMIC_DIM_END):
                self.assert_results(reshape_with_dynamic_list, paddle.randn([dim, 2, 5]), [dim * 2, 5])
                self.assertEqual(ctx.translate_count, 2)

    def test_custom_convolution_stride_fallback(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            for stride_value in range(1, 5):
                conv_layer = CustomStridedConv(3, 3, 3, stride=stride_value)
                conv_layer(paddle.randn([1, 3, IMAGE_SIZE, IMAGE_SIZE]))
                self.assertEqual(ctx.translate_count, stride_value)

    def test_pooling_kernel_fallback(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            for kernel in range(1, 5):
                input_tensor = paddle.randn([1, 3, IMAGE_SIZE, IMAGE_SIZE])
                self.assert_results(dynamic_pooling, input_tensor, kernel)
                self.assertEqual(ctx.translate_count, kernel)

    def test_dynamic_padding_fallback(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            padding_func = check_no_breakgraph(
                lambda x, pad: paddle.nn.functional.pad(x, [0, pad, 0, 0])
            )
            for padding in range(1, 5):
                self.assert_results(padding_func, paddle.randn([1, 3, IMAGE_SIZE, IMAGE_SIZE]), padding)
                self.assertEqual(ctx.translate_count, padding)

    def test_dimension_calculation_consistency(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            for value in range(1, 6):
                self.assert_results(process_dimension_calculation, value)

    def test_mixed_dimension_scenarios(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            base_tensor = paddle.randn([BASE_DIMENSION, 5, 6])
            self.assert_results(reshape_with_dynamic_dim, base_tensor, 1)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(reshape_with_dynamic_dim, base_tensor, 0)
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(reshape_with_dynamic_dim, base_tensor, 2)
            self.assertEqual(ctx.translate_count, 3)
            for dim in range(DYNAMIC_DIM_START, 6):
                self.assert_results(reshape_with_dynamic_dim, base_tensor, dim)
                self.assertEqual(ctx.translate_count, 4)

    def test_sequential_dimension_processing(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            base_tensor = paddle.randn([BASE_DIMENSION, 5, 6])
            self.assert_results(reshape_with_dynamic_dim, base_tensor, 2)
            self.assertEqual(ctx.translate_count, 1)
            for dim in range(DYNAMIC_DIM_START, 6):
                self.assert_results(reshape_with_dynamic_dim, base_tensor, dim)
                self.assertEqual(ctx.translate_count, 2)
            self.assert_results(reshape_with_dynamic_dim, base_tensor, 0)
            self.assertEqual(ctx.translate_count, 3)
            self.assert_results(reshape_with_dynamic_dim, base_tensor, 1)
            self.assertEqual(ctx.translate_count, 4)

    def test_constrained_dimension_processing(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            self.assert_results(apply_dimension_constraint, paddle.randn([4, 5, 6]), 2)
            self.assertEqual(ctx.translate_count, 1)
            for dim in range(DYNAMIC_DIM_START, DYNAMIC_DIM_END):
                self.assert_results(
                    apply_dimension_constraint,
                    paddle.randn([4 + dim, 5, 6]),
                    dim,
                )
                self.assertEqual(ctx.translate_count, 2)

@check_no_breakgraph
def execute_dimension_operations(input_tensor):
    base_dimension = input_tensor.shape[0]
    adjusted_dimension = base_dimension + 1
    calculated_dimension = 1 + base_dimension
    dimension_sum = adjusted_dimension + calculated_dimension
    dimension_product = adjusted_dimension * calculated_dimension
    dimension_difference = adjusted_dimension - calculated_dimension
    dimension_division = adjusted_dimension / calculated_dimension
    floor_division = adjusted_dimension // calculated_dimension
    remainder = adjusted_dimension % calculated_dimension
    power_result = adjusted_dimension ** calculated_dimension
    bitwise_and = adjusted_dimension & calculated_dimension
    bitwise_or = adjusted_dimension | calculated_dimension
    bitwise_xor = adjusted_dimension ^ calculated_dimension
    left_shift = adjusted_dimension << calculated_dimension
    right_shift = adjusted_dimension >> calculated_dimension
    equality_check = adjusted_dimension == calculated_dimension
    inequality_check = adjusted_dimension != calculated_dimension
    less_than = adjusted_dimension < calculated_dimension
    less_equal = adjusted_dimension <= calculated_dimension
    greater_than = adjusted_dimension > calculated_dimension
    greater_equal = adjusted_dimension >= calculated_dimension
    boolean_cast = bool(adjusted_dimension)
    logical_not = not calculated_dimension
    return (
        base_dimension,
        adjusted_dimension,
        calculated_dimension,
        dimension_sum,
        dimension_product,
        dimension_difference,
        dimension_division,
        floor_division,
        remainder,
        power_result,
        bitwise_and,
        bitwise_or,
        bitwise_xor,
        left_shift,
        right_shift,
        equality_check,
        inequality_check,
        less_than,
        less_equal,
        greater_than,
        greater_equal,
        boolean_cast,
        logical_not,
    )

@check_no_breakgraph
def execute_inplace_operations(input_tensor):
    base_dimension = input_tensor.shape[0]
    adjusted_dimension = base_dimension + 1
    sum_result = product_result = subtract_result = divide_result = floor_div_result = mod_result = power_result = bit_and_result = bit_or_result = xor_result = left_shift_result = right_shift_result = base_dimension
    
    sum_result += adjusted_dimension
    product_result *= adjusted_dimension
    subtract_result -= adjusted_dimension
    # TODO: Re-enable after resolving numerical precision differences
    # divide_result /= adjusted_dimension
    floor_div_result //= adjusted_dimension
    mod_result %= adjusted_dimension
    power_result **= adjusted_dimension
    bit_and_result &= adjusted_dimension
    bit_or_result |= adjusted_dimension
    xor_result ^= adjusted_dimension
    left_shift_result <<= adjusted_dimension
    right_shift_result >>= adjusted_dimension
    
    return (
        base_dimension,
        adjusted_dimension,
        sum_result,
        product_result,
        subtract_result,
        floor_div_result,
        mod_result,
        power_result,
        bit_and_result,
        bit_or_result,
        xor_result,
        left_shift_result,
        right_shift_result,
    )

class DimensionOperationTests(TestCaseBase):
    def test_complex_dimension_operations(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            self.assert_results(execute_dimension_operations, paddle.randn([BASE_DIMENSION, 5, 6]))
            self.assertEqual(ctx.translate_count, 1)
            for dim in range(5, 9):
                self.assert_results(execute_dimension_operations, paddle.randn([dim, 5, 6]))
                self.assertEqual(ctx.translate_count, 2)

    def test_inplace_dimension_operations(self):
        with allow_dynamic_shape_guard(True), test_instruction_translator_cache_context() as ctx:
            self.assert_results(execute_inplace_operations, paddle.randn([BASE_DIMENSION, 5, 6]))
            self.assertEqual(ctx.translate_count, 1)
            for dim in range(5, 9):
                self.assert_results(execute_inplace_operations, paddle.randn([dim, 5, 6]))
                self.assertEqual(ctx.translate_count, 2)

if __name__ == '__main__':
    unittest.main()
