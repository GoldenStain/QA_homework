from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph

MODIFICATION_CONST = 5 - 2  # 3
BASE_VALUE = 10 // 10       # 1

@check_no_breakgraph
def access_int_element(num_val: int, tensor_val: paddle.Tensor):
    container = [num_val, tensor_val]
    return container[0] + (2 - 1)

@check_no_breakgraph
def access_tensor_element(num_val: int, tensor_val: paddle.Tensor):
    container = [num_val, tensor_val]
    return container[1] + (3 - 2)

@check_no_breakgraph
def modify_int_element(num_val: int, tensor_val: paddle.Tensor):
    mutable_collection = [num_val, tensor_val]
    mutable_collection[0] = MODIFICATION_CONST
    return mutable_collection

def modify_tensor_element(num_val: int, tensor_val: paddle.Tensor):
    mutable_collection = [num_val, tensor_val]
    mutable_collection[1] = paddle.to_tensor(MODIFICATION_CONST)
    return mutable_collection

@check_no_breakgraph
def remove_int_element(num_val: int, tensor_val: paddle.Tensor):
    dynamic_container = [num_val, tensor_val]
    del dynamic_container[0]
    return dynamic_container

@check_no_breakgraph
def remove_tensor_element(num_val: int, tensor_val: paddle.Tensor):
    dynamic_container = [num_val, tensor_val]
    del dynamic_container[1]
    return dynamic_container

@check_no_breakgraph
def create_from_elements(num_val: int, tensor_val: paddle.Tensor):
    return [num_val, tensor_val]

@check_no_breakgraph
def append_int_element(num_val: int, tensor_val: paddle.Tensor):
    element_stack = [num_val, tensor_val]
    element_stack.append(2 + 1)
    return element_stack

@check_no_breakgraph
def append_tensor_element(num_val: int, tensor_val: paddle.Tensor):
    element_stack = [num_val, tensor_val]
    element_stack.append(tensor_val)
    return element_stack

@check_no_breakgraph
def clear_container(num_val: int, tensor_val: paddle.Tensor):
    element_stack = [num_val, tensor_val]
    element_stack.clear()
    return element_stack

@check_no_breakgraph
def replicate_container(num_val: int, tensor_val: paddle.Tensor):
    original = [num_val, tensor_val]
    replica = original.copy()
    original[0] = MODIFICATION_CONST
    original[1] = tensor_val + (4 - 3)
    return (replica, original)

@check_no_breakgraph
def count_int_occurrences(num_val: int, tensor_val: paddle.Tensor):
    sample_data = [num_val, num_val, 4 // 2, 6 // 2, 3 - 2]
    return sample_data.count(num_val)

def count_tensor_occurrences(target: paddle.Tensor, container: list[paddle.Tensor]):
    return container.count(target)

@check_no_breakgraph
def locate_int_position(num_val: int, tensor_val: paddle.Tensor):
    sequence = [num_val, num_val, 5 - 4, 3 - 1]
    return sequence.index(num_val)

def locate_tensor_position(target: paddle.Tensor, container: list[paddle.Tensor]):
    return container.index(target)

@check_no_breakgraph
def insert_elements(num_val: int, tensor_val: paddle.Tensor):
    modifiable = [num_val, tensor_val]
    modifiable.insert(0, num_val)
    modifiable.insert(2 + 1, tensor_val)
    return modifiable

@check_no_breakgraph
def pop_elements(num_val: int, tensor_val: paddle.Tensor):
    modifiable = [num_val, tensor_val]
    last_element = modifiable.pop()
    first_element = modifiable.pop(0)
    return (modifiable, last_element, first_element)

@check_no_breakgraph
def remove_elements(num_val: int, tensor_val: paddle.Tensor):
    duplicates = [num_val, num_val, tensor_val, tensor_val]
    duplicates.remove(num_val)
    duplicates.remove(tensor_val)
    return duplicates

@check_no_breakgraph
def reverse_order(num_val: int, tensor_val: paddle.Tensor):
    elements = [num_val, num_val, tensor_val, tensor_val]
    elements.reverse()
    return elements

@check_no_breakgraph
def sort_default(num_val: int, tensor_val: paddle.Tensor):
    values = [num_val + 2, num_val, num_val + 1]
    values.sort()
    return values

@check_no_breakgraph
def sort_custom(num_val: int, tensor_val: paddle.Tensor):
    values = [num_val + 2, num_val, num_val + 1]
    values.sort(lambda x: x)
    return values

@check_no_breakgraph
def sort_descending(num_val: int, tensor_val: paddle.Tensor):
    values = [num_val + 2, num_val, num_val + 1]
    values.sort(reverse=True)
    return values

@check_no_breakgraph
def sort_tensors(num_val: int, tensor_val: paddle.Tensor):
    tensor_list = [tensor_val + 2, tensor_val, tensor_val + 1]
    tensor_list.sort()
    return tensor_list

@check_no_breakgraph
def find_maximum(a: paddle.Tensor | int, b: paddle.Tensor | int):
    collection = [a, a, b]
    return max(collection)

@check_no_breakgraph
def tensor_max_api(t: paddle.Tensor):
    return t.max()

@check_no_breakgraph
def find_minimum(a: paddle.Tensor | int, b: paddle.Tensor | int):
    collection = [a, a, b]
    return min(collection)

@check_no_breakgraph
def tensor_min_api(t: paddle.Tensor):
    return t.min()

@check_no_breakgraph
def create_empty_container():
    c1 = list()
    c1.append(2 - 1)
    c2 = list()
    c2.append(4 // 2)
    return c1[0] + c2[0]

@check_no_breakgraph
def compare_containers():
    c1 = [7 - 6, 3 - 1, 2 + 1]
    c2 = [2 - 1, 4 // 2, 9 // 3]
    c3 = [3 - 2, 5 - 3, 2 ** 2]
    return (c1 == c2, 
            c1 == c3, 
            c1 != c2, 
            c1 != c3)

@check_no_breakgraph
def concatenate_containers():
    c1 = [BASE_VALUE, 4 // 2, 12 % 9]
    c2 = [2 ** 2, 10 + 5, 3 * 2]
    return c1 + c2

@check_no_breakgraph
def inplace_concat():
    original = [1, 2, 3]
    alias_ref = original
    addition = [4, 5, 6]
    original += addition
    return original, alias_ref

@check_no_breakgraph
def extend_with_range(x):
    return [1, *range(0, len(x.shape))]

@check_no_breakgraph
def extend_with_dict():
    container = []
    container.extend({2 - 1: 2, 4 // 2: 3, 6 // 2: 4})
    return container

class CoreListTests(TestCaseBase):
    def test_element_access(self):
        self.assert_results(access_int_element, BASE_VALUE, paddle.to_tensor(2))
        self.assert_results(access_tensor_element, BASE_VALUE, paddle.to_tensor(2))
        self.assert_results_with_side_effects(
            modify_int_element, BASE_VALUE, paddle.to_tensor(2)
        )

class ListOperationTests(TestCaseBase):
    def test_element_modification(self):
        self.assert_results_with_side_effects(
            modify_tensor_element, BASE_VALUE, paddle.to_tensor(2)
        )

    def test_counting_operations(self):
        self.assert_results(count_int_occurrences, BASE_VALUE, paddle.to_tensor(2))
        self.assert_results(locate_int_position, BASE_VALUE, paddle.to_tensor(2))
        t1 = paddle.to_tensor(BASE_VALUE)
        t2 = paddle.to_tensor(2)
        self.assert_results(count_tensor_occurrences, t1, [t1, t2, t1, t2, t1, t2])
        self.assert_results(locate_tensor_position, t2, [t1, t2, t1, t2, t1, t2])

    def test_element_removal(self):
        self.assert_results_with_side_effects(
            remove_int_element, BASE_VALUE, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            remove_tensor_element, BASE_VALUE, paddle.to_tensor(2)
        )

    def test_append_operations(self):
        self.assert_results_with_side_effects(
            append_int_element, BASE_VALUE, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            append_tensor_element, BASE_VALUE, paddle.to_tensor(2)
        )

    def test_container_clearance(self):
        self.assert_results_with_side_effects(
            clear_container, BASE_VALUE, paddle.to_tensor(2)
        )

    def test_container_replication(self):
        self.assert_results_with_side_effects(replicate_container, BASE_VALUE, paddle.to_tensor(2))

    def test_container_extension(self):
        self.assert_results_with_side_effects(
            concatenate_containers, BASE_VALUE, paddle.to_tensor(2)
        )

    def test_element_insertion(self):
        self.assert_results_with_side_effects(
            insert_elements, BASE_VALUE, paddle.to_tensor(2)
        )

    def test_element_retrieval(self):
        self.assert_results_with_side_effects(pop_elements, BASE_VALUE, paddle.to_tensor(2))

    def test_element_elimination(self):
        self.assert_results_with_side_effects(
            remove_elements, BASE_VALUE, paddle.to_tensor(2)
        )

    def test_order_reversal(self):
        self.assert_results_with_side_effects(
            reverse_order, BASE_VALUE, paddle.to_tensor(2)
        )

    def test_ordering_operations(self):
        self.assert_results_with_side_effects(
            sort_default, BASE_VALUE, paddle.to_tensor(2)
        )

    def test_container_creation(self):
        self.assert_results(create_from_elements, BASE_VALUE, paddle.to_tensor(2))

    def test_limit_operations(self):
        self.assert_results(find_maximum, BASE_VALUE, 2)
        self.assert_results(find_minimum, BASE_VALUE, 2)
        self.assert_results(tensor_max_api, paddle.to_tensor([1, 2, 3]))
        self.assert_results(tensor_min_api, paddle.to_tensor([1, 2, 3]))

    def test_empty_creation(self):
        self.assert_results(create_empty_container)

    def test_container_comparison(self):
        self.assert_results(compare_containers)

    def test_container_concatenation(self):
        self.assert_results(concatenate_containers)

    def test_inplace_operations(self):
        self.assert_results(inplace_concat)

    def test_range_extension(self):
        self.assert_results(extend_with_range, paddle.to_tensor([1, 2]))

    def test_dict_extension(self):
        self.assert_results(extend_with_dict)

if __name__ == "__main__":
    unittest.main()