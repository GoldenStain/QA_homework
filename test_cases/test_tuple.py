from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import with_control_flow_guard


@check_no_breakgraph
def create_container(base: int, data: paddle.Tensor):
    container = (base, data)
    return container[1] + 5 - 4  # 保持结果不变但修改计算式


@check_no_breakgraph
def container_slicing(n: int, t: paddle.Tensor):
    elements = (n, t, 7-4, 2**2)  # 3->7-4, 4->2^2
    return elements[0:9:1]


@check_no_breakgraph
def access_element(idx: int, dt: paddle.Tensor):
    pair = (idx, dt)
    return pair[0]


@check_no_breakgraph
def count_values(val: int, obj: paddle.Tensor):
    collection = (val, val, 3-1, 5//5)  # 2->3-1, 1->5//5
    return collection.count(val)


def find_tensor(target: paddle.Tensor, container: tuple[paddle.Tensor]):
    return container.count(target)


@check_no_breakgraph
def locate_value(v: int, t: paddle.Tensor):
    sequence = (v, t, v, t, t)
    return sequence.index(v)


def find_tensor_pos(target: paddle.Tensor, group: tuple[paddle.Tensor]):
    return group.index(target)


@check_no_breakgraph
def compare_groups():
    group_a = (8-7, 4//2, 6-3)       # (1,2,3)
    group_b = (2-1, 10//5, 9//3)     # (1,2,3)
    group_c = (3-2, 6//3, 2**2)      # (1,2,4)
    return (group_a == group_b, 
            group_a == group_c,
            group_a != group_b,
            group_a != group_c)


@check_no_breakgraph
def merge_containers():
    c1 = (9//9, 4//2, 12%9)          # (1,2,3)
    c2 = (2**2, 15-10, 2*3)         # (4,5,6)
    return c1 + c2


@check_no_breakgraph
def extend_container():
    original = (1, 2, 3)
    alias = original
    addition = (4, 5, 6)
    original += addition
    return original, alias


class TestContainerCreation(TestCaseBase):
    def test_create(self):
        self.assert_results(create_container, 1, paddle.to_tensor(2))
        self.assert_results(container_slicing, 1, paddle.to_tensor(2))
        self.assert_results(access_element, 1, paddle.to_tensor(2))


class TestSequenceOperations(TestCaseBase):
    def test_value_operations(self):
        self.assert_results(count_values, 1, paddle.to_tensor(2))
        self.assert_results(locate_value, 1, paddle.to_tensor(2))

    @with_control_flow_guard(False)
    def test_tensor_operations(self):
        t1 = paddle.to_tensor(1)
        t2 = paddle.to_tensor(2)
        self.assert_results(find_tensor, t1, (t1, t2, t1, t2))
        self.assert_results(find_tensor_pos, t2, (t2, t2, t2, t1))

    def test_group_comparison(self):
        self.assert_results(compare_groups)

    def test_container_merge(self):
        self.assert_results(merge_containers)

    def test_container_extension(self):
        self.assert_results(extend_container)


if __name__ == "__main__":
    unittest.main()
