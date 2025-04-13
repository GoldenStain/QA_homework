import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit.sot.psdb import check_no_breakgraph

DEFAULT_KEY = 1
ALTERNATE_KEY = 2
NEW_KEY = 3
MISSING_KEY = 4
DEFAULT_VALUE = 2
TENSOR_VALUE = paddle.to_tensor(2)

@check_no_breakgraph
def create_basic_dictionary(key: int, value_tensor: paddle.Tensor):
    data_dict = {key: value_tensor}
    return data_dict[key] + 1

@check_no_breakgraph
def create_predefined_dictionary(key: int, value_tensor: paddle.Tensor):
    preset_dict = {DEFAULT_KEY: value_tensor, ALTERNATE_KEY: value_tensor + 1}
    return preset_dict[key] + 1

@check_no_breakgraph
def retrieve_dictionary_items(key: int, value_tensor: paddle.Tensor):
    sample_dict = {DEFAULT_KEY: key, ALTERNATE_KEY: value_tensor + 1}
    return (sample_dict.get(DEFAULT_KEY), sample_dict.get(ALTERNATE_KEY))

@check_no_breakgraph
def handle_missing_keys(key: int, value_tensor: paddle.Tensor):
    sample_dict = {DEFAULT_KEY: key, ALTERNATE_KEY: value_tensor + 1}
    return (sample_dict.get(MISSING_KEY, DEFAULT_VALUE), sample_dict.get(MISSING_KEY + 1, value_tensor))

@check_no_breakgraph
def modify_integer_value(key: int, value_tensor: paddle.Tensor):
    mutable_dict = {DEFAULT_KEY: key, ALTERNATE_KEY: value_tensor + 1}
    mutable_dict[DEFAULT_KEY] = key * 2
    return mutable_dict[DEFAULT_KEY]

@check_no_breakgraph
def update_tensor_value(key: int, value_tensor: paddle.Tensor):
    mutable_dict = {DEFAULT_KEY: key, ALTERNATE_KEY: value_tensor + 1}
    mutable_dict[ALTERNATE_KEY] = value_tensor
    return mutable_dict[DEFAULT_KEY]

@check_no_breakgraph
def basic_dictionary_update(key: int, value_tensor: paddle.Tensor):
    base_dict = {DEFAULT_KEY: key, ALTERNATE_KEY: value_tensor + 1}
    update_data = {DEFAULT_KEY: key * 2, ALTERNATE_KEY: value_tensor, NEW_KEY: value_tensor + 2}
    base_dict.update(update_data)
    return base_dict

@check_no_breakgraph
def chained_dictionary_update(key: int, value_tensor: paddle.Tensor):
    base_dict = {DEFAULT_KEY: key, ALTERNATE_KEY: value_tensor + 1}
    update_data = {
        DEFAULT_KEY: key * 2,
        ALTERNATE_KEY: value_tensor,
        NEW_KEY: base_dict[ALTERNATE_KEY] + 2
    }
    base_dict.update(update_data)
    return base_dict

@check_no_breakgraph
def remove_integer_entry(key: int, value_tensor: paddle.Tensor):
    mutable_dict = {DEFAULT_KEY: key, ALTERNATE_KEY: value_tensor + 1}
    del mutable_dict[DEFAULT_KEY]
    return mutable_dict

@check_no_breakgraph
def remove_tensor_entry(key: int, value_tensor: paddle.Tensor):
    mutable_dict = {DEFAULT_KEY: key, ALTERNATE_KEY: value_tensor + 1}
    del mutable_dict[ALTERNATE_KEY]
    return mutable_dict

@check_no_breakgraph
def clear_dictionary_contents(key: int, value_tensor: paddle.Tensor):
    mutable_dict = {DEFAULT_KEY: key, ALTERNATE_KEY: value_tensor + 1}
    mutable_dict.clear()
    return mutable_dict

@check_no_breakgraph
def clone_dictionary(key: int, value_tensor: paddle.Tensor):
    original = {DEFAULT_KEY: key, ALTERNATE_KEY: value_tensor + 1}
    replica = original.copy()
    original[DEFAULT_KEY] = DEFAULT_VALUE
    return replica

@check_no_breakgraph
def set_default_values(key: int, value_tensor: paddle.Tensor):
    base_dict = {DEFAULT_KEY: key, ALTERNATE_KEY: value_tensor + 1}
    missing_entry = base_dict.setdefault(MISSING_KEY)
    existing_entry = base_dict.setdefault(DEFAULT_KEY, DEFAULT_VALUE)
    new_entry = base_dict.setdefault(NEW_KEY, DEFAULT_VALUE + 2)
    return (base_dict, missing_entry, existing_entry, new_entry)

@check_no_breakgraph
def remove_entries_with_pop(key: int, value_tensor: paddle.Tensor):
    mutable_dict = {
        DEFAULT_KEY: key,
        ALTERNATE_KEY: value_tensor + 1,
        NEW_KEY: value_tensor
    }
    removed_value = mutable_dict.pop(DEFAULT_KEY)
    existing_removed = mutable_dict.pop(ALTERNATE_KEY, NEW_KEY)
    missing_removed = mutable_dict.pop(MISSING_KEY, NEW_KEY)
    tensor_removed = mutable_dict.pop(MISSING_KEY + 1, value_tensor)
    return (mutable_dict, removed_value, existing_removed, missing_removed, tensor_removed)

@check_no_breakgraph
def remove_last_entry():
    mutable_dict = {DEFAULT_KEY: 1, ALTERNATE_KEY: 2, NEW_KEY: 3}
    last_item = mutable_dict.popitem()
    return (mutable_dict, last_item)

@check_no_breakgraph
def create_from_existing_dict():
    source = {DEFAULT_KEY: DEFAULT_VALUE, ALTERNATE_KEY: DEFAULT_VALUE * 2}
    new_dict = dict(source)
    return new_dict

@check_no_breakgraph
def create_from_nested_sequences():
    pairs = [[DEFAULT_KEY, DEFAULT_VALUE], [ALTERNATE_KEY, DEFAULT_VALUE * 2]]
    new_dict = dict(pairs)
    return new_dict

@check_no_breakgraph
def create_from_tuple_pairs():
    pairs = ((DEFAULT_KEY, DEFAULT_VALUE), (ALTERNATE_KEY, DEFAULT_VALUE * 2))
    new_dict = dict(pairs)
    return new_dict

@check_no_breakgraph
def create_with_comprehension():
    source = {DEFAULT_KEY: DEFAULT_VALUE, ALTERNATE_KEY: DEFAULT_VALUE * 2}
    transformed = {k: v + 1 for k, v in source.items()}
    return transformed

@check_no_breakgraph
def initialize_empty_dictionaries():
    first_dict = dict()  # noqa: C408
    first_dict.update({DEFAULT_KEY: DEFAULT_VALUE})
    second_dict = dict()  # noqa: C408
    second_dict.update({ALTERNATE_KEY: DEFAULT_VALUE * 2})
    return first_dict[DEFAULT_KEY] + second_dict[ALTERNATE_KEY]

@check_no_breakgraph
def create_with_fromkeys(keys):
    return dict.fromkeys(keys)

@check_no_breakgraph
def create_with_default_fromkeys(keys, default_value):
    return dict.fromkeys(keys, default_value)

@check_no_breakgraph
def initialize_with_keywords():
    config = dict(x=DEFAULT_VALUE, y=DEFAULT_VALUE * 2)  # noqa: C408
    return config["x"] + config["y"]

class DictionaryConstructionTests(TestCaseBase):
    def test_basic_creation(self):
        self.assert_results(create_basic_dictionary, DEFAULT_KEY, TENSOR_VALUE)
    
    def test_predefined_creation(self):
        self.assert_results(create_predefined_dictionary, DEFAULT_KEY, TENSOR_VALUE)

class DictionaryOperationTests(TestCaseBase):
    def test_item_retrieval(self):
        self.assert_results(retrieve_dictionary_items, DEFAULT_KEY, TENSOR_VALUE)
        self.assert_results(handle_missing_keys, DEFAULT_KEY, TENSOR_VALUE)

    def test_value_modification(self):
        self.assert_results_with_side_effects(
            modify_integer_value, DEFAULT_KEY, TENSOR_VALUE
        )
        self.assert_results_with_side_effects(
            update_tensor_value, DEFAULT_KEY, TENSOR_VALUE
        )

    def test_dictionary_cloning(self):
        self.assert_results_with_side_effects(clone_dictionary, DEFAULT_KEY, TENSOR_VALUE)

    def test_content_updates(self):
        self.assert_results_with_side_effects(
            basic_dictionary_update, DEFAULT_KEY, TENSOR_VALUE
        )
        self.assert_results_with_side_effects(
            chained_dictionary_update, DEFAULT_KEY, TENSOR_VALUE
        )

    def test_default_value_handling(self):
        self.assert_results_with_side_effects(set_default_values, DEFAULT_KEY, TENSOR_VALUE)

    def test_entry_removal(self):
        self.assert_results_with_side_effects(
            remove_integer_entry, DEFAULT_KEY, TENSOR_VALUE
        )
        self.assert_results_with_side_effects(
            remove_tensor_entry, DEFAULT_KEY, TENSOR_VALUE
        )
        self.assert_results_with_side_effects(
            clear_dictionary_contents, DEFAULT_KEY, TENSOR_VALUE
        )

    def test_pop_operations(self):
        self.assert_results_with_side_effects(remove_entries_with_pop, DEFAULT_KEY, TENSOR_VALUE)
        self.assert_results_with_side_effects(remove_last_entry)

    def test_initialization_methods(self):
        self.assert_results(create_from_existing_dict)
        self.assert_results(create_from_nested_sequences)
        self.assert_results(create_from_tuple_pairs)
        self.assert_results(create_with_comprehension)
        self.assert_results(initialize_empty_dictionaries)
        self.assert_results(create_with_fromkeys, (DEFAULT_KEY, ALTERNATE_KEY, NEW_KEY, MISSING_KEY))
        self.assert_results(create_with_fromkeys, [DEFAULT_KEY, ALTERNATE_KEY, NEW_KEY, MISSING_KEY])
        self.assert_results(create_with_default_fromkeys, (DEFAULT_KEY, ALTERNATE_KEY, NEW_KEY, MISSING_KEY), DEFAULT_VALUE)
        self.assert_results(create_with_default_fromkeys, [DEFAULT_KEY, ALTERNATE_KEY, NEW_KEY, MISSING_KEY], DEFAULT_VALUE)
        self.assert_results(initialize_with_keywords)

if __name__ == "__main__":
    unittest.main()
