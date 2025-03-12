from simple_rag.models import ModelCreator
import pytest

from simple_rag.models.stub_model import StubModel


@pytest.fixture
def model_creator():
    creator = ModelCreator()
    yield creator


def test_model_creation():
    creator = ModelCreator()


def test_create_model_by_name(model_creator):
    name = "stub_model"

    model = model_creator.build(name=name)
    assert isinstance(model, StubModel)


def test_throws_when_model_not_found(model_creator):
    name = "does_not_exist"

    with pytest.raises(ValueError) as e_info:
        model = model_creator.build(name)
