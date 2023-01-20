import torch
import os
import pytest
import tempfile
from openmm import XmlSerializer, OpenMMException
from openmmtorch import TorchForce

class ForceModule(torch.nn.Module):
    """A simple module that can be serialized"""
    def forward(self, positions):
        return torch.sum(positions**2)


class ForceModule2(torch.nn.Module):
    """A dummy module distict from ForceModule"""
    def forward(self, positions):
        return torch.sum(positions**3)

with tempfile.NamedTemporaryFile(delete=False) as temp:
    model_filename = temp.name

with tempfile.NamedTemporaryFile(delete=False) as temp:
    serialized_filename = temp.name

def createAndSerialize():
    module = torch.jit.script(ForceModule())
    module.save(model_filename)
    torch_force = TorchForce(model_filename)
    stored = XmlSerializer.serialize(torch_force)
    with open(serialized_filename, 'w') as f:
        f.write(stored)

def readXML():
    with open(serialized_filename, 'r') as f:
        fileContents = f.read()
    return fileContents

def deserialize():
    other_force = XmlSerializer.deserialize(readXML())

def test_serialize():
    createAndSerialize()

def test_deserialize():
    createAndSerialize()
    deserialize()

def test_fails_if_model_changed():
    createAndSerialize()
    module = torch.jit.script(ForceModule2())
    module.save(model_filename)
    with pytest.raises(OpenMMException):
        deserialize()

def test_same_module_serializes_identically():
    createAndSerialize()
    module = torch.jit.script(ForceModule())
    module.save(model_filename)
    deserialize()

@pytest.fixture(autouse=True)
def cleanup():
    os.remove(serialized_filename)
    os.remove(model_filename)
