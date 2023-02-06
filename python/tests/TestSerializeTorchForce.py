import torch
import shutil
import pytest
from openmm import XmlSerializer, OpenMMException
from openmmtorch import TorchForce

@pytest.fixture
def temporal_path(tmp_path_factory):
    temporal = tmp_path_factory.mktemp("data")
    yield str(temporal)
    shutil.rmtree(str(temporal))

class ForceModule(torch.nn.Module):
    """A simple module that can be serialized"""
    def forward(self, positions):
        return torch.sum(positions**2)


class ForceModule2(torch.nn.Module):
    """A dummy module distict from ForceModule"""
    def forward(self, positions):
        return torch.sum(positions**3)


def createAndSerialize(model_filename, serialized_filename):
    module = torch.jit.script(ForceModule())
    module.save(model_filename)
    torch_force = TorchForce(model_filename)
    stored = XmlSerializer.serialize(torch_force)
    with open(serialized_filename, 'w') as f:
        f.write(stored)

def readXML(filename):
    with open(filename, 'r') as f:
        fileContents = f.read()
    return fileContents

def deserialize(filename):
    other_force = XmlSerializer.deserialize(readXML(filename))

def test_serialize(temporal_path):
    model_filename = temporal_path + "/model.pt"
    serialized_filename = temporal_path+ "/stored.xml"
    createAndSerialize(model_filename, serialized_filename)

def test_deserialize(temporal_path):
    model_filename = temporal_path+ "/model.pt"
    serialized_filename = temporal_path+ "/stored.xml"
    createAndSerialize(model_filename, serialized_filename)
    deserialize(serialized_filename)
