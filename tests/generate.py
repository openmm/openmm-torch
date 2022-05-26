import torch as pt

class Central(pt.nn.Module):
    def forward(self, pos):
        return pos.pow(2).sum()

class Forces(pt.nn.Module):
    def forward(self, pos):
        return pos.pow(2).sum(), -2 * pos

class Global(pt.nn.Module):
    def forward(self, pos, k):
        return k * pos.pow(2).sum()

class Periodic(pt.nn.Module):
    def forward(self, pos, box):
        box = box.diagonal().unsqueeze(0)
        pos = pos - (pos / box).floor() * box
        return pos.pow(2).sum()

pt.jit.script(Central()).save('central.pt')
pt.jit.script(Forces()).save('forces.pt')
pt.jit.script(Global()).save('global.pt')
pt.jit.script(Periodic()).save('periodic.pt')