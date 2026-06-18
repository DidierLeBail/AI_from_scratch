from ParallelESN import (
    MixingBlock, ReservoirLayer, ReservoirFirstLayer,
    ParallelESN
)
import torch

def testMixingBlock():
    kwargs = dict(kernelSize = 10)
    block = MixingBlock(**kwargs)

    bs = 2
    inp = torch.rand( (bs, 1, 15), dtype=torch.complex32 )
    out = block(inp)

    print(f"size of input: {inp.size()}")
    print(f"size of output: {out.size()}")
    print()
    print(f"input: {inp}")
    print(f"output: {out}")
    print()

def testReservoirLayer():
    kwargs = dict(n_hid=11, leakyRate=0.3)
    block = ReservoirLayer(**kwargs)

    bs = 2
    inp = (
        torch.rand( (bs, kwargs["n_hid"]), dtype=torch.complex32 ),
        torch.rand( (bs, kwargs["n_hid"]), dtype=torch.complex32 )
    )
    out = block(*inp)

    for el in inp:
        print(f"size of input: {el.size()}")
    print(f"size of output: {out.size()}")
    print()
    print(f"input: {inp}")
    print(f"output: {out}")
    print()

def testReservoirFirstLayer():
    kwargs = dict(n_hid=11, n_in=7, leakyRate=0.3)
    block = ReservoirFirstLayer(**kwargs)

    bs = 2
    inp = (
        torch.rand( (bs, kwargs["n_hid"]), dtype=torch.complex32 ),
        torch.rand( (bs, kwargs["n_in"], 1), dtype=torch.complex32 )
    )
    out = block(*inp)

    for el in inp:
        print(f"size of input: {el.size()}")
    print(f"size of output: {out.size()}")
    print()
    print(f"input: {inp}")
    print(f"output: {out}")
    print()

def testParallelESN():
    kwargs = dict(n_layers=3, n_hid=11, n_in=7, leakyRate=0.3)
    block = ParallelESN(**kwargs)

    bs = 2
    inp = (
        torch.rand( (bs, kwargs["n_hid"]), dtype=torch.complex32 ),
        torch.rand( (bs, kwargs["n_in"], 1), dtype=torch.complex32 )
    )
    out = block(*inp)

    for el in inp:
        print(f"size of input: {el.size()}")
    print(f"size of output: {out.size()}")
    print()
    print(f"input: {inp}")
    print(f"output: {out}")
    print()

if __name__ == "__main__":
    testParallelESN()
