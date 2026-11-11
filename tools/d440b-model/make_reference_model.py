#!/usr/bin/env python3
"""Write the d440b reference trajectory model and its manifest (nextgsim#463).

This script plays the OAM *training* role of TS 38.300 §16.20.2 for the
d440b-model-verify demo: it produces the model artefact the gNB verifies before
it activates it. It is a file drop standing in for OAM, not a TS 28.105 MnS
producer, and **no training happens**. The "model" is a fixed ONNX graph, a
constant-velocity step:

    next = 2 * positions[:, -1, :] - positions[:, -2, :]

i.e. the position one sample step after the last one. Its accuracy is not the
point and is not claimed; what the demo proves is that a *specific* artefact
was checked, activated and served live inferences.

Standard library only (no `onnx` package): the ONNX ModelProto is written with
a minimal protobuf encoder, so the bytes are a pure function of this file and
`--check` can prove the committed artefacts are exactly what it writes.

    tools/d440b-model/make_reference_model.py           # (re)write the artefacts
    tools/d440b-model/make_reference_model.py --check   # fail if they differ

Golden vectors use small integers, so the expected outputs are exact in f32 on
any ONNX Runtime; the tolerance only absorbs a runtime that reorders the sum.
They are still asserted only on the ONNX Runtime version CI pins.
"""

import hashlib
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "nwdaf-trajectory-cv"
MODEL_VERSION = "1.0.0"
OPSET = 13
IR_VERSION = 7
MODEL_FILE = MODEL_ID + ".onnx"
MANIFEST_FILE = MODEL_ID + ".manifest.json"

FLOAT = 1  # TensorProto.DataType.FLOAT
INT64 = 7  # TensorProto.DataType.INT64
INT64_MAX = (1 << 63) - 1


# --- a minimal protobuf encoder ------------------------------------------------

def varint(n):
    if n < 0:
        n += 1 << 64  # int64 two's complement, as protobuf encodes it
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def key(field, wire):
    return varint((field << 3) | wire)


def f_varint(field, n):
    return key(field, 0) + varint(n)


def f_bytes(field, data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return key(field, 2) + varint(len(data)) + data


# --- ONNX messages (onnx.proto3 field numbers) ----------------------------------

def dim(value=None, param=None):
    # TensorShapeProto.Dimension: dim_value = 1, dim_param = 2
    return f_varint(1, value) if param is None else f_bytes(2, param)


def value_info(name, elem_type, dims):
    # ValueInfoProto: name = 1, type = 2
    # TypeProto: tensor_type = 1; TypeProto.Tensor: elem_type = 1, shape = 2
    # TensorShapeProto: dim = 1
    shape = b"".join(f_bytes(1, d) for d in dims)
    tensor = f_varint(1, elem_type) + f_bytes(2, shape)
    return f_bytes(1, name) + f_bytes(2, f_bytes(1, tensor))


def int64_tensor(name, values):
    # TensorProto: dims = 1, data_type = 2, name = 8, raw_data = 9
    raw = b"".join(struct.pack("<q", v) for v in values)
    return f_varint(1, len(values)) + f_varint(2, INT64) + f_bytes(8, name) + f_bytes(9, raw)


def float_scalar(name, value):
    # A 0-d tensor: no dims field.
    return f_varint(2, FLOAT) + f_bytes(8, name) + f_bytes(9, struct.pack("<f", value))


def node(op_type, inputs, outputs, name):
    # NodeProto: input = 1, output = 2, name = 3, op_type = 4
    return (
        b"".join(f_bytes(1, i) for i in inputs)
        + b"".join(f_bytes(2, o) for o in outputs)
        + f_bytes(3, name)
        + f_bytes(4, op_type)
    )


def model_bytes():
    nodes = [
        node("Slice", ["positions", "last_start", "last_end", "seq_axis"], ["last"], "take_last"),
        node("Slice", ["positions", "prev_start", "prev_end", "seq_axis"], ["prev"], "take_prev"),
        node("Mul", ["last", "two"], ["last_x2"], "double_last"),
        node("Sub", ["last_x2", "prev"], ["next_position"], "step"),
    ]
    initializers = [
        int64_tensor("last_start", [-1]),
        int64_tensor("last_end", [INT64_MAX]),
        int64_tensor("prev_start", [-2]),
        int64_tensor("prev_end", [-1]),
        int64_tensor("seq_axis", [1]),
        float_scalar("two", 2.0),
    ]
    # GraphProto: node = 1, name = 2, initializer = 5, input = 11, output = 12
    graph = (
        b"".join(f_bytes(1, n) for n in nodes)
        + f_bytes(2, "d440b_constant_velocity")
        + b"".join(f_bytes(5, t) for t in initializers)
        + f_bytes(11, value_info("positions", FLOAT, [dim(1), dim(param="n"), dim(3)]))
        + f_bytes(12, value_info("next_position", FLOAT, [dim(1), dim(1), dim(3)]))
    )
    # OperatorSetIdProto: domain = 1, version = 2 (the default domain is "")
    opset = f_bytes(1, "") + f_varint(2, OPSET)
    # ModelProto: ir_version = 1, producer_name = 2, producer_version = 3,
    # model_version = 5, doc_string = 6, graph = 7, opset_import = 8
    return (
        f_varint(1, IR_VERSION)
        + f_bytes(2, "nextgsim tools/d440b-model")
        + f_bytes(3, MODEL_VERSION)
        + f_varint(5, 1)
        + f_bytes(6, "d440b reference model (nextgsim#463): constant-velocity step; not trained")
        + f_bytes(7, graph)
        + f_bytes(8, opset)
    )


def reference(positions):
    """The graph's own arithmetic, for the golden outputs."""
    last, prev = positions[-1], positions[-2]
    return [2.0 * a - b for a, b in zip(last, prev)]


def golden(positions):
    flat = [float(c) for p in positions for c in p]
    return {
        "input_shape": [1, len(positions), 3],
        "input": flat,
        "output_shape": [1, 1, 3],
        "output": reference(positions),
    }


def manifest(data):
    return {
        "id": MODEL_ID,
        "version": MODEL_VERSION,
        "sha256": hashlib.sha256(data).hexdigest(),
        "opset": OPSET,
        "inputs": [{"name": "positions", "dtype": "float32", "shape": [1, -1, 3]}],
        "outputs": [{"name": "next_position", "dtype": "float32", "shape": [1, 1, 3]}],
        "golden": [
            golden([[0, 0, 0], [1, 2, 0], [3, 5, 1]]),
            golden([[10, 10, 0], [12, 9, 0]]),
            golden([[-4, 7, 2], [-2, 6, 2], [0, 5, 2], [2, 4, 2]]),
        ],
        "tolerance": 1e-5,
        "inference_name": "NG_RAN_MOBILITY_OPTIMIZATION",
        "training_type": "INITIAL_TRAINING",
        "producer": "nextgsim tools/d440b-model/make_reference_model.py (OAM role, file drop; not trained)",
    }


def main():
    check = "--check" in sys.argv[1:]
    data = model_bytes()
    text = json.dumps(manifest(data), indent=2) + "\n"
    want = {MODEL_FILE: data, MANIFEST_FILE: text.encode("utf-8")}
    bad = []
    for name, content in want.items():
        path = os.path.join(HERE, name)
        if check:
            try:
                with open(path, "rb") as f:
                    if f.read() != content:
                        bad.append(name)
            except FileNotFoundError:
                bad.append(name)
        else:
            with open(path, "wb") as f:
                f.write(content)
    if bad:
        print("d440b artefacts differ from the generator: " + ", ".join(bad), file=sys.stderr)
        return 1
    print(("checked " if check else "wrote ") + ", ".join(want) + " sha256=" + hashlib.sha256(data).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
