#!/usr/bin/env python
"""
mimo_bake.py -- export MIMO-UNet / MIMO-UNet+ (.pkl checkpoints) to a
vsncnn/NCNN_VK-compatible ONNX.

The official checkpoints (chosj95/MIMO-UNet, Google Drive links in the repo
README) are plain torch.save() files with a '.pkl' extension; the state dict
sits under the 'model' key. This script loads them, wraps the model so ONLY the
full-resolution output (outputs[2]) is exported (vsncnn requires a single
1- or 3-channel output; the two coarse outputs are training-time aids), exports
with the legacy tracer, and simplifies.

The architecture is already vsncnn-friendly: Conv, ConvTranspose, ReLU, Concat,
Mul, Add, and static-scale-factor F.interpolate (-> constant-scale Resize ->
ncnn Interp). No LayerNorm, no shape reads, no crop -- so no --ln-conv
equivalent is needed and dynamic H/W export is clean by construction.

Constraints:
  * input H/W must be multiples of 4 (two stride-2 downsamples + a x0.25
    internal branch). vsmlrt tile sizes that are multiples of 16 are safe.
  * requires MIMOUNet.py and layers.py from the official repo next to this
    script (with 'from .layers import *' changed to 'from layers import *').

Usage:
    python mimo_bake.py MIMO-UNet.pkl                 # dynamic, num_res=8
    python mimo_bake.py MIMO-UNetPlus.pkl --plus      # dynamic, num_res=20
    python mimo_bake.py MIMO-UNet.pkl --fixed 640 640 # fixed-shape variant
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class FullResOnly:
    """Wrapper module: forward returns only the full-resolution output."""
    def __new__(cls, inner):
        import torch.nn as nn

        class _W(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x)[2]
        return _W(inner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="MIMO-UNet .pkl (torch.save format)")
    ap.add_argument("--plus", action="store_true",
                    help="checkpoint is MIMO-UNet+ (num_res=20)")
    ap.add_argument("--fixed", nargs=2, type=int, metavar=("W", "H"),
                    default=None, help="bake a fixed W H instead of dynamic")
    ap.add_argument("--input-name", default="lq")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import torch
    from MIMOUNet import MIMOUNet, MIMOUNetPlus

    m = MIMOUNetPlus() if args.plus else MIMOUNet()
    ck = torch.load(args.checkpoint, map_location="cpu")
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    m.eval()
    w = FullResOnly(m).eval()

    fixed = args.fixed is not None
    # trace size for dynamic export is arbitrary (graph is shape-agnostic);
    # keep it small so export works on modest-RAM machines
    W, H = (args.fixed if fixed else (256, 256))
    if W % 4 or H % 4:
        sys.exit("W and H must be multiples of 4")

    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    d = os.path.dirname(os.path.abspath(args.checkpoint))
    tok = f"{W}x{H}" if fixed else "dynamic"
    # MIMO-UNet: two stride-2 stages + x0.25 branch -> tiles multiples of 4
    div_tok = "" if fixed else "_div4"
    exported = os.path.join(d, f"{stem}_{tok}{div_tok}.onnx")
    simple = os.path.join(d, f"{stem}_{tok}{div_tok}_simple.onnx")

    dyn = None if fixed else {args.input_name: {2: "h", 3: "w"},
                              "output": {2: "h", 3: "w"}}
    torch.onnx.export(w, torch.randn(1, 3, H, W), exported,
                      input_names=[args.input_name], output_names=["output"],
                      opset_version=args.opset, dynamo=False, dynamic_axes=dyn)

    import onnx
    from onnxsim import simplify
    sm, ok = simplify(onnx.load(exported))
    if not ok:
        sys.exit("onnxsim check failed")
    onnx.save(sm, simple)

    ops = sorted(set(n.op_type for n in sm.graph.node))
    bad = [o for o in ops if o in ("Shape", "Gather", "Mod", "Cast",
                                   "ConstantOfShape", "Expand", "Range")]
    print("ops:", ops)
    if bad:
        print("WARNING: vsncnn-hostile ops present:", bad, file=sys.stderr)
    print("wrote:", simple)


if __name__ == "__main__":
    main()
