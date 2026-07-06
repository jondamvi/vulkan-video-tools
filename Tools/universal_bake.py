#!/usr/bin/env python
"""
universal_bake.py -- export ANY OpenModelDB-family .pth checkpoint to a
vsncnn/NCNN_VK-compatible ONNX, with automatic architecture detection.

Uses spandrel (the loader inside chaiNNer) to fingerprint the state dict and
instantiate the correct architecture -- ESRGAN/RRDB, Compact/SRVGG, SPAN,
SwiftSRGAN, OmniSR, and dozens more -- so no per-model GitHub repo is needed.
Handles .pth, .pth.tar, .pt, .ckpt, and safetensors.

This does NOT replace the architecture-specific bakers:
  * bake.py       -- NAFNet (needs crop-free forward + --ln-conv rewrites)
  * mimo_bake.py  -- MIMO-UNet (needs full-res-output wrapper)
Those models aren't in OpenModelDB/spandrel's registry. Use this script for
everything that IS (the 1x deblur/sharpen family, upscalers, etc.).

Usage:
    pip install spandrel onnx onnxsim
    python universal_bake.py 1x-ReFocus-V3.pth
    python universal_bake.py swift_srgan_2x.pth.tar
    python universal_bake.py model.pth --fixed 512 512   # static variant
    python universal_bake.py model.pth --fp32-check      # extra ORT sanity run

Output: <stem>_<arch>_<Nx>_dynamic_simple.onnx  (+ op-compatibility report)
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _load_arch_scoped(entry, companions, subdirs):
    """Import a repo's architecture file (plus its same-dir companions) without
    polluting the process: repo files use generic names (arch_util.py,
    layers.py, local_arch.py ...) that collide across architectures, so after
    loading we evict those names from sys.modules and restore sys.path. The
    caller keeps direct class references; nothing else in the process can
    accidentally pick up the wrong repo's module later.

    Search order: each subdir under the script dir, then flat files next to
    the script. Returns the imported entry module, or None if not found."""
    import importlib
    names = [entry] + list(companions)
    cand = [os.path.join(HERE, s) for s in subdirs] + [HERE]
    src_dir = next((d for d in cand
                    if os.path.isfile(os.path.join(d, entry + ".py"))), None)
    if src_dir is None:
        return None
    saved = {n: sys.modules.pop(n, None) for n in names}
    sys.path.insert(0, src_dir)
    loaded = {}
    try:
        loaded[entry] = importlib.import_module(entry)
        for n in companions:
            if n in sys.modules:            # pulled in by the entry's imports
                loaded[n] = sys.modules[n]
    finally:
        try:
            sys.path.remove(src_dir)
        except ValueError:
            pass
        for n in names:
            sys.modules.pop(n, None)
            if saved[n] is not None:
                sys.modules[n] = saved[n]
    return loaded


def _fp32ify(model):
    """Convert an fp16-weight ONNX to fp32 in place. vsncnn cannot read fp16
    tensors ('Unknown data type 0'); runtime precision is a backend decision,
    never a file property. Returns the number of tensors converted."""
    import numpy as np
    from onnx import numpy_helper, TensorProto
    n = 0
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
            n += 1
    pools = (list(model.graph.input) + list(model.graph.output)
             + list(model.graph.value_info))
    for vi in pools:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
    for node in model.graph.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == TensorProto.FLOAT16:
                    a.i = TensorProto.FLOAT
        elif node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value" and \
                        a.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(a.t).astype(np.float32)
                    a.t.CopyFrom(numpy_helper.from_array(arr))
                    n += 1
    return n


class _LocalDescriptor:
    """spandrel-shaped shim for architectures spandrel doesn't know."""
    def __init__(self, model, name, scale=1, mult=1):
        self.model, self.scale = model, scale
        self.input_channels = self.output_channels = 3
        self.architecture = type("A", (), {"name": name})()
        self.size_requirements = type(
            "S", (), {"multiple_of": mult, "minimum": 0, "square": False})()


def _build_nafnet(sd):
    """Build an export-ready NAFNet from a raw state dict. Linear recipe:

    1. LOAD ARCH CODE  -- from archs/nafnet, nafnet_standalone, or flat files
                          next to this script (NAFNet_arch.py + arch_util.py +
                          local_arch.py), namespace-isolated so the generic
                          file names cannot collide with other architectures.
    2. DERIVE CONFIG   -- width / enc / middle / dec read from the state dict.
    3. LOAD WEIGHTS    -- strict=True; any mismatch aborts loudly.
    4. CROP-FREE FWD   -- removes `B,C,H,W = inp.shape` + final crop, which
                          otherwise export as Shape/Gather/Mod (vsncnn-fatal).
    5. LN->CONV        -- rewrites every LayerNorm2d as 1x1 convs + elementwise
                          ops; the stock ReduceMean/Pow decomposition produces
                          NaN on ncnn Vulkan. Numerically identical (~3e-7).

    Returns a _LocalDescriptor(model, "NAFNet", scale=1, mult=16).
    """
    import types
    import torch
    import torch.nn as nn

    print("state dict fingerprint matches NAFNet")

    # -- 1. load architecture code (isolated) --------------------------------
    mods = _load_arch_scoped(
        entry="NAFNet_arch",
        companions=("arch_util", "local_arch"),
        subdirs=(os.path.join("archs", "nafnet"), "nafnet_standalone"))
    if mods is None:
        sys.exit("NAFNet needs its architecture code. Put NAFNet_arch.py, "
                 "arch_util.py and local_arch.py into archs\\nafnet\\ next "
                 "to this script (nafnet_standalone\\ or flat also work).")
    NAFNet = mods["NAFNet_arch"].NAFNet
    LayerNorm2d = (getattr(mods["NAFNet_arch"], "LayerNorm2d", None)
                   or getattr(mods.get("arch_util"), "LayerNorm2d", None))
    if LayerNorm2d is None:
        sys.exit("LayerNorm2d not found in arch_util.py / NAFNet_arch.py -- "
                 "wrong architecture files?")

    # -- 2. derive config from the weights ------------------------------------
    keys = set(sd)

    def blocks(prefix):
        top = {}
        for k in keys:
            if k.startswith(prefix):
                i, j = k[len(prefix):].split(".")[:2]
                top[int(i)] = max(top.get(int(i), -1), int(j))
        return [top[i] + 1 for i in sorted(top)]

    width = sd["intro.weight"].shape[0]
    enc = blocks("encoders.")
    dec = blocks("decoders.")
    middle = 1 + max(int(k.split(".")[1]) for k in keys
                     if k.startswith("middle_blks."))
    print(f"derived config: width={width} enc={enc} middle={middle} dec={dec}")

    # -- 3. weights ------------------------------------------------------------
    m = NAFNet(img_channel=3, width=width, middle_blk_num=middle,
               enc_blk_nums=enc, dec_blk_nums=dec)
    m.load_state_dict(sd, strict=True)
    m.eval()

    # -- 4. crop-free forward ---------------------------------------------------
    m.check_image_size = (lambda x: x)

    def fwd(self, inp):
        x = self.intro(inp)
        skips = []
        for enc_blk, down in zip(self.encoders, self.downs):
            x = enc_blk(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for dec_blk, up, sk in zip(self.decoders, self.ups, skips[::-1]):
            x = dec_blk(up(x) + sk)
        return self.ending(x) + inp

    m.forward = types.MethodType(fwd, m)

    # -- 5. LayerNorm2d -> conv rewrite -----------------------------------------
    def to_conv(ln):
        C = ln.weight.numel()
        mean_c = nn.Conv2d(C, C, 1, bias=False)
        scale_c = nn.Conv2d(C, C, 1, bias=True)
        with torch.no_grad():
            mean_c.weight.fill_(1.0 / C)
            scale_c.weight.zero_()
            scale_c.weight[range(C), range(C), 0, 0] = ln.weight.detach()
            scale_c.bias.copy_(ln.bias.detach())
        ln.add_module("_ln_mean_conv", mean_c)
        ln.add_module("_ln_scale_conv", scale_c)

        def lf(self, x):
            mu = self._ln_mean_conv(x)
            d = x - mu
            var = self._ln_mean_conv(d * d)
            return self._ln_scale_conv(d / torch.sqrt(var + self.eps))

        ln.forward = types.MethodType(lf, ln)

    n = 0
    for mod in m.modules():
        if isinstance(mod, LayerNorm2d):
            to_conv(mod)
            n += 1
    print(f"applied automatically: crop-free forward + LayerNorm->conv "
          f"rewrite ({n} instances)")

    return _LocalDescriptor(m, "NAFNet", scale=1, mult=2 ** len(enc))


def _build_deblurganv2(sd):
    """Build an export-ready DeblurGAN-v2 fpn_mobilenet generator.

    Checkpoint facts (VITA-Group/DeblurGANv2): torch pickle despite the .h5
    extension, state dict under 'model', keys carry a 'module.' DataParallel
    prefix. The repo files use 'from models.X import ...' package imports and
    FPN.__init__(pretrained=True) tries to torch.load an ImageNet backbone
    file at CONSTRUCTION -- so we rewrite imports, construct with
    pretrained=False, then load the checkpoint weights.

    Export surgery: the model computes in [-1, 1] (tanh residual + clamp), so
    a range-adapter wrapper bakes x*2-1 / (y+1)/2 into the graph as Mul/Add,
    keeping the pipeline's [0, 1] RGBS convention. InstanceNorm2d layers were
    trained with track_running_stats=True, so in eval they use fixed running
    stats == BatchNorm inference -> exported as foldable BatchNormalization
    (no runtime norm ops survive). FPN downsamples 5x -> inputs multiple of
    32.
    """
    import importlib.util
    import types
    import torch
    import torch.nn as nn
    import functools

    print("state dict fingerprint matches DeblurGAN-v2 (fpn_mobilenet)")

    # locate arch files (archs/deblurganv2 preferred, flat fallback)
    src_dir = None
    for cand in (os.path.join(HERE, "archs", "deblurganv2"), HERE):
        if os.path.isfile(os.path.join(cand, "fpn_mobilenet.py")) and \
           os.path.isfile(os.path.join(cand, "mobilenet_v2.py")):
            src_dir = cand
            break
    if src_dir is None:
        sys.exit("DeblurGAN-v2 needs fpn_mobilenet.py and mobilenet_v2.py "
                 "from VITA-Group/DeblurGANv2 (models/ dir) in "
                 "archs\\deblurganv2\\ next to this script.")

    # scoped load with 'models.' package-prefix imports rewritten
    saved = {n: sys.modules.pop(n, None)
             for n in ("fpn_mobilenet", "mobilenet_v2")}
    sys.path.insert(0, src_dir)
    try:
        mv2_spec = importlib.util.spec_from_file_location(
            "mobilenet_v2", os.path.join(src_dir, "mobilenet_v2.py"))
        mv2 = importlib.util.module_from_spec(mv2_spec)
        mv2_spec.loader.exec_module(mv2)
        sys.modules["mobilenet_v2"] = mv2
        code = open(os.path.join(src_dir, "fpn_mobilenet.py"),
                    encoding="utf-8").read().replace(
            "from models.mobilenet_v2 import", "from mobilenet_v2 import")
        spec = importlib.util.spec_from_loader("fpn_mobilenet", loader=None)
        fm = importlib.util.module_from_spec(spec)
        exec(compile(code, "fpn_mobilenet.py", "exec"), fm.__dict__)
    finally:
        try:
            sys.path.remove(src_dir)
        except ValueError:
            pass
        for n in ("fpn_mobilenet", "mobilenet_v2"):
            sys.modules.pop(n, None)
            if saved[n] is not None:
                sys.modules[n] = saved[n]

    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,
                                   track_running_stats=True)
    g = fm.FPNMobileNet(norm_layer=norm_layer, pretrained=False)
    clean = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    g.load_state_dict(clean, strict=True)
    g.eval()

    # torch's ONNX symbolic for instance_norm cannot handle dynamic axes,
    # but eval-mode InstanceNorm2d with track_running_stats=True is a FIXED
    # per-channel affine: y = (x - mean) / sqrt(var + eps). We express it as
    # a DIAGONAL 1x1 CONV (weight=diag(a), bias=b) rather than Mul/Add:
    # scalar/broadcast BinaryOps are avoided throughout this handler because
    # ncnn's with_scalar BinaryOp path is crash-prone (reproduced: SIGSEGV in
    # the pip ncnn CPU build), while Convolution is its most battle-tested
    # layer. onnxsim folds these into neighbors where possible.
    def _diag_conv(a, b):
        C = a.numel()
        conv = nn.Conv2d(C, C, 1, bias=True)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[range(C), range(C), 0, 0] = a
            conv.bias.copy_(b)
        return conv

    def _swap_inorm(mod):
        n = 0
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.InstanceNorm2d):
                if not child.track_running_stats:
                    sys.exit("InstanceNorm without running stats cannot be "
                             "frozen -- unsupported checkpoint variant")
                a = (child.running_var + child.eps).rsqrt()
                b = -child.running_mean * a
                setattr(mod, name, _diag_conv(a, b))
                n += 1
            else:
                n += _swap_inorm(child)
        return n

    with torch.no_grad():
        n_in = _swap_inorm(g)
    print(f"applied automatically: {n_in} eval-frozen InstanceNorm layers "
          "rewritten as diagonal 1x1 convs")

    class RangeAdapter(nn.Module):
        """[0,1] pipeline I/O around the model's native [-1,1] domain,
        expressed as 1x1 convs (x*2-1 in, (y+1)/2 out) -- no scalar
        BinaryOps anywhere in the exported graph."""
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            t3 = torch.ones(3)
            self.pre = _diag_conv(t3 * 2.0, t3 * -1.0)
            self.post = _diag_conv(t3 * 0.5, t3 * 0.5)

        def forward(self, x):
            return self.post(self.inner(self.pre(x)))

    print("applied automatically: [-1,1] range adapters baked in as 1x1 convs")
    return _LocalDescriptor(RangeAdapter(g).eval(), "DeblurGANv2_mobilenet",
                            scale=1, mult=32)


HOSTILE = ("Shape", "Gather", "Mod", "Cast", "ConstantOfShape", "Expand",
           "Range", "NonZero", "Loop", "If", "ScatterND", "GridSample")

# op types verified safe for vsncnn / ncnn Vulkan in this repo's pipelines
KNOWN_GOOD = {"Conv", "ConvTranspose", "Add", "Sub", "Mul", "Div", "Relu",
              "LeakyRelu", "PRelu", "Sigmoid", "Tanh", "Clip", "Concat",
              "Slice", "DepthToSpace", "SpaceToDepth", "Resize", "MaxPool",
              "AveragePool", "GlobalAveragePool", "Sqrt", "Pad", "HardSigmoid",
              "HardSwish", "Softmax", "Constant", "Identity"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--fixed", nargs=2, type=int, metavar=("W", "H"),
                    default=None, help="bake fixed W H instead of dynamic")
    ap.add_argument("--input-name", default="lq")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp32-check", action="store_true",
                    help="run the simplified ONNX in onnxruntime vs PyTorch")
    ap.add_argument("--ncnn-fp16", action="store_true",
                    help="with --ncnn: also run ncnnoptimize to write fp16-"
                         "storage .param/.bin (half size/bandwidth; fp32 math "
                         "on GPUs without fp16 arithmetic like Pascal). Needs "
                         "ncnnoptimize.exe on PATH (ships in ncnn release "
                         "zips). For the ONNX/vsmlrt path use "
                         "Backend.NCNN_VK(fp16=True) instead -- never bake "
                         "fp16 into ONNX.")
    ap.add_argument("--ncnn", action="store_true",
                    help="also convert the simplified ONNX to ncnn .param/.bin "
                         "via pnnx (pip install pnnx). Only needed for direct "
                         "ncnn use (e.g. a param/bin loader); vsmlrt/vsncnn "
                         "consumes the ONNX itself and does NOT need this.")
    args = ap.parse_args()

    import torch
    from spandrel import ModelLoader


    def _try_local_archs(sd):
        """Fingerprint state dicts of models outside spandrel's registry.
        Requires the architecture .py files next to this script."""
        keys = set(sd)

        # DeblurGAN-v2 fpn_mobilenet (VITA-Group): unmistakable FPN keys
        # (after the DataParallel 'module.' prefix these checkpoints carry)
        _dg = {k[7:] if k.startswith("module.") else k for k in keys}
        if "fpn.lateral4.weight" in _dg and "head1.block0.weight" in _dg:
            return _build_deblurganv2(sd)

        # NAFNet (megvii-research) -- see _build_nafnet() below for the
        # full recipe. Handled locally (never via spandrel) because a valid
        # NCNN/vsncnn export REQUIRES graph surgery.
        if "intro.weight" in keys and "ending.weight" in keys and \
           any(k.startswith("middle_blks.") for k in keys):
            return _build_nafnet(sd)

        # MIMO-UNet / MIMO-UNet+ (chosj95): unmistakable module names
        if any(k.startswith("AFFs.") for k in keys) and \
           any(k.startswith("SCM1.") for k in keys):
            print("state dict fingerprint matches MIMO-UNet")
            _mods = None
            try:
                _mods = _load_arch_scoped(
                    "MIMOUNet", ("layers",),
                    (os.path.join("archs", "mimo"), "mimo"))
            except ImportError:
                # repo file uses package-relative 'from .layers import *';
                # load with the import rewritten, same scoped discipline
                import importlib.util
                for _d in (os.path.join(HERE, "archs", "mimo"),
                           os.path.join(HERE, "mimo"), HERE):
                    _fp = os.path.join(_d, "MIMOUNet.py")
                    if os.path.isfile(_fp):
                        _saved = {n: sys.modules.pop(n, None)
                                  for n in ("MIMOUNet", "layers")}
                        sys.path.insert(0, _d)
                        try:
                            code = open(_fp, encoding="utf-8").read().replace(
                                "from .layers import", "from layers import")
                            spec = importlib.util.spec_from_loader(
                                "MIMOUNet", loader=None)
                            m_ = importlib.util.module_from_spec(spec)
                            exec(compile(code, _fp, "exec"), m_.__dict__)
                            _mods = {"MIMOUNet": m_}
                        finally:
                            try:
                                sys.path.remove(_d)
                            except ValueError:
                                pass
                            for n in ("MIMOUNet", "layers"):
                                sys.modules.pop(n, None)
                                if _saved[n] is not None:
                                    sys.modules[n] = _saved[n]
                        break
            if _mods is None:
                sys.exit(
                    "MIMO-UNet checkpoints need the architecture code. Place\n"
                    "these two files (flat next to this script, or in an "
                    "archs/mimo subfolder):\n"
                    "  https://raw.githubusercontent.com/chosj95/MIMO-UNet/main/models/MIMOUNet.py\n"
                    "  https://raw.githubusercontent.com/chosj95/MIMO-UNet/main/models/layers.py")
            MIMOUNet = _mods["MIMOUNet"].MIMOUNet
            import torch.nn as nn
            num_res = 1 + max(int(k.split(".")[3]) for k in keys
                              if k.startswith("Encoder.0.layers."))
            inner = MIMOUNet(num_res=num_res)
            inner.load_state_dict(sd, strict=True)

            class _FullRes(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    return self.m(x)[2]     # full-resolution output only

            name = "MIMOUNetPlus" if num_res == 20 else "MIMOUNet"
            print(f"local fingerprint: {name} (num_res={num_res})")
            return _LocalDescriptor(_FullRes(inner), name, scale=1, mult=4)
        return None

    def _unwrap(ck):
        if isinstance(ck, dict):
            for key in ("model", "params_ema", "params", "state_dict",
                        "generator", "model_state_dict"):
                if key in ck and isinstance(ck[key], dict):
                    return ck[key]
        return ck

    is_onnx = args.checkpoint.lower().endswith(".onnx")
    if is_onnx:
        # Pre-existing ONNX cleanup mode: fp32-ify, simplify, audit, probe,
        # optional --ncnn -- no architecture code needed. Scale and channels
        # are detected empirically from the file itself.
        import onnx as _oximport
        import numpy as _npimport
        import onnxruntime as _ortimport
        _src_model = _oximport.load(args.checkpoint)
        n16 = _fp32ify(_src_model)
        if n16:
            print(f"fp32-ified {n16} fp16 tensors (vsncnn cannot read fp16 "
                  "files; runtime precision belongs to the backend)")
        args.input_name = _src_model.graph.input[0].name
        _ch = _src_model.graph.input[0].type.tensor_type.shape.dim[1]
        _ch = _ch.dim_value if _ch.HasField("dim_value") else 3
        # empirical scale: run the ORIGINAL model once (self-padding graphs
        # accept any size, so 64 works)
        _so0 = _ortimport.SessionOptions()
        _so0.log_severity_level = 4
        _sc = 1
        try:
            _s0 = _ortimport.InferenceSession(
                _src_model.SerializeToString(), _so0,
                providers=["CPUExecutionProvider"])
            _y0 = _s0.run(None, {args.input_name: _npimport.zeros(
                (1, _ch, 64, 64), _npimport.float32)})[0]
            _sc = max(1, round(_y0.shape[2] / 64))
        except Exception:
            for _t in (64, 128, 256):
                try:
                    _y0 = _s0.run(None, {args.input_name: _npimport.zeros(
                        (1, _ch, _t, _t), _npimport.float32)})[0]
                    _sc = max(1, round(_y0.shape[2] / _t))
                    break
                except Exception:
                    continue
        desc = _LocalDescriptor(None, "", scale=_sc, mult=1)
        desc.input_channels = desc.output_channels = _ch
        print(f"onnx input: {_ch}ch, detected scale {_sc}x")

    loader = ModelLoader()
    if is_onnx:
        loader = None
    desc = desc if is_onnx else None
    sd = None
    try:
        if not is_onnx:
            sd = _unwrap(torch.load(args.checkpoint, map_location="cpu",
                                    weights_only=False))
    except Exception:
        pass                     # not a torch pickle (e.g. safetensors)
    if isinstance(sd, dict):
        desc = _try_local_archs(sd)     # local archs take precedence:
                                        # spandrel's NAFNet is export-unsafe
    if desc is None:
        if sd is not None:
            desc = loader.load_from_state_dict(sd)
        else:
            desc = loader.load_from_file(args.checkpoint)
    arch = desc.architecture.name
    scale = getattr(desc, "scale", 1)
    print(f"detected: {arch}  scale={scale}x  "
          f"in={desc.input_channels}ch out={desc.output_channels}ch")
    sr = getattr(desc, "size_requirements", None)
    if sr is not None:
        mult = getattr(sr, "multiple_of", 1) or 1
        mini = getattr(sr, "minimum", 0) or 0
        sq = getattr(sr, "square", False)
        print(f"tile constraint: multiple of {mult}"
              + (f", minimum {mini}px" if mini else "")
              + (", square tiles only" if sq else "")
              + ("  (i.e. no constraint)" if mult == 1 and not mini and not sq
                 else ""))
    if desc.input_channels != 3 or desc.output_channels != 3:
        print("WARNING: vsncnn expects 1 or 3 channel I/O; "
              f"this model is {desc.input_channels}->{desc.output_channels}",
              file=sys.stderr)

    m = None
    if not is_onnx:
        m = desc.model.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    fixed = args.fixed is not None
    W, H = (args.fixed if fixed else (256, 256))  # trace size; arbitrary if dynamic

    def _sane(s):
        # keep original letter casing; only map separators to underscores
        return "".join(c if c.isalnum() else "_" for c in s).strip("_")

    def _cmp(s):
        # case/separator-insensitive form, for dedupe comparisons only
        return "".join(c for c in s.lower() if c.isalnum())

    stem = os.path.basename(args.checkpoint)
    for ext in (".pth.tar", ".pth", ".pt", ".ckpt", ".safetensors", ".pkl",
                ".h5", ".onnx"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    if is_onnx and stem.lower().endswith("_fp16"):
        stem = stem[:-5]        # weights are fp32 after cleanup
    d = os.path.dirname(os.path.abspath(args.checkpoint))
    tok = f"{args.fixed[0]}x{args.fixed[1]}" if fixed else "dynamic"

    # dedupe: skip tokens already present in the checkpoint's own name
    parts = [_sane(stem)]
    if _cmp(arch) not in _cmp(stem):
        parts.append(_sane(arch).upper())
    if f"{scale}x" not in _cmp(stem) and f"x{scale}" not in _cmp(stem):
        parts.append(f"{scale}x")
    base = "_".join(parts) + f"_{tok}"      # div token appended after probing
    exported = os.path.join(d, base + ".onnx")
    simple = os.path.join(d, base + "_simple.onnx")

    if is_onnx:
        import onnx
        onnx.save(_src_model, exported)
    else:
        dyn = None if fixed else {args.input_name: {2: "h", 3: "w"},
                                  "output": {2: "h", 3: "w"}}
        torch.onnx.export(m, torch.rand(1, desc.input_channels, H, W),
                          exported, input_names=[args.input_name],
                          output_names=["output"], opset_version=args.opset,
                          dynamo=False, dynamic_axes=dyn)

    import onnx
    from collections import Counter
    from onnxsim import simplify
    raw = onnx.load(exported)
    before = Counter(n.op_type for n in raw.graph.node)
    if is_onnx and fixed:
        sm, ok = simplify(raw, overwrite_input_shapes={
            args.input_name: [1, desc.input_channels, H, W]})
    else:
        sm, ok = simplify(raw)
    if not ok:
        sys.exit("onnxsim check failed (simplified model did not validate)")

    if is_onnx and not fixed:
        # Self-padding graphs (in-graph pad-to-/N + unshuffle) keep Shape/Mod/
        # Gather under dynamic simplification. Rescue: fold statically at 256,
        # then test whether re-dynamizing survives (it does when no Reshape
        # baked spatial dims); if not, ship the static 256 model and say so.
        hostile_now = [o for o in set(n.op_type for n in sm.graph.node)
                       if o in HOSTILE]
        if hostile_now:
            print(f"dynamic cleanup left hostile ops {hostile_now}; "
                  "attempting static fold + re-dynamize rescue")
            st, ok2 = simplify(raw, overwrite_input_shapes={
                args.input_name: [1, desc.input_channels, 256, 256]})
            if not ok2:
                sys.exit("static rescue failed onnxsim validation")
            for vi, sym in ((st.graph.input[0], ("h", "w")),
                            (st.graph.output[0], ("oh", "ow"))):
                dd = vi.type.tensor_type.shape.dim
                for k, name in ((2, sym[0]), (3, sym[1])):
                    dd[k].ClearField("dim_value")
                    dd[k].dim_param = name
            import io as _io
            import numpy as _np
            import onnxruntime as _ort
            _so = _ort.SessionOptions()
            _so.log_severity_level = 4
            redyn_ok = True
            try:
                _sx = _ort.InferenceSession(st.SerializeToString(), _so,
                                            providers=["CPUExecutionProvider"])
                for s0 in (256, 192):
                    _y = _sx.run(None, {args.input_name: _np.zeros(
                        (1, desc.input_channels, s0, s0), _np.float32)})[0]
                    if _y.shape[2] != s0 * scale:
                        redyn_ok = False
            except Exception:
                redyn_ok = False
            if redyn_ok:
                sm = st
                print("rescue succeeded: clean DYNAMIC model")
            else:
                # static-only: re-pin 256 dims and rename accordingly
                for vi, v in ((st.graph.input[0], 256),
                              (st.graph.output[0], 256 * scale)):
                    dd = vi.type.tensor_type.shape.dim
                    for k in (2, 3):
                        dd[k].ClearField("dim_param")
                        dd[k].dim_value = v
                sm = st
                fixed = True
                W = H = 256
                new_base = base.replace("_dynamic", "_256x256")
                os.replace(exported, os.path.join(
                    d, os.path.basename(exported).replace(base, new_base, 1)))
                exported = os.path.join(
                    d, os.path.basename(exported).replace(base, new_base, 1))
                base = new_base
                simple = os.path.join(d, base + "_simple.onnx")
                print("rescue: graph is STATIC-ONLY (baked reshapes) -- "
                      "emitted at 256x256. Tile size must be exactly 256x256; "
                      "rerun with --fixed W H for a different tile.")
    onnx.save(sm, simple)
    after = Counter(n.op_type for n in sm.graph.node)
    print("onnxsim simplification (original -> simplified):")
    for op in sorted(set(before) | set(after)):
        b, a = before.get(op, 0), after.get(op, 0)
        mark = "" if b == a else "   <- folded" if a < b else "   <- added"
        print(f"  {op:<18} {b:>5} -> {a:<5}{mark}")

    # Empirical input-divisibility probe on the ARTIFACT itself. Architecture
    # metadata lies here by design: self-padding modules (NAFNet et al.) accept
    # any size as PyTorch models, but tracing folds the padding away, leaving a
    # /N-strict graph. ncnn will NOT error on a violated constraint -- it
    # silently computes misaligned skip connections (verified: 0.28 max
    # deviation at h%16=8) -- so the label must come from measurement.
    div = None
    if not fixed:
        import numpy as np
        import onnxruntime as ort
        _so = ort.SessionOptions()
        _so.log_severity_level = 4      # probe intentionally triggers kernel
                                        # failures at illegal sizes; don't let
                                        # ORT spam them to stderr
        sess = ort.InferenceSession(
            simple, _so, providers=["CPUExecutionProvider"])
        for cand in (1, 2, 4, 8, 16, 32, 64):
            s0 = 64 + cand              # divisible by cand, not by 2*cand
            try:
                r0 = sess.run(None, {args.input_name: np.zeros(
                    (1, desc.input_channels, s0, s0), np.float32)})[0]
                if r0.shape[2] == s0 * scale and r0.shape[3] == s0 * scale:
                    div = cand
                    break
            except Exception:
                continue
        if div is None:
            print("WARNING: divisor probe inconclusive (model refused all "
                  "test sizes 65..128); labeling as div64 conservatively",
                  file=sys.stderr)
            div = 64
        div_tok = f"_div{div}" if div > 1 else "_anytile"
        meta_mult = getattr(sr, "multiple_of", 1) or 1 if sr is not None else 1
        print(f"measured input divisor: {div} (metadata claimed {meta_mult})"
              + ("  <- metadata was WRONG; trust the measurement"
                 if div != meta_mult else ""))
        new_base = base + div_tok
        for old_p, attr in ((exported, "exported"), (simple, "simple")):
            new_p = os.path.join(d, os.path.basename(old_p).replace(
                base, new_base, 1))
            os.replace(old_p, new_p)
            if attr == "exported":
                exported = new_p
            else:
                simple = new_p
        base = new_base

    ops = sorted(set(n.op_type for n in sm.graph.node))
    bad = [o for o in ops if o in HOSTILE]
    unknown = [o for o in ops if o not in KNOWN_GOOD and o not in HOSTILE]
    print("ops:", ops)
    if bad:
        print(f"VERDICT: NOT vsncnn-compatible -- hostile ops: {bad}",
              file=sys.stderr)
    elif {"ReduceMean", "Pow"} <= set(ops) or "LayerNormalization" in ops:
        print("VERDICT: contains a LayerNorm decomposition (ReduceMean/Pow) -- "
              "known to produce NaN (green frames) on vsncnn/NCNN_VK. For "
              "NAFNet-class models use bake.py --ln-conv instead. (pnnx may "
              "fuse it into a native ncnn LayerNorm layer, so the --ncnn "
              "param/bin route can still work.)", file=sys.stderr)
    elif unknown:
        print(f"VERDICT: probably OK; unverified ops worth a test run: {unknown}")
    else:
        print("VERDICT: all ops in the verified-safe set")
    print("wrote:", simple)

    if args.ncnn:
        import shutil, subprocess
        exe = shutil.which("pnnx")
        if exe is None:
            print("--ncnn: 'pnnx' executable not found; pip install pnnx",
                  file=sys.stderr)
        else:
            trace_w, trace_h = ((W, H) if fixed else (256, 256))
            dv_chk = (2 if fixed else (div or 1))
            # verification size: the fixed size itself, or a /div dynamic size
            ver_w, ver_h = (trace_w, trace_h) if fixed else (
                (64 if 64 % (div or 1) == 0
                 else ((64 // div) + 1) * div),) * 2

            # stale outputs from earlier runs mixing with new ones is a known
            # source of 'load_model failed' -- clear targets first
            for suf in ("_ncnn.param", "_ncnn.bin",
                        "_ncnn_fp16.param", "_ncnn_fp16.bin"):
                try:
                    os.remove(os.path.join(d, base + suf))
                except OSError:
                    pass

            static_tmp = os.path.join(d, base + "_pnnxtmp.onnx")
            if fixed:
                onnx.save(sm, static_tmp)     # already static at W x H
            else:
                st, ok2 = simplify(raw, overwrite_input_shapes={
                    args.input_name:
                    [1, desc.input_channels, trace_h, trace_w]})
                if not ok2:
                    sys.exit("--ncnn: static re-simplification failed")
                onnx.save(st, static_tmp)

            # fp16=0 -> exact fp32 weights in the .bin (pnnx defaults to
            # fp16 weights, adding ~0.5% deviation on deep nets and muddying
            # verification). Runtime precision on Vulkan is unaffected;
            # --ncnn-fp16 remains the explicit half-size variant.
            r = subprocess.run([exe, os.path.basename(static_tmp), "fp16=0"],
                               cwd=d, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            os.remove(static_tmp)
            produced = []
            for f in os.listdir(d):
                if f.startswith(base + "_pnnxtmp") and \
                   (f.endswith(".param") or f.endswith(".bin")):
                    if ".ncnn." not in f:
                        os.remove(os.path.join(d, f))
                        continue
                    nf = f.replace("_pnnxtmp", "").replace(".ncnn", "_ncnn")
                    os.replace(os.path.join(d, f), os.path.join(d, nf))
                    produced.append(nf)
            for f in os.listdir(d):
                if "_pnnxtmp" in f:
                    try:
                        os.remove(os.path.join(d, f))
                    except OSError:
                        pass
            if not produced:
                print("--ncnn: pnnx produced no param/bin. pnnx output tail:",
                      file=sys.stderr)
                print((r.stderr or r.stdout or "")[-600:], file=sys.stderr)
            else:
                pin = os.path.join(d, base + "_ncnn.param")
                bi = os.path.join(d, base + "_ncnn.bin")

                # ---- MANDATORY VERIFICATION: a pair that fails is DELETED,
                # ---- never shipped. Requires the python 'ncnn' package.
                def _verify(pp, bb, label, soft_max, hard_max, mean_tol):
                    try:
                        import ncnn as _ncnn
                        import numpy as _np
                        import onnxruntime as _ort
                    except ImportError:
                        print("!! {} UNVERIFIED -- python 'ncnn' package "
                              "missing (pip install ncnn). Do NOT trust this "
                              "artifact until verified.".format(label),
                              file=sys.stderr)
                        return None
                    net = _ncnn.Net()
                    if net.load_param(pp) != 0 or net.load_model(bb) != 0:
                        return "failed to load (truncated/mismatched files)"
                    so = _ort.SessionOptions()
                    so.log_severity_level = 4
                    sess = _ort.InferenceSession(
                        simple, so, providers=["CPUExecutionProvider"])
                    x = __import__("numpy").random.rand(
                        desc.input_channels, ver_h, ver_w).astype("float32")
                    want = sess.run(None, {args.input_name: x[None]})[0][0]
                    ex = net.create_extractor()
                    ex.input("in0", _ncnn.Mat(
                        __import__("numpy").ascontiguousarray(x)))
                    ret, out = ex.extract("out0")
                    if ret != 0:
                        return "extract failed (ret={})".format(ret)
                    got = __import__("numpy").array(out)
                    if got.shape != want.shape:
                        return "shape mismatch {} vs {}".format(
                            got.shape, want.shape)
                    npx = __import__("numpy")
                    if not npx.isfinite(got).all():
                        return "non-finite output (NaN/Inf)"
                    dmax = float(npx.abs(got - want).max())
                    dmean = float(npx.abs(got - want).mean())
                    # Two-metric gate. Real corruption (misaligned skips,
                    # scrambled planes) inflates BOTH metrics by orders of
                    # magnitude (observed: max 0.28, mean ~0.03). Precision /
                    # accumulation-order drift keeps the mean tiny even when
                    # the max ticks up on very deep graphs.
                    if dmax > hard_max or dmean > mean_tol:
                        return ("output diverges structurally "
                                "(max {:.3g}, mean {:.3g})".format(
                                    dmax, dmean))
                    if dmax > soft_max:
                        print("{} PASSED WITH ELEVATED DRIFT at {}x{}: "
                              "max {:.3g}, mean {:.3g} -- numerically noisy "
                              "but structurally sound; eyeball the output"
                              .format(label, ver_w, ver_h, dmax, dmean))
                        return ""
                    print("{} VERIFIED vs ONNX at {}x{}: max diff {:.3g}, "
                          "mean {:.3g}".format(label, ver_w, ver_h,
                                               dmax, dmean))
                    return ""
                    print("{} VERIFIED vs ONNX at {}x{}: max diff {:.3g}, "
                          "mean {:.3g}".format(label, ver_w, ver_h,
                                               dmax, dmean))
                    return ""

                bad = _verify(pin, bi, "ncnn pair",
                              soft_max=2e-2, hard_max=0.1, mean_tol=2e-3)
                if bad:
                    os.remove(pin)
                    os.remove(bi)
                    print("!! ncnn pair CORRUPT ({}) -- files DELETED. "
                          "pnnx output tail:".format(bad), file=sys.stderr)
                    print((r.stderr or r.stdout or "")[-600:],
                          file=sys.stderr)
                elif bad == "":
                    print("ncnn files:", base + "_ncnn.param,",
                          base + "_ncnn.bin",
                          "(blobs in0/out0; any /{} size at runtime)".format(
                              dv_chk) if not fixed else
                          "(blobs in0/out0; run at exactly {}x{})".format(
                              trace_w, trace_h))

                if args.ncnn_fp16 and os.path.isfile(pin):
                    opt = shutil.which("ncnnoptimize")
                    if opt is None:
                        print("--ncnn-fp16: ncnnoptimize not on PATH; "
                              "skipping", file=sys.stderr)
                    else:
                        po = os.path.join(d, base + "_ncnn_fp16.param")
                        bo = os.path.join(d, base + "_ncnn_fp16.bin")
                        r2 = subprocess.run([opt, pin, bi, po, bo, "1"],
                                            capture_output=True, text=True,
                                            encoding="utf-8",
                                            errors="replace")
                        if not (os.path.isfile(po) and os.path.isfile(bo)):
                            print("--ncnn-fp16: ncnnoptimize produced "
                                  "nothing:", (r2.stderr or r2.stdout
                                               or "")[-300:],
                                  file=sys.stderr)
                        else:
                            bad2 = _verify(po, bo, "fp16 pair",
                                           soft_max=5e-2, hard_max=0.2,
                                           mean_tol=5e-3)
                            if bad2:
                                os.remove(po)
                                os.remove(bo)
                                print("!! fp16 pair CORRUPT ({}) -- files "
                                      "DELETED. Use the plain pair; ncnn "
                                      "Vulkan runs fp16 storage at runtime "
                                      "by default anyway.".format(bad2),
                                      file=sys.stderr)
                            elif bad2 == "":
                                print("fp16 ncnn files:",
                                      base + "_ncnn_fp16.param,",
                                      base + "_ncnn_fp16.bin")

    if True:   # ONNX-vs-PyTorch verification is MANDATORY (--fp32-check kept
               # for compatibility; it is now always on)
        import numpy as np
        import onnxruntime as ort
        s = ort.InferenceSession(simple, providers=["CPUExecutionProvider"])
        _vw, _vh = ((W, H) if fixed else (
            (256 if 256 % (div or 1) == 0 else (((256 // (div or 1)) + 1)
                                                * (div or 1))),) * 2)
        x = torch.rand(1, desc.input_channels, _vh, _vw)
        import onnxruntime as ort
        if is_onnx:
            _soZ = ort.SessionOptions()
            _soZ.log_severity_level = 4
            _sref = ort.InferenceSession(exported, _soZ,
                                         providers=["CPUExecutionProvider"])
            want = _sref.run(None, {args.input_name: x.numpy()})[0]
        else:
            with torch.no_grad():
                want = m(x).cpu().numpy()
        got = s.run(None, {args.input_name: x.numpy()})[0]
        print(f"fp32 check: out shape {got.shape}, "
              f"max|diff| vs pytorch = {abs(got - want).max():.3e}")


if __name__ == "__main__":
    main()
