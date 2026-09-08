"""Diff rustft against PyTorch, in float64, case by case.

Everything here is float64 on purpose: torch's window helpers default to
float32, and that alone injects ~1e-7 of error that has nothing to do with
rustft.
"""
import os, shutil, subprocess, sys, tempfile
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "target", "release", "conformance")

WINDOWS = {
    "hann": torch.hann_window,
    "hamming": torch.hamming_window,
    "blackman": torch.blackman_window,
    "bartlett": torch.bartlett_window,
    "rectangular": lambda n, periodic, dtype: torch.ones(n, dtype=dtype),
}


def make_window(name, n_fft, periodic):
    if name == "rectangular":
        return torch.ones(n_fft, dtype=torch.float64)
    return WINDOWS[name](n_fft, periodic=periodic, dtype=torch.float64)


def run_case(channels, length, n_fft, hop, win="hann", periodic=True, seed=0):
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal((channels, length))
    window = make_window(win, n_fft, periodic)

    tmp = tempfile.mkdtemp(prefix="rustft-conf-")
    try:
        sig.astype("<f8").tofile(os.path.join(tmp, "in.bin"))

        try:
            t_stft = torch.stft(torch.from_numpy(sig), n_fft, hop, window=window,
                                center=True, return_complex=True)
        except RuntimeError as e:
            # PyTorch refuses this shape. rustft should refuse it too, rather than
            # panicking or inventing an answer.
            np.zeros(0, "<f8").tofile(os.path.join(tmp, "torch_stft.bin"))
            r = subprocess.run([BIN, str(n_fft), str(hop), win, str(periodic).lower(),
                                str(channels), str(length), tmp],
                               capture_output=True, text=True)
            return {"both_reject": r.returncode != 0,
                    "torch_error": str(e).splitlines()[0][:60]}
        ts = t_stft.numpy()
        np.stack([ts.real, ts.imag], -1).astype("<f8").tofile(
            os.path.join(tmp, "torch_stft.bin"))

        r = subprocess.run([BIN, str(n_fft), str(hop), win, str(periodic).lower(),
                            str(channels), str(length), tmp],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {"error": (r.stderr.strip().splitlines() or ["?"])[-1]}

        c, f, t = map(int, open(os.path.join(tmp, "rust_stft.shape")).read().split())
        raw = np.fromfile(os.path.join(tmp, "rust_stft.bin"), "<f8").reshape(c, f, t, 2)
        rs = raw[..., 0] + 1j * raw[..., 1]

        shape = open(os.path.join(tmp, "rust_istft.shape")).read()
        rust_rejected = shape.startswith("error")
        try:
            t_istft = torch.istft(t_stft, n_fft, hop, window=window, center=True).numpy()
            torch_rejected = False
        except RuntimeError:
            torch_rejected = True
        if rust_rejected or torch_rejected:
            return {"both_reject": rust_rejected and torch_rejected}

        rc, rl = map(int, shape.split())
        ri = np.fromfile(os.path.join(tmp, "rust_istft.bin"), "<f8").reshape(rc, rl)

        out = {}
        if rs.shape != ts.shape:
            out["stft_shape"] = f"rust {rs.shape} vs torch {ts.shape}"
        else:
            out["stft_max"] = float(np.abs(rs - ts).max())
            out["stft_rel"] = float(np.abs(rs - ts).max() / max(np.abs(ts).max(), 1e-30))
        if ri.shape != t_istft.shape:
            out["istft_shape"] = f"rust {ri.shape} vs torch {t_istft.shape}"
        else:
            out["istft_max"] = float(np.abs(ri - t_istft).max())
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


CASES = [
    # (label, channels, length, n_fft, hop, window, periodic)
    ("baseline hann n_fft=1024 hop=512",   2, 16384, 1024, 512, "hann", True),
    ("hop = n_fft/4",                      2, 16384, 1024, 256, "hann", True),
    ("hop = n_fft/3 (not a divisor)",      2, 16384,  768, 256, "hann", True),
    ("hop = 5*n_fft/8 (awkward)",          2, 16384, 1024, 640, "hann", True),
    ("hop = n_fft (no overlap)",           1,  8192,  512, 512, "rectangular", True),
    ("hop = n_fft, hann (fails NOLA)",     1,  8192,  512, 512, "hann", True),
    ("symmetric hann",                     1,  8192,  512, 128, "hann", False),
    ("hamming periodic",                   1,  8192,  512, 128, "hamming", True),
    ("blackman periodic",                  1,  8192,  512, 128, "blackman", True),
    ("bartlett periodic",                  1,  8192,  512, 128, "bartlett", True),
    ("odd n_fft = 501",                    1,  8192,  501, 125, "hann", True),
    ("odd n_fft = 63",                     1,  4096,   63,  16, "hann", True),
    ("length not a multiple of hop",       2, 16000, 1024, 512, "hann", True),
    ("non-power-of-two n_fft = 6144",      2, 65536, 6144, 1024, "hann", True),
    ("short signal, length < n_fft",       1,   300, 1024, 256, "hann", True),
    ("tiny signal length == n_fft",        1,   512,  512, 128, "hann", True),
    ("single channel long",                1, 261120, 6144, 1024, "hann", True),
]

if __name__ == "__main__":
    if not os.path.exists(BIN):
        sys.exit(f"build first: cargo build -p conformance --release ({BIN} missing)")
    width = max(len(c[0]) for c in CASES)
    worst = 0.0
    for label, ch, ln, n, h, w, p in CASES:
        try:
            res = run_case(ch, ln, n, h, w, p)
        except Exception as e:
            print(f"{label:<{width}}  EXC  {type(e).__name__}: {str(e).splitlines()[0][:90]}")
            continue
        if "both_reject" in res:
            flag = "ok " if res["both_reject"] else "BAD"
            note = "both reject" if res["both_reject"] else "only one of the two rejects it"
            print(f"{label:<{width}}  {flag}  {note}")
            continue
        if "error" in res:
            print(f"{label:<{width}}  RUST-ERR  {res['error'][:90]}")
            continue
        bits = []
        for k in ("stft_shape", "istft_shape"):
            if k in res:
                bits.append(f"{k} {res[k]}")
        for k in ("stft_max", "istft_max"):
            if k in res:
                bits.append(f"{k}={res[k]:.3e}")
                worst = max(worst, res[k])
        flag = "ok " if all(res.get(k, 1) < 1e-9 for k in ("stft_max", "istft_max")) \
                        and "stft_shape" not in res and "istft_shape" not in res else "BAD"
        print(f"{label:<{width}}  {flag}  {'  '.join(bits)}")
    print(f"\nworst absolute difference across all passing cases: {worst:.3e}")
