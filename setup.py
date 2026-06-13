"""Build script for vecdbc — compiles the C core into a shared library at
install time using portable architecture flags.

Why a custom build step: pyvecdb loads libvecdb.so via ctypes from its own
directory, so we compile the C sources and drop the .so beside the installed
pyvecdb module. We deliberately avoid -march=native / -mcpu=native (which bake
in the build machine's exact CPU and crash on any other), choosing a portable
SIMD baseline per architecture instead. The C kernels are guarded by compiler
feature macros (__AVX512F__, __ARM_NEON, __ARM_FEATURE_DOTPROD), so they
activate automatically when the chosen flags enable them and fall back to
scalar code otherwise.
"""
import os
import platform
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
SRCS = ["vecdb.c", "turboquant.c", "hybrid.c"]


def arch_flags():
    machine = platform.machine().lower()
    system = platform.system()
    if machine in ("x86_64", "amd64"):
        # AVX2 is a safe baseline for x86-64 servers/desktops (~2013+).
        # Opt into AVX-512 kernels with VECDB_AVX512=1 when the target supports it.
        if os.environ.get("VECDB_AVX512") == "1":
            return ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512vnni", "-mfma"]
        return ["-mavx2", "-mfma"]
    if machine in ("arm64", "aarch64"):
        if system == "Darwin":
            return ["-mcpu=apple-m1"]            # portable across Apple Silicon
        return ["-march=armv8.2-a+dotprod"]      # Graviton2+, modern ARM servers
    return []                                    # unknown: scalar fallback only


class build_py_with_lib(build_py):
    """Compile libvecdb.so and place it inside the installed pyvecdb package."""

    def run(self):
        super().run()
        cc = os.environ.get("CC", "cc")
        base = ["-O3", "-fPIC", "-shared"] + arch_flags()
        srcs = [os.path.join(HERE, s) for s in SRCS]
        # pyvecdb is a top-level module; install the .so next to it.
        out_dir = self.build_lib
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "libvecdb.so")
        for omp in (["-fopenmp"], []):
            cmd = [cc] + base + omp + srcs + ["-o", out, "-lm"]
            try:
                subprocess.check_call(cmd)
                tag = "with OpenMP" if omp else "serial (no OpenMP)"
                print(f"vecdbc: built libvecdb.so [{tag}] flags={' '.join(base)}")
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                if omp:
                    continue
                raise RuntimeError(
                    f"failed to compile libvecdb.so with {cc}; "
                    "set CC to a working C compiler")


setup(cmdclass={"build_py": build_py_with_lib})
