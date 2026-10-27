# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Activate the Moore Threads MUSA compatibility layer before CUDA capture.

``torchada`` redirects ``torch.cuda.*``, ``"cuda"`` device strings, ``nccl``
process groups and ``torch.cuda.CUDAGraph`` to their MUSA equivalents. Patching
happens when the adapter is imported, so it must run before any module captures
one of those symbols -- which is why both package roots call it first instead
of a platform branch deeper in the tree.

The check looks for the installed ``torch_musa`` distribution rather than
importing torch, so a CUDA or ROCm host imports nothing extra and does not need
``torchada`` installed at all.
"""

import importlib.util

__all__ = ["activate_musa_shims"]


def activate_musa_shims() -> None:
    """Import ``torchada`` when this is a torch_musa installation."""
    if importlib.util.find_spec("torch_musa") is None:
        return
    import torchada  # noqa: F401
