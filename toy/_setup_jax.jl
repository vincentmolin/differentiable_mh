import Pkg
Pkg.add("PyCall")
Pkg.add("Conda")
Conda.pip_interop(true)
Conda.pip("install", ["jax[cpu]", "dm-haiku"])

using PyCall

py"""
import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
def jitted_sinsum(x):
    return jnp.sum(jnp.sin(x))

def sinsum(x):
    x = jnp.array(x)
    return jitted_sinsum(x)

"""

x = ones(Float32, 16)

py"jitted_sinsum"(x)