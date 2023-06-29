using PyCall

py"""
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import pickle

latent_dim = 1

gen_inner = lambda x: hk.Sequential(
    [
        hk.Linear(256),
        jax.nn.swish,
        hk.Linear(256),
        jax.nn.swish,
        hk.Linear(256),
        jax.nn.swish,
        hk.Linear(2),
    ]
)(x)

gen_inner_t = hk.without_apply_rng(hk.transform(gen_inner))

# latent_mapping_network = lambda z: hk.Linear(latent_dim, with_bias=False)(z)
latent_mapping_network = lambda z: hk.Sequential(
    [hk.Linear(128), jax.nn.swish, hk.Linear(2 * latent_dim)]
)(z)
latent_map_t = hk.without_apply_rng(hk.transform(latent_mapping_network))

with open("numpy_swiss_genps.pkl", "rb") as f:
    np_gen_ps = pickle.load(f)
gen_ps = jax.tree_util.tree_map(lambda x: jnp.array(x), np_gen_ps)

@jax.jit
def gen_fwd_jit(z):
    w = latent_map_t.apply(gen_ps["flow"], z)
    return gen_inner_t.apply(gen_ps["gen"], w)


def gen_fwd(z):
    z = jnp.array(z)
    x = gen_fwd_jit(z)
    x = np.array(x)
    return x
"""

z = randn((64,1))
x = py"gen_fwd"(z)