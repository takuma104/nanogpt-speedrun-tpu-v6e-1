import time

import jax
import jax.numpy as jnp


def bench(fn, *args, iters: int = 20) -> float:
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters


def main() -> None:
    print("jax", jax.__version__)
    import jaxlib

    print("jaxlib", jaxlib.__version__)
    try:
        import libtpu  # type: ignore

        print("libtpu", getattr(libtpu, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        print("libtpu import failed:", e)
    devs = jax.devices()
    print("devices", devs)
    d = devs[0]
    print("device_kind", d.device_kind, "platform", d.platform)
    for k in ("num_cores", "core_on_chip", "coords"):
        print(k, getattr(d, k, None))
    stats = d.memory_stats()
    if stats:
        print("bytes_limit GB", stats.get("bytes_limit", 0) / 1e9)

    # bf16 matmul throughput
    for n in (4096, 8192):
        a = jnp.ones((n, n), jnp.bfloat16)
        b = jnp.ones((n, n), jnp.bfloat16)
        f = jax.jit(lambda x, y: x @ y)
        t = bench(f, a, b)
        print(f"bf16 matmul {n}: {2 * n**3 / t / 1e12:.1f} TFLOPS")
    # int8 matmul throughput
    n = 8192
    a8 = jnp.ones((n, n), jnp.int8)
    f8 = jax.jit(lambda x, y: jax.lax.dot(x, y, preferred_element_type=jnp.int32))
    t = bench(f8, a8, a8)
    print(f"int8 matmul {n}: {2 * n**3 / t / 1e12:.1f} TOPS")
    # fp8 (likely emulated on v6e)
    try:
        af = jnp.ones((n, n), jnp.float8_e4m3fn)
        ff = jax.jit(lambda x, y: jax.lax.dot(x, y, preferred_element_type=jnp.float32))
        t = bench(ff, af, af)
        print(f"fp8 matmul {n}: {2 * n**3 / t / 1e12:.1f} TFLOPS")
    except Exception as e:  # noqa: BLE001
        print("fp8 matmul failed:", type(e).__name__, str(e)[:200])
    # HBM bandwidth (copy-ish: read + write)
    x = jnp.ones((256 * 1024 * 1024,), jnp.float32)  # 1 GiB
    g = jax.jit(lambda v: v * 1.0001 + 1.0)
    t = bench(g, x)
    print(f"HBM elementwise r+w: {2 * x.nbytes / t / 1e9:.0f} GB/s")


if __name__ == "__main__":
    main()
