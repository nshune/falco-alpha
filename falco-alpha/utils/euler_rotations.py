import jax.numpy as jnp
import jax
import numpy as np

@jax.jit
def build_rotation_matrix(angles: jnp.ndarray, seq: str = 'zyx'):
    """
    Builds a rotation matrix from Euler angles.

    The rotation matrix, R, is constructed from the following sequence of rotations by default:
    1. Rotation about the z-axis by angle phi
    2. Rotation about the y-axis by angle theta
    3. Rotation about the x-axis by angle psi

    Args:
        angles: The Euler angles in radians, shape (3,).
        seq: The sequence of rotations as a string, e.g., 'zyx'.

    Returns:
        The rotation matrix.
    """

    # Ensure angles is a JAX array of shape (3,)
    angles = jnp.asarray(angles)
    angles = jnp.reshape(angles, (3,))

    phi, theta, psi = angles
    c1, s1 = jnp.cos(phi), jnp.sin(phi)
    c2, s2 = jnp.cos(theta), jnp.sin(theta)
    c3, s3 = jnp.cos(psi), jnp.sin(psi)

    def rot_zyx():
        return jnp.array([
            [c2 * c3, -c2 * s3, s2],
            [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
            [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]
        ])

    R = jax.lax.cond(
        seq == 'zyx',
        rot_zyx,
        lambda: jnp.eye(3)
    )

    return R