import jax.numpy as jnp
import jax

@jax.jit
def create_aircraft_states(axis, u, v, w, p, q, r):
    """Create aircraft states as a dictionary of JAX arrays.
    
    Args:
        u, v, w: Velocity components in body frame
        p, q, r: Angular velocity components in body frame  
        
    Returns:
        Dictionary containing all state variables and derived quantities
    """
    phi = axis.euler_angles.phi
    theta = axis.euler_angles.theta
    psi = axis.euler_angles.psi
    x = axis.translation_from_origin.x
    y = axis.translation_from_origin.y
    z = axis.translation_from_origin.z

    state_vector = build_state_vector(u, v, w, p, q, r, phi, theta, psi, x, y, z)
    angular_rates_vector = build_angular_rates_vector(p, q, r)
    position_vector = build_position_vector(x, y, z)
    euler_angles_vector = build_euler_angles_vector(phi, theta, psi)
    VTAS = calculate_VTAS(u, v, w)
    alpha = calculate_alpha(u, w)
    beta = calculate_beta(u, v, w)
    gamma = calculate_gamma(u, v)
    
    return {
        # Primary state variables
        'u': u, 'v': v, 'w': w,
        'p': p, 'q': q, 'r': r,
        'phi': phi, 'theta': theta, 'psi': psi,
        'x': x, 'y': y, 'z': z,
        
        # Derived state vectors
        'state_vector': state_vector,
        'angular_rates_vector': angular_rates_vector,
        'position_vector': position_vector,
        'euler_angles_vector': euler_angles_vector,
        
        # Flight dynamics quantities
        'VTAS': VTAS,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma
    }

@jax.jit
def build_state_vector(u, v, w, p, q, r, phi, theta, psi, x, y, z):
    """Build the full state vector."""
    return jnp.stack([u, v, w, p, q, r, phi, theta, psi, x, y, z])

@jax.jit
def build_angular_rates_vector(p, q, r):
    """Build angular rates vector."""
    return jnp.stack([p, q, r])

@jax.jit
def build_position_vector(x, y, z):
    """Build position vector."""
    return jnp.stack([x, y, z])

@jax.jit
def build_euler_angles_vector(phi, theta, psi):
    """Build Euler angles vector."""
    return jnp.stack([phi, theta, psi])

@jax.jit
def calculate_VTAS(u, v, w):
    """Calculate true airspeed."""
    return jnp.linalg.norm(jnp.stack([u, v, w]))

@jax.jit
def calculate_alpha(u, w):
    """Calculate angle of attack."""
    return jnp.arctan2(w, u)

@jax.jit
def calculate_beta(u, v, w):
    """Calculate sideslip angle."""
    return jnp.arctan2(v, jnp.sqrt(u**2 + w**2))

@jax.jit
def calculate_gamma(u, v):
    """Calculate flight path angle."""
    return jnp.arctan2(v, u)

    