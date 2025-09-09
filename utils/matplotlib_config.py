import os
import sys
import platform
import logging


def configure_matplotlib_backend(force_backend=None, verbose=False):
    """
    Configure optimal matplotlib backend based on OS and environment.
    
    Args:
        force_backend (str, optional): Force a specific backend
        verbose (bool): Display configuration information
        
    Returns:
        str: Configured backend or None if matplotlib unavailable
    """
    logger = logging.getLogger(__name__) if verbose else None
    
    try:
        import matplotlib
    except ImportError:
        if logger:
            logger.warning("matplotlib not available")
        return None

    if force_backend:
        try:
            matplotlib.use(force_backend, force=True)
            if logger:
                logger.info(f"Forced backend: {force_backend}")
            return force_backend
        except Exception as e:
            if logger:
                logger.warning(f"Cannot force backend {force_backend}: {e}")
    
    is_headless = not bool(os.environ.get('DISPLAY', '')) and sys.platform.startswith('linux')
    is_ssh = bool(os.environ.get('SSH_CONNECTION', ''))
    is_ci = bool(os.environ.get('CI', ''))
    
    if is_headless or is_ssh or is_ci:
        backend = 'Agg'
    else:
        system = platform.system().lower()
        
        if system == 'darwin':
            backend = _configure_macos_backend(logger)
        elif system == 'windows':
            backend = _configure_windows_backend(logger)
        elif system == 'linux':
            backend = _configure_linux_backend(logger)
        else:
            backend = 'Agg'
            if logger:
                logger.info(f"Unknown OS ({system}), using Agg")
    
    try:
        matplotlib.use(backend, force=True)
        
        if backend == 'Agg':
            matplotlib.pyplot.ioff()
        else:
            try:
                matplotlib.pyplot.ion()
            except Exception:
                matplotlib.pyplot.ioff()
                
        if logger:
            logger.info(f"Matplotlib backend configured: {backend}")
            
        return backend
        
    except Exception as e:
        try:
            matplotlib.use('Agg', force=True)
            if logger:
                logger.warning(f"Error with {backend}, fallback to Agg: {e}")
            return 'Agg'
        except Exception:
            if logger:
                logger.error("Cannot configure matplotlib")
            return None


def _configure_macos_backend(logger=None):
    """Configure optimal backend for macOS."""
    backends_to_try = ['MacOSX', 'Qt5Agg', 'TkAgg', 'Agg']
    
    for backend in backends_to_try:
        if _test_backend(backend, logger):
            return backend
    
    if logger:
        logger.warning("No GUI backend available on macOS, using Agg")
    return 'Agg'


def _configure_windows_backend(logger=None):
    """Configure optimal backend for Windows."""
    backends_to_try = ['TkAgg', 'Qt5Agg', 'Agg']
    
    for backend in backends_to_try:
        if _test_backend(backend, logger):
            return backend
    
    if logger:
        logger.warning("No GUI backend available on Windows, using Agg")
    return 'Agg'


def _configure_linux_backend(logger=None):
    """Configure optimal backend for Linux."""
    backends_to_try = ['Qt5Agg', 'TkAgg', 'GTK3Agg', 'Agg']
    
    for backend in backends_to_try:
        if _test_backend(backend, logger):
            return backend
    
    if logger:
        logger.warning("No GUI backend available on Linux, using Agg")
    return 'Agg'


def _test_backend(backend, logger=None):
    """Test if a matplotlib backend works."""
    try:
        import matplotlib
        matplotlib.use(backend, force=True)
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.close(fig)
        
        return True
        
    except Exception as e:
        if logger:
            logger.debug(f"Backend {backend} not available: {e}")
        return False


def get_backend_info():
    """
    Return detailed information about current matplotlib backend.
    
    Returns:
        dict: Information about matplotlib configuration
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        
        info = {
            'backend': matplotlib.get_backend(),
            'interactive': matplotlib.is_interactive(),
            'rcParams_backend': matplotlib.rcParams.get('backend', 'unknown'),
            'gui_available': _has_gui_available(),
            'environment': {
                'os': platform.system(),
                'platform': platform.platform(),
                'display': bool(os.environ.get('DISPLAY', '')),
                'ssh': bool(os.environ.get('SSH_CONNECTION', '')),
                'jupyter': any(name in sys.modules for name in ['ipykernel', 'jupyter_client']),
                'colab': 'google.colab' in sys.modules,
                'ci': bool(os.environ.get('CI', ''))
            }
        }
        return info
        
    except ImportError:
        return {
            'backend': None,
            'error': 'matplotlib not available',
            'environment': {
                'os': platform.system(),
                'platform': platform.platform()
            }
        }


def _has_gui_available():
    """Check if GUI is available."""
    system = platform.system().lower()
    
    if system == 'darwin':
        return True
    elif system == 'windows':
        return True
    elif system == 'linux':
        return bool(os.environ.get('DISPLAY', ''))
    else:
        return False


def setup_matplotlib_for_headless():
    """Special configuration for headless environments (servers, CI, etc.)."""
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        plt.ioff()
        
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        
        return True
        
    except ImportError:
        return False


def ensure_matplotlib_works():
    """
    Ensure matplotlib works, configure automatically if necessary.
    
    Returns:
        bool: True if matplotlib is operational
    """
    try:
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(1, 1))
        plt.plot([1, 2], [1, 2])
        plt.close(fig)
        
        return True
        
    except Exception:
        backend = configure_matplotlib_backend(verbose=False)
        
        if backend:
            try:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(1, 1))
                plt.close(fig)
                return True
            except Exception:
                return False
        
        return False


_configured_backend = configure_matplotlib_backend(verbose=False)

__all__ = [
    'configure_matplotlib_backend',
    'get_backend_info',
    'setup_matplotlib_for_headless',
    'ensure_matplotlib_works'
]
