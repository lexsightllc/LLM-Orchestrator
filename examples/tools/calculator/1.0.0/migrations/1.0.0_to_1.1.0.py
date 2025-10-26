# SPDX-License-Identifier: MPL-2.0
"""
Migration script for calculator tool from 1.0.0 to 1.1.0.

Changes in 1.1.0:
- Added 'power' operation
- Added 'round' parameter to control decimal places
- Improved error handling for invalid operations
"""


def migrate(params):
    """
    Migrate parameters from v1.0.0 to v1.1.0.

    Args:
        params: Input parameters from v1.0.0

    Returns:
        dict: Parameters for v1.1.0
    """
    # Copy all existing parameters
    new_params = params.copy()
    
    # Add default for new 'round' parameter
    new_params['round'] = 2  # Default to 2 decimal places
    
    return new_params

def rollback(params):
    """
    Rollback parameters from v1.1.0 to v1.0.0.
    
    Args:
        params: Input parameters from v1.1.0
        
    Returns:
        dict: Parameters for v1.0.0
    """
    # Remove the 'round' parameter that was added in 1.1.0
    return {k: v for k, v in params.items() if k != 'round'}
