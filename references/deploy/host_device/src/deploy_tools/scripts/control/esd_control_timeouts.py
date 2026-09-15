#!/usr/bin/env python3
"""Shared timing contract for ESD-Link control-mode service clients."""

# The bridge can spend up to 5.0 s waiting for SET_ENABLE acknowledgement and
# then 6.5 s waiting for SAFE_DAMPING to reach DISABLED. Client calls must cover
# both server-side windows plus scheduling margin.
CONTROL_MODE_SERVICE_TIMEOUT_SECONDS = 13.0

