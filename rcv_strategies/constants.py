"""
Constants for RCV Strategy Computation
======================================

This module defines global constants used throughout the RCV strategies package.
"""

# Tractability constraint for strategy computation
# Strategy computation is exponential in candidate count and becomes intractable
# for >= 9 candidates. For larger elections, use candidate removal (Theorem 4.1/4.3)
# to reduce the candidate set below this threshold.
MAX_TRACTABLE_CANDIDATES = 9
