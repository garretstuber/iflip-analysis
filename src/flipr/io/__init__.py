"""IO loaders for FLIPR data (tidy CSV exports + session CSV exports + raw .iFLiP2)."""

from flipr.io.iflip2 import (
    IFlip2File,
    list_iflip2_files,
    load_iflip2,
    match_iflip2_to_session,
)
from flipr.io.session_csv import (
    AcquisitionEntry,
    SessionData,
    discover_acquisitions,
    list_sessions,
    load_session,
    session_from_tcspc_source,
)
from flipr.io.tidy_csv import (
    TidyData,
    list_tidy_files,
    load_tidy,
    match_tidy_to_session,
)

__all__ = [
    "AcquisitionEntry",
    "IFlip2File",
    "SessionData",
    "TidyData",
    "discover_acquisitions",
    "list_iflip2_files",
    "list_sessions",
    "list_tidy_files",
    "load_iflip2",
    "load_session",
    "load_tidy",
    "match_iflip2_to_session",
    "match_tidy_to_session",
    "session_from_tcspc_source",
]
