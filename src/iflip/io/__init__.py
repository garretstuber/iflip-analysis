"""IO loaders for FLIPR data (tidy CSV exports + session CSV exports + raw .iFLiP2)."""

from iflip.io.arduino_lick import (
    ArduinoLog,
    ClockAlignment,
    align_clocks,
    build_session_folder,
    load_arduino_log,
)
from iflip.io.iflip2 import (
    IFlip2File,
    list_iflip2_files,
    load_iflip2,
    match_iflip2_to_session,
)
from iflip.io.session_csv import (
    AcquisitionEntry,
    SessionData,
    discover_acquisitions,
    list_sessions,
    load_session,
    session_from_tcspc_source,
)
from iflip.io.tidy_csv import (
    TidyData,
    list_tidy_files,
    load_tidy,
    match_tidy_to_session,
)

__all__ = [
    "AcquisitionEntry",
    "ArduinoLog",
    "ClockAlignment",
    "IFlip2File",
    "SessionData",
    "TidyData",
    "align_clocks",
    "build_session_folder",
    "discover_acquisitions",
    "list_iflip2_files",
    "list_sessions",
    "list_tidy_files",
    "load_arduino_log",
    "load_iflip2",
    "load_session",
    "load_tidy",
    "match_iflip2_to_session",
    "match_tidy_to_session",
    "session_from_tcspc_source",
]
