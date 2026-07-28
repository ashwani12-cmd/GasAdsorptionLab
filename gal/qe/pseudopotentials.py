"""Pseudopotential filename defaults and user override support."""

from __future__ import annotations


DEFAULT_PSEUDOPOTENTIALS = {
    "Mo": "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "S": "S.pbe-n-kjpaw_psl.1.0.0.UPF",
}


def pseudopotential_for(symbol: str, overrides: dict[str, str] | None = None) -> str:
    """Return a PBE pseudopotential filename, applying user overrides first."""
    if overrides and symbol in overrides:
        return overrides[symbol]
    return DEFAULT_PSEUDOPOTENTIALS.get(symbol, f"{symbol}.pbe-n-kjpaw_psl.1.0.0.UPF")


def pseudopotential_map(symbols: list[str] | set[str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Return pseudopotential filenames for a collection of elements."""
    return {symbol: pseudopotential_for(symbol, overrides) for symbol in sorted(set(symbols))}
