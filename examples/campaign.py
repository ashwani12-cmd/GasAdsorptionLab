"""Generate a complete temporary MoS2 adsorption campaign."""

from __future__ import annotations

from tempfile import TemporaryDirectory

from ase import Atoms

from gal import AdsorptionCampaign


def mos2() -> Atoms:
    return Atoms(
        "MoS2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6], [0.0, 0.0, -1.6]],
        cell=[[3.18, 0.0, 0.0], [1.59, 2.753, 0.0], [0.0, 0.0, 20.0]],
        pbc=(True, True, False),
    )


if __name__ == "__main__":
    with TemporaryDirectory(prefix="gal-campaign-") as directory:
        campaign = AdsorptionCampaign(mos2(), ["CO", "NH3", "NO2"], output_dir=directory)
        jobs = campaign.generate()
        print(f"Number of calculations: {len(jobs)}")
        print(f"Campaign directory: {directory}")
