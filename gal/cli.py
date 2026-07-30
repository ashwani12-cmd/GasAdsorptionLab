"""Command-line interface for GasAdsorptionLab."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .campaign import AdsorptionCampaign
from .qe import QEInputBuilder
from .results import CampaignResults


def campaign_from_config(filename: str | Path) -> AdsorptionCampaign:
    """Create a campaign from a YAML file with a ``campaign`` section."""
    data = yaml.safe_load(Path(filename).read_text()) or {}
    campaign = data.get("campaign", data)
    qe = campaign.get("qe", {})
    builder = QEInputBuilder(
        pseudo_dir=qe.get("pseudo_dir", "./pseudo"),
        ecutwfc=qe.get("ecutwfc", 60),
        ecutrho=qe.get("ecutrho", 480),
        kpts=tuple(qe.get("kpts", [6, 6, 1])),
        xc=qe.get("xc", "PBE"),
        pseudopotentials=qe.get("pseudopotentials"),
    )
    return AdsorptionCampaign(
        surface=campaign.get("surfaces", campaign.get("surface")),
        adsorbates=campaign["adsorbates"],
        output_dir=campaign.get("output_dir", "campaign"),
        heights=campaign.get("heights", "auto"),
        orientations=campaign.get("orientations", "auto"),
        qe_builder=builder,
        slurm=campaign.get("slurm"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gal")
    subparsers = parser.add_subparsers(dest="command", required=True)
    campaign_parser = subparsers.add_parser("campaign", help="generate a QE adsorption campaign")
    campaign_parser.add_argument("config")
    campaign_parser.add_argument("--overwrite", action="store_true")
    analyze_parser = subparsers.add_parser("analyze", help="analyze completed QE campaign results")
    analyze_parser.add_argument("directory")
    analyze_parser.add_argument("--clean-surface-energy", type=float)
    analyze_parser.add_argument("--gas-phase-energy", type=float)
    args = parser.parse_args(argv)
    if args.command == "campaign":
        campaign_from_config(args.config).generate(overwrite=args.overwrite)
    if args.command == "analyze":
        results = CampaignResults.from_directory(args.directory)
        if args.clean_surface_energy is not None or args.gas_phase_energy is not None:
            if args.clean_surface_energy is None or args.gas_phase_energy is None:
                parser.error("--clean-surface-energy and --gas-phase-energy must be supplied together")
            results.compute_adsorption_energy(args.clean_surface_energy, args.gas_phase_energy)
        results.to_csv()
        results.to_excel()
        results.to_json()
        results.export_optimized_structures()
        if "AdsorptionEnergy" in results.dataframe:
            results.plot_adsorption_energy()
        print(f"Analyzed {len(results.dataframe)} calculations")
    return 0
