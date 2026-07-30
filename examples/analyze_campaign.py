"""Analyze a completed Quantum ESPRESSO campaign directory."""

from gal import CampaignResults


if __name__ == "__main__":
    results = CampaignResults.from_directory("campaigns/WSe2_CO")
    print(results.dataframe)
    # Supply reference energies from separately converged calculations before
    # ranking or plotting adsorption energies.
    if not results.dataframe.empty:
        results.to_csv()
        results.to_json()
