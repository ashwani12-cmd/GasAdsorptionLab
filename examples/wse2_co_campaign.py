"""Generate a complete QE-relax WSe2 + CO adsorption campaign."""

from gal import generate_wse2_co_campaign


if __name__ == "__main__":
    directories = generate_wse2_co_campaign(supercell=(3, 3, 1))
    print(f"Generated {len(directories)} WSe2 + CO calculations")
    print("Campaign directory: campaigns/WSe2_CO")
