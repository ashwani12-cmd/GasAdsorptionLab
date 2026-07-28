"""Find adsorption sites on a primitive graphene sheet."""

from ase.build import graphene

from gal import SiteFinder


if __name__ == "__main__":
    sites = SiteFinder(graphene(vacuum=8.0)).find_all()
    print(f"Graphene sites: {len(sites)}")
    print([site.name for site in sites])
