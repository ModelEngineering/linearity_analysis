'''Builds a CRN from a system of Quadratic ODEs.'''

from collections import namedtuple
import numpy as np  # type: ignore
import pandas as pd # type: ignore
from typing import List, Optional, Tuple, Dict


Reaction = namedtuple("Reaction", ["reactants", "products", "rate_constant",
        "monomial"])




class CRNBuilder:
    '''Builds a CRN from a system of Quadratic ODEs.'''
    def __init__(self, system_df: pd.DataFrame) -> None:
        """
        Args:
            system_df (pd.DataFrame): Output from NetworkDiscovery.summary()
        """
        self.system_df = system_df
        self.species_names = [n[1:-3] for n in system_df.columns]
        self.system_df.columns = self.species_names
        self.monomials = self.system_df.index.tolist()

    def build(self) -> List[Reaction]:
        '''Builds a CRN from the system_df.'''
        reactions: List[Reaction] = []
        for monomial in self.monomials:
            stoichiometry_dct : Dict[str, float] = {}
            # Find the smallest nonzero coefficient across species for this monomial, to use as the rate constant.
            values = self.system_df.loc[monomial, [sp for sp in self.species_names]].abs().values
            min_coeff = np.min(values[values != 0])
            # Check that all other coefficients are an integer multiple
            for sp in self.species_names:
                coeff = self.system_df.loc[monomial, f"d{sp}/dt"]
                stoichiometry_dct[sp] = coeff / min_coeff
                if coeff != 0 and not np.isclose(stoichiometry_dct[sp], round(stoichiometry_dct[sp])):
                    raise ValueError(f"Coefficient {coeff} for species {sp} is not an integer multiple of the minimum coefficient {min_coeff} for monomial {monomial}.")
            # Construction the reaction. Negative coefficients indicate reactants;
            #   positive coefficients indicate products.
            reactants = [sp for sp in self.species_names if self.system_df.loc[monomial, sp ] < 0]
            products = [sp for sp in self.species_names if float(self.system_df.loc[monomial, sp]) > 0]
            # Calculate stoichiometric coefficients as integer multiples of the minimum coefficient.
            reactions.append(Reaction(reactants, products, min_coeff, monomial))
        return reactions