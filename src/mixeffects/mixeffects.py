import numpy as np
import pandas as pd

class MixEffectImpact:
    """
    Computes contribution of weights and values to the differences between two weighted kpis (m1 and m2)

    Attributes:
        components: panda series with names for each component.
        w1: panda series with weights to compute kpi1 (m1) for each component
        x1: panda series with values to compute kpi1 (m1) for each component
        w2: panda series with weights to compute kpi2 (m2) for each component
        x2: panda series with values to compute kpi2 (m2) for each component
        m1: kpi1 np.dot(w1,x1)
        m2: kpi2 np.dot(w2,x2)
        deltam: difference between m2 and m1, also called gap
        deltam_rel: relative difference between m2 and m1
        w2x1: kpi1 computed keeping weights as in kpi2
        delta_values: deltam part that can be attributed to differences in values across components
        delta_weights: deltam part that can be attributed to differences in weights across components
        p_values: proportion of deltam attributed to difference in values
        p_weights: proportion of deltam attributed to difference in weights
    Methods:
        compute_attributes(): computes derived attributes from initialized attributes
        _check_length(): check length of all initial attributes have the same length
    """
    def __init__(self, components, w1, x1, w2, x2):
        """
        Initialize attributes required for the class
        Parameters:
            components: panda series with names for each component.
            w1: panda series with weights to compute kpi1 (m1) for each component
            x1: panda series with values to compute kpi1 (m1) for each component
            w2: panda series with weights to compute kpi2 (m2) for each component
            x2: panda series with values to compute kpi2 (m2) for each component
        """    
        self._check_lenght(components, w1, x1, w2, x2)
        self.components = components
        self.w1 = pd.Series(w1)
        self.x1 = pd.Series(x1)
        self.w2 = pd.Series(w2)
        self.x2 = pd.Series(x2)
        self.compute_attributes()
    
    def __str__(self):
        """
        Return a string representation of the mix effect.
        """
        return f"""
        Difference between kpi2 ({self.m2:.2f}) and kpi1 ({self.m1:.2f}), {self.deltam:.2f} ({self.deltam_rel * 100:.1f}%), can be split as:
        * {self.delta_values:.2f} due to actual difference between kpi2 and kpi1 across the {self.n} components, and
        * {self.delta_weights:.2f} due to the difference in the weights of each component for kpi2 and kpi1
        In relative terms, value difference account for {100*self.p_values:.1f}% of the gap and the weights account for {100*self.p_weights:.1f}%.
        """
    
    def compute_attributes(self):
        """
        Computes attributes related to mix effects
        """
        self.m1 = np.dot(self.w1, self.x1)
        self.m2 = np.dot(self.w2, self.x2)
        self.deltam = self.m2 - self.m1
        self.deltam_rel = self.m2 / self.m1 - 1
        self.w2x1 = np.dot(self.w2, self.x1)
        self.delta_values = self.m2 - self.w2x1
        self.delta_weights = self.w2x1 - self.m1
        self.p_values = self.delta_values / self.deltam
        self.p_weights = self.delta_weights / self.deltam
        self.n = self.components.size
        self.diff_by_component = self.x2.values - self.x1.values

    def _check_lenght(self, components, w1, x1, w2, x2):
        if not (components.size == w1.size == x1.size == w2.size == x2.size):
            raise ValueError("The lenght of all parameters should be the same")
