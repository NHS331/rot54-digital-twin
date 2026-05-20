from __future__ import annotations

import math


class ClearSkyIrradianceModel:
    """
    Simple engineering clear-sky direct normal irradiance model.

    Output:
        DNI_clear in W/m^2

    This is not a measured meteorological model.
    It is a deterministic clear-sky proxy for building the solar-load pipeline.
    """

    def __init__(
        self,
        solar_constant_w_m2: float = 1361.0,
        clear_sky_transmittance: float = 0.72,
    ) -> None:
        self.solar_constant_w_m2 = solar_constant_w_m2
        self.clear_sky_transmittance = clear_sky_transmittance

    def extraterrestrial_normal_irradiance(
        self,
        day_of_year: int,
    ) -> float:
        """
        Approximate extraterrestrial normal irradiance.

        Unit:
            W/m^2
        """

        return self.solar_constant_w_m2 * (
            1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.0)
        )

    @staticmethod
    def relative_air_mass(
        altitude_deg: float,
    ) -> float:
        """
        Kasten-Young type relative optical air mass.

        Input:
            solar altitude in degrees

        Output:
            relative air mass
        """

        if altitude_deg <= 0.0:
            return math.inf

        zenith_deg = 90.0 - altitude_deg

        if zenith_deg >= 90.0:
            return math.inf

        zenith_rad = math.radians(zenith_deg)

        denominator = (
            math.cos(zenith_rad)
            + 0.50572 * ((96.07995 - zenith_deg) ** -1.6364)
        )

        if denominator <= 0.0:
            return math.inf

        return 1.0 / denominator

    def direct_normal_irradiance_clear_sky(
        self,
        altitude_deg: float,
        day_of_year: int,
    ) -> float:
        """
        Calculate clear-sky direct normal irradiance.

        Unit:
            W/m^2
        """

        if altitude_deg <= 0.0:
            return 0.0

        i_0 = self.extraterrestrial_normal_irradiance(
            day_of_year=day_of_year,
        )

        air_mass = self.relative_air_mass(
            altitude_deg=altitude_deg,
        )

        if not math.isfinite(air_mass):
            return 0.0

        attenuation_exponent = air_mass ** 0.678

        dni_clear = i_0 * (
            self.clear_sky_transmittance ** attenuation_exponent
        )

        return max(0.0, dni_clear)

    def projected_clear_sky_load(
        self,
        altitude_deg: float,
        day_of_year: int,
        mu_front: float,
    ) -> float:
        """
        Clear-sky projected load on the reflector front side.

        Formula:
            Q_clear_projected = DNI_clear * mu_front

        Unit:
            W/m^2
        """

        dni_clear = self.direct_normal_irradiance_clear_sky(
            altitude_deg=altitude_deg,
            day_of_year=day_of_year,
        )

        return dni_clear * max(0.0, mu_front)
