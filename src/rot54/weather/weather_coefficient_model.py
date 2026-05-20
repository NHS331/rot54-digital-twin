from __future__ import annotations

from datetime import date


class WeatherCoefficientModel:
    """
    Smooth monthly-anchored average weather coefficient model.

    K_weather is a dimensionless reduction factor:

        Q_eff = Q_clear_projected * K_weather

    Important:
        This is an engineering climatological placeholder.
        It must later be replaced or calibrated with real meteorological data
        if measured cloudiness / humidity / wind records are available.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _anchor_points_for_year(
        year: int,
    ) -> list[tuple[date, float]]:
        """
        Monthly anchor values.

        Higher values correspond to clearer / more solar-effective periods.
        The interpolation avoids artificial month-step jumps.
        """

        return [
            (date(year - 1, 12, 15), 0.541),
            (date(year, 1, 15), 0.548),
            (date(year, 2, 15), 0.574),
            (date(year, 3, 15), 0.628),
            (date(year, 4, 15), 0.681),
            (date(year, 5, 15), 0.738),
            (date(year, 6, 15), 0.789),
            (date(year, 7, 15), 0.823),
            (date(year, 8, 15), 0.806),
            (date(year, 9, 15), 0.746),
            (date(year, 10, 15), 0.674),
            (date(year, 11, 15), 0.596),
            (date(year, 12, 15), 0.549),
            (date(year + 1, 1, 15), 0.548),
        ]

    def coefficient_for_date(
        self,
        day: date,
    ) -> float:
        """
        Return interpolated K_weather for a given date.
        """

        anchors = self._anchor_points_for_year(day.year)

        day_ordinal = day.toordinal()

        for index in range(len(anchors) - 1):
            left_date, left_value = anchors[index]
            right_date, right_value = anchors[index + 1]

            left_ordinal = left_date.toordinal()
            right_ordinal = right_date.toordinal()

            if left_ordinal <= day_ordinal <= right_ordinal:
                fraction = (
                    (day_ordinal - left_ordinal)
                    / (right_ordinal - left_ordinal)
                )

                value = left_value + fraction * (right_value - left_value)

                return max(0.0, min(1.0, value))

        if day_ordinal < anchors[0][0].toordinal():
            return anchors[0][1]

        return anchors[-1][1]
