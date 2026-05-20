import pandas as pd

from src.solar_graph import (
    SolarGraphConfig,
    build_sun_altaz_table,
    generate_solar_graph_outputs,
)


def test_solar_graph_table_has_expected_columns():
    config = SolarGraphConfig(
        start_iso="2026-06-21T00:00:00+04:00",
        end_iso="2026-06-21T01:00:00+04:00",
        step_minutes=30,
    )

    df = build_sun_altaz_table(config)

    expected_columns = {
        "time_armenia",
        "project_timezone",
        "sun_altitude_deg",
        "sun_azimuth_deg",
        "sun_enu_x_east",
        "sun_enu_y_north",
        "sun_enu_z_up",
        "sun_above_horizon",
    }

    assert expected_columns.issubset(set(df.columns))


def test_solar_graph_table_has_correct_number_of_rows():
    config = SolarGraphConfig(
        start_iso="2026-06-21T00:00:00+04:00",
        end_iso="2026-06-21T01:00:00+04:00",
        step_minutes=15,
    )

    df = build_sun_altaz_table(config)

    assert len(df) == 4


def test_solar_graph_values_are_in_valid_ranges():
    config = SolarGraphConfig(
        start_iso="2026-06-21T00:00:00+04:00",
        end_iso="2026-06-21T01:00:00+04:00",
        step_minutes=30,
    )

    df = build_sun_altaz_table(config)

    assert df["sun_altitude_deg"].between(-90.0, 90.0).all()
    assert df["sun_azimuth_deg"].between(0.0, 360.0, inclusive="left").all()


def test_solar_graph_outputs_are_created(tmp_path):
    table_path = tmp_path / "sun_altaz_day.csv"
    altitude_path = tmp_path / "sun_altitude_day.png"
    azimuth_path = tmp_path / "sun_azimuth_day.png"

    config = SolarGraphConfig(
        start_iso="2026-06-21T00:00:00+04:00",
        end_iso="2026-06-21T01:00:00+04:00",
        step_minutes=30,
        output_table_path=str(table_path),
        output_altitude_figure_path=str(altitude_path),
        output_azimuth_figure_path=str(azimuth_path),
    )

    generate_solar_graph_outputs(config)

    assert table_path.exists()
    assert altitude_path.exists()
    assert azimuth_path.exists()

    assert table_path.stat().st_size > 0
    assert altitude_path.stat().st_size > 0
    assert azimuth_path.stat().st_size > 0


def test_saved_csv_can_be_read(tmp_path):
    table_path = tmp_path / "sun_altaz_day.csv"
    altitude_path = tmp_path / "sun_altitude_day.png"
    azimuth_path = tmp_path / "sun_azimuth_day.png"

    config = SolarGraphConfig(
        start_iso="2026-06-21T00:00:00+04:00",
        end_iso="2026-06-21T01:00:00+04:00",
        step_minutes=30,
        output_table_path=str(table_path),
        output_altitude_figure_path=str(altitude_path),
        output_azimuth_figure_path=str(azimuth_path),
    )

    generate_solar_graph_outputs(config)

    df = pd.read_csv(table_path)

    assert len(df) == 2
    assert "time_armenia" in df.columns
    assert "sun_altitude_deg" in df.columns
    assert "sun_azimuth_deg" in df.columns