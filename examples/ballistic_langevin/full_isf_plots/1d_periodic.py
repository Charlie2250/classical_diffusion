from dataclasses import dataclass

import jax.random as jrandom
import numpy as np
from scipy.constants import electron_volt

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    breakdown_filtered_ballistic_trajectory_butterworth,
    get_initial_conditions,
    solve_ballistic_ensemble,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import (
    PeriodicSystem1D,
    UnitSystem,
    get_diffusion_time,
)


@dataclass(frozen=True, kw_only=True)
class CamColor:
    """A class to hold CAM color palettes."""

    light: str
    warm: str
    base: str
    dark: str


CAM_BLUE = CamColor(
    light="#D1F9F1",
    warm="#00BDB6",
    base="#8EE8D8",
    dark="#133844",
)
CAM_CHERRY = CamColor(
    light="#F2CAD8",
    warm="#E18AAC",
    base="#CD3572",
    dark="#911449",
)
system = PeriodicSystem1D(
    gamma=9e11,
    temperature=115,
    barrier_energy=55e-3 * electron_volt,
    delta_x=(1 / np.sqrt(3)) * 2.558e-10,
    m=3.8175458e-26,
    units=UnitSystem(),
)
normalized_system = system.with_normalized_units()

key = jrandom.PRNGKey(100)


def _plot_periodic_isf() -> None:

    full_result = solve_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(10e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        (np.full((700, 1), 0.0), np.full((700, 1), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()

    delta_k = (2 * np.pi / system.delta_x * 0.4,)
    _, ax, line_0, fill_0 = plot_isf(
        result=full_result.with_si_units(),
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("full simulation")
    line_0.set_color(CAM_BLUE.dark)
    fill_0.set_color(CAM_BLUE.dark)

    initial_conditions = get_initial_conditions(normalized_system, n_samples=10_000)

    ballistic_result = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(5e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions=initial_conditions,
        _key=key,
    )

    _, ax, line_1, fill_1 = plot_isf(
        result=ballistic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_1.set_label("ballistic simulation")
    line_1.set_color(CAM_BLUE.warm)
    fill_1.set_color(CAM_BLUE.warm)

    ax.set_xlim(
        0,
        5e-12,
    )

    ax.set_ylim(
        0.85,
        1.0,
    )

    ax.legend(handles=[line_0, line_1])
    fig.savefig(
        "examples/ballistic_langevin/full_isf_plots/1d_periodic.isf.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    fig, ax = get_fancy_figure()

    _, ax, line_1, fill_1 = plot_isf(
        result=ballistic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_1.set_label("ballistic simulation")
    line_1.set_color(CAM_BLUE.warm)
    fill_1.set_color(CAM_BLUE.warm)

    elastic_result, inelastic_result = (
        breakdown_filtered_ballistic_trajectory_butterworth(
            ballistic_result,
            minimum_timescale=get_diffusion_time(
                normalized_system, characteristic_length=normalized_system.delta_x
            ),
        )
    )

    _, ax, line_2, fill_2 = plot_isf(
        result=elastic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_2.set_label("elastic")
    line_2.set_color(CAM_CHERRY.warm)
    fill_2.set_color(CAM_CHERRY.warm)

    _, ax, line_3, fill_3 = plot_isf(
        result=inelastic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_3.set_label("inelastic")
    line_3.set_color(CAM_CHERRY.dark)
    fill_3.set_color(CAM_CHERRY.dark)

    ax.set_xlim(
        0,
        4.5e-12,
    )

    ax.set_ylim(
        0.85,
        1.0,
    )

    ax.legend(handles=[line_1, line_2, line_3])
    fig.savefig(
        "examples/ballistic_langevin/full_isf_plots/1d_periodic.isf.e.vs.i.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_periodic_isf()
