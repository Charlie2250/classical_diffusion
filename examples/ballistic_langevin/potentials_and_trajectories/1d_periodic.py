import jax.random as jrandom
import numpy as np
from scipy.constants import electron_volt

from classical_diffusion.analysis import (
    plot_single_x_evolution,
)
from classical_diffusion.langevin import (
    animate_elastic_inelastic_breakdown_1d_periodic,
    breakdown_filtered_ballistic_trajectory_butterworth,
    plot_periodic_potential_1d,
    solve_single,
)
from classical_diffusion.plot import _get_two_panel_figure, get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import (
    PeriodicSystem1D,
    UnitSystem,
)

system = PeriodicSystem1D(
    gamma=4e11,
    temperature=115,
    barrier_energy=55e-3 * electron_volt,
    delta_x=(1 / np.sqrt(3)) * 2.558e-10,
    m=3.8175458e-26,
    units=UnitSystem(),
)


normalized_system = system.with_normalized_units()

key = jrandom.PRNGKey(100)


def _plot_periodic_system() -> None:

    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    fig.savefig(
        "examples/ballistic_langevin/potentials_and_trajectories/1d_periodic.potential.pdf"
    )


def _plot_ballistic_trajectory() -> None:

    key = jrandom.PRNGKey(100)

    result = solve_single(
        normalized_system.with_gamma(0.0),
        TimeSpan(
            t_end=normalized_system.units.time_into(10e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        (np.full((1,), 0.0), np.full((1,), 3.35)),
        _key=key,
    )

    elastic, inelastic = breakdown_filtered_ballistic_trajectory_butterworth(
        result,
        minimum_timescale=1 / normalized_system.gamma,
    )

    fig, ax = _get_two_panel_figure()

    _, _ax_0, line = plot_single_x_evolution(result=result.with_si_units(), ax=ax[0])
    _, _ax_0, line_e = plot_single_x_evolution(result=elastic.with_si_units(), ax=ax[0])

    _, _ax_1, line_i = plot_single_x_evolution(
        result=inelastic.with_si_units(), ax=ax[1]
    )

    line_i.set_color("C2")

    ax[0].legend(
        handles=[line, line_e],
        labels=["full", "elastic"],
    )
    ax[1].legend(
        handles=[line_i],
        labels=["inelastic"],
    )
    ax[0].set_ylim(2.5e-10, 6.5e-10)
    ax[0].set_xlim(1e-12, 2e-12)
    ax[1].set_xlim(1e-12, 2e-12)
    ax[1].set_ylim(-1e-10, 1e-10)
    fig.savefig(
        "examples/ballistic_langevin/potentials_and_trajectories/1d_periodic.trajectory.pdf"
    )


def _animate_result() -> None:
    result = solve_single(
        normalized_system.with_gamma(0.0),
        TimeSpan(
            t_end=normalized_system.units.time_into(10e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        (np.full((1,), 0.0), np.full((1,), 2.2)),
        _key=key,
    )

    ani = animate_elastic_inelastic_breakdown_1d_periodic(
        result, normalized_system, start_time=1e-12, end_time=2e-12
    )
    ani.save(
        "examples/ballistic_langevin/potentials_and_trajectories/elastic_inelastic_breakdown.mp4",
        writer="ffmpeg",
        dpi=150,
    )


if __name__ == "__main__":
    _plot_periodic_system()
    _plot_ballistic_trajectory()
