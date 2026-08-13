from pathlib import Path

import jax
import jax.random as jrandom
import numpy as np
from scipy.constants import Boltzmann
from tqdm import tqdm

from classical_diffusion.analysis import plot_isf
from classical_diffusion.langevin import (
    breakdown_filtered_ballistic_trajectory_butterworth,
    calculate_effective_mass_01_fcc,
    get_full_effective_mass_from_free,
    get_initial_conditions,
    get_over_barrier_initial_conditions,
    get_under_barrier_probability_ballistic,
    plot_effective_mass_ratio,
    plot_exact_gaussian_isf,
    solve_ballistic_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import (
    PeriodicSystemFCC,
    UnitSystem,
    get_diffusion_time,
)
from classical_diffusion.util import cached, disabled_timing, hash_array

system = PeriodicSystemFCC(
    gamma=4e11,
    temperature=102,
    m=6e-26,
    delta_x=3e-10,
    barrier_energy=3e-21,
    units=UnitSystem(),
)

key = jrandom.PRNGKey(100)

normalized_system = system.with_normalized_units()


def _plot_effective_mass_isf() -> None:  # ruff:ignore[too-many-locals]

    fig, ax = get_fancy_figure()
    direction = np.array([0, 1])
    delta_k = tuple(
        2 * np.pi / system.delta_x * 0.2 * direction / np.linalg.norm(direction)
    )

    initial_conditions = get_initial_conditions(normalized_system, n_samples=2000)
    result_full = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(10e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions,
        _key=key,
    )

    prob_under_barrier = get_under_barrier_probability_ballistic(
        normalized_system,
        result_full.x_points,
        result_full.p_points,
        normalized_system.barrier_energy,
    )

    elastic_result, _ = breakdown_filtered_ballistic_trajectory_butterworth(
        result_full,
        minimum_timescale=get_diffusion_time(
            normalized_system, characteristic_length=normalized_system.delta_x / 0.5
        ),
    )

    _, ax, line_0, _ = plot_isf(
        result=elastic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    _, ax, line_1 = plot_exact_gaussian_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        effective_mass=np.array([[system.m]]),
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")

    initial_conditions = get_over_barrier_initial_conditions(
        system=normalized_system,
        barrier_energy=normalized_system.barrier_energy,
        n_samples=2000,
    )
    result_free = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(5e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions=initial_conditions,
        _key=key,
    )

    elastic_result_free, _ = breakdown_filtered_ballistic_trajectory_butterworth(
        result_free,
        minimum_timescale=get_diffusion_time(
            normalized_system, characteristic_length=normalized_system.delta_x / 0.5
        ),
    )

    effective_mass = UnitSystem().mass_into(
        get_full_effective_mass_from_free(
            elastic_result_free,
            prob_under_barrier=prob_under_barrier,
        ),
        units=normalized_system.units,
    )

    _, ax, line_2 = plot_exact_gaussian_isf(
        system=system.with_si_units(),
        ax=ax,
        delta_k=delta_k,
        effective_mass=effective_mass,
    )
    line_2.set_label("simulation")
    line_2.set_linestyle(":")

    lattice_vectors = np.asarray(normalized_system.lattice_vectors)

    frac_coords = [0.5, 0.5]
    saddle_cart = lattice_vectors @ frac_coords

    frac_height = [1, 0]
    height = (lattice_vectors @ frac_height)[1]

    effective_mass_1d = UnitSystem().mass_into(
        calculate_effective_mass_01_fcc(
            system=normalized_system, x0=saddle_cart[0], height=height
        ),
        units=normalized_system.units,
    )

    effective_mass_matrix = np.zeros((2, 2))
    effective_mass_matrix[1, 1] = effective_mass_1d
    effective_mass_matrix[0, 0] = UnitSystem().mass_into(
        1e-12,
        units=normalized_system.units,
    )

    _, ax, line_3 = plot_exact_gaussian_isf(
        system=system.with_si_units(),
        ax=ax,
        delta_k=delta_k,
        effective_mass=effective_mass_matrix,
    )
    line_3.set_label("analytical approximation")
    line_3.set_linestyle(":")

    ax.set_xlim(0, 10e-12)
    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig(
        "examples/ballistic_langevin/Chuddley_Elliot/2d_fcc_01.Chuddley_Elliot.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _plot_effective_mass_ratio() -> None:

    def _solve_effective_mass_path(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
        temperature: float,
        delta_x: float,
        m: float,
        end_time: float,
        n_samples: int,
        _key: jax.Array,
        barrier_energy: np.ndarray,
    ) -> Path:
        filename = f"{temperature}_{delta_x}_{m}_{end_time}_{n_samples}_{key}_{hash_array((barrier_energy,))}.npz"
        return Path("examples/data") / filename

    @cached(_solve_effective_mass_path)
    def _effective_mass_simulation(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
        temperature: float,
        delta_x: float,
        m: float,
        end_time: float,
        n_samples: int,
        _key: jax.Array,
        barrier_energy: np.ndarray,
    ) -> tuple:
        jrandom.split(jrandom.PRNGKey(100), barrier_energy.size)

        kbt = temperature * Boltzmann
        barrier_energy_grid = barrier_energy * kbt

        full_effective_mass_ratio = np.zeros_like(barrier_energy)
        full_effective_mass_exact_ratio = np.zeros_like(barrier_energy)

        with disabled_timing():
            for _idx, i in enumerate(
                tqdm(np.ndindex(barrier_energy.shape), total=barrier_energy.size)
            ):
                system = PeriodicSystemFCC(
                    gamma=0,
                    temperature=temperature,
                    m=m,
                    delta_x=delta_x,
                    barrier_energy=barrier_energy_grid[i],
                    units=UnitSystem(),
                )
                normalized_system = system.with_normalized_units()
                initial_conditions = get_initial_conditions(
                    system=normalized_system,
                    n_samples=2000,
                )

                result_full = solve_ballistic_ensemble(
                    normalized_system,
                    TimeSpan(
                        t_end=normalized_system.units.time_into(
                            1e-30, units=UnitSystem()
                        ),
                        n_steps=2,
                    ),
                    initial_conditions,
                    _key=key,
                )

                prob_under_barrier = get_under_barrier_probability_ballistic(
                    normalized_system,
                    result_full.x_points,
                    result_full.p_points,
                    normalized_system.barrier_energy,
                )

                initial_conditions = get_over_barrier_initial_conditions(
                    normalized_system,
                    barrier_energy=normalized_system.barrier_energy,
                    n_samples=n_samples,
                )
                result_free = solve_ballistic_ensemble(
                    normalized_system,
                    TimeSpan(
                        t_end=normalized_system.units.time_into(
                            5e-12, units=UnitSystem()
                        ),
                        n_steps=1000,
                    ),
                    initial_conditions=initial_conditions,
                    _key=key,
                )

                elastic_result_free, _ = (
                    breakdown_filtered_ballistic_trajectory_butterworth(
                        result_free,
                        minimum_timescale=get_diffusion_time(
                            normalized_system,
                            characteristic_length=normalized_system.delta_x / 0.5,
                        ),
                    )
                )

                full_effective_mass_ratio[i] = get_full_effective_mass_from_free(
                    elastic_result_free,
                    prob_under_barrier=prob_under_barrier,
                )[1, 1]

                lattice_vectors = np.asarray(normalized_system.lattice_vectors)

                frac_coords = [0.5, 0.5]
                saddle_cart = lattice_vectors @ frac_coords

                frac_height = [1, 0]
                height = (lattice_vectors @ frac_height)[1]

                full_effective_mass_exact_ratio[i] = calculate_effective_mass_01_fcc(
                    system=normalized_system, x0=saddle_cart[0], height=height
                )

            return (
                full_effective_mass_ratio,
                full_effective_mass_exact_ratio,
            )

    barrier_energy = np.linspace(1, 5, 20)
    (
        full_effective_mass_ratio,
        full_effective_mass_exact_ratio,
    ) = _effective_mass_simulation(
        temperature=105,
        m=6e-26,
        delta_x=3e-10,
        n_samples=1000,
        end_time=5e-12,
        _key=key,
        barrier_energy=barrier_energy,
    )

    fig, ax = get_fancy_figure()
    _, ax, line_0 = plot_effective_mass_ratio(
        barrier_energy=barrier_energy,
        mass_ratio=full_effective_mass_ratio,
        ax=ax,
    )
    line_0.set_label("simulation")

    _, ax, line_1 = plot_effective_mass_ratio(
        barrier_energy=barrier_energy,
        mass_ratio=full_effective_mass_exact_ratio,
        ax=ax,
    )
    line_1.set_label("Analytical approximation")
    ax.legend(handles=[line_0, line_1])
    fig.savefig(
        "examples/ballistic_langevin/Chuddley_Elliot/2d_fcc_01.barrier_energy_comparison.pdf",
        dpi=1000,
    )


if __name__ == "__main__":
    _plot_effective_mass_isf()
    _plot_effective_mass_ratio()
