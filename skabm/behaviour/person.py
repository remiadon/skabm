"""People: where they live, whom they meet, whether they fall ill, and whether they work.

- Moving out when too few neighbours share one's group, to a vacant cell ranked by a
  draw: Schelling (1971), J. Math. Sociology 1(2), as AMBER's segregation demo runs it.
- Meeting people at home, at school, at work and in the community, through the places
  they share; an infection moving through latent, pre-symptomatic, symptomatic,
  asymptomatic, removed and dead; workers hired and fired person by person, to the count
  their industry sets: Pangallo et al. (2024), Nat. Hum. Behav. 8, 264-275.

The epidemic's transmissibility and fear_infection are calibrated by Pangallo et al.
(2024) and published only as posteriors (SI Fig. 13), so they stay required.
"""

import numpy as np
import polars as pl
import sympy as sp
from sympy.stats import Uniform

from skabm.dsl import Agents, coalesce, pick, running_sum, sum_over, total

Cell, Person = Agents("Cell"), Agents("Person")
Place, Region, Industry = Agents("Place"), Agents("Region"), Agents("Industry")

want_similar = sp.Symbol("want_similar")
transmissibility, fear_infection = sp.symbols("transmissibility fear_infection")
asymptomatic_infectiousness, presymptomatic_share = sp.symbols(
    "asymptomatic_infectiousness presymptomatic_share"
)
incubation_days, presymptomatic_days, infectious_days = sp.symbols(
    "incubation_days presymptomatic_days infectious_days"
)
death_days, notification_days = sp.symbols("death_days notification_days")
unattributed_weight = sp.Symbol("unattributed_weight")

PARAMETERS = {
    want_similar: 0.375,  # 3 of 8 Moore neighbours, Schelling (1971)
    asymptomatic_infectiousness: 0.5,  # r, Pangallo et al. (2024) SI Table 5 (assumed)
    presymptomatic_share: 0.5,  # k, Pangallo et al. (2024) SI Table 5
    incubation_days: 5.0,  # epsilon, Pangallo et al. (2024) SI Table 5
    presymptomatic_days: 2.0,  # gamma, Pangallo et al. (2024) SI Table 5
    infectious_days: 2.5,  # mu, time to isolation, Pangallo et al. (2024) SI Table 5
    death_days: 12.5,  # delta, Pangallo et al. (2024) SI Table 5
    notification_days: 7.0,  # T_n, Pangallo et al. (2024) SI Table 5
    # community contact weight at places of no economic activity, Pangallo et al.
    # (2024) SI Table 3
    unattributed_weight: 1742638.0,
}

# Schelling (1971)
occupancy = {
    Cell.occupied: sum_over(Person.location, 1),
    Cell.resident: sum_over(Person.location, Person.group),
    Cell.draw: Uniform("u", 0, 1),
}
around = Person.location.neighbor
happiness = {
    Person.share_similar: sum_over(
        around,
        sp.Piecewise(
            (1, sp.Eq(around.resident, Person.group) & (around.occupied > 0)),
            (0, True),
        ),
    )
    / sp.Max(sum_over(around, around.occupied), 1),
    Person.draw: Uniform("u", 0, 1),
}
cell_rank = {
    Cell.rank: running_sum(
        Cell.draw, sp.Piecewise((1, sp.Eq(Cell.occupied, 0)), (0, True))
    )
}
unhappy = Person.share_similar < want_similar
person_rank = {
    Person.rank: running_sum(Person.draw, sp.Piecewise((1, unhappy), (0, True)))
}
relocate = {
    Person.location: coalesce(
        pick(
            Cell,
            sp.Eq(Cell.rank, Person.rank) & (sp.Eq(Cell.occupied, 0)) & unhappy,
            Cell.rank,
        ),
        Person.location,
    )
}

# The states of Pangallo et al. (2024)'s epidemic module, Methods, Fig. 6c
(
    SUSCEPTIBLE,
    LATENT,
    PRESYMPTOMATIC,
    SYMPTOMATIC,
    ASYMPTOMATIC,
    REMOVED,
    DYING,
    DEAD,
    REPORTED,
) = range(9)
state = Person.state
# Methods: transmission rates beta_S, beta and r beta; beta_S / beta makes a share k of
# a symptomatic case's transmission pre-symptomatic (SI Table 5's k)
infectiousness = sp.Piecewise(
    (
        presymptomatic_share
        * infectious_days
        / (presymptomatic_days * (1 - presymptomatic_share)),
        sp.Eq(state, PRESYMPTOMATIC),
    ),
    (1, sp.Eq(state, SYMPTOMATIC)),
    (asymptomatic_infectiousness, sp.Eq(state, ASYMPTOMATIC)),
    (0, True),
)
# the region's day: deaths reported yesterday, the lockdown, school closures and work
# from home (macro.region_day)
deaths, lockdown, mandates = (
    total(Region.deaths),
    total(Region.lockdown),
    total(Region.mandates),
)
# Methods eqs. 2, 4, 6: fear keeps those who can work from home away from work
fear = 1 - sp.exp(-fear_infection * deaths)
presence = Person.employed * (1 - Person.wfh * (mandates + (1 - mandates) * fear))
# SI §1, eqs. 14-18, through the places people share: sum_j omega_ij I_j is a sum over
# i's places of the place's weight times the infectiousness met there
place_contacts = {
    Place.mass: sum_over(Person.home, infectiousness)
    + sum_over(Person.school, infectiousness)
    + sum_over(Person.community, infectiousness)
    + sum_over(Person.work, infectiousness * presence)
}
# Methods: closures cut contacts in industry k by s_k = 1 - ESS_k, and fear those in
# customer-facing ones by Lambda^EPI (eq. 5)
weight, closed = Industry.community_weight, 1 - Industry.essential
facing = Industry.customer_facing
left_open = (
    unattributed_weight
    + total(weight)
    - lockdown * total(weight * closed)
    - fear * (total(weight * facing) - lockdown * total(weight * closed * facing))
) / (unattributed_weight + total(weight))
exposure = {
    Person.exposure: Person.home.weight * Person.home.mass
    + (1 - mandates) * coalesce(Person.school.weight * Person.school.mass, 0)
    + left_open * Person.community.weight * Person.community.mass
    + presence * coalesce(Person.work.weight * Person.work.mass, 0)
}
# Methods eq. 1 and Fig. 6c, each stage left with a daily probability of 1 over its
# mean length, as the replication package does (spread.c)
u = Uniform("u", 0, 1)
latent_exit = 1 / (incubation_days - presymptomatic_days)


def moves(*steps):
    """The next state: the first ``(state, bound)`` whose bound u is below, else stay."""
    return sp.Piecewise(*((to, u < bound) for to, bound in steps), (state, True))


caught = 1 - sp.exp(-transmissibility * Person.susceptibility * Person.exposure)
compartment_progression = {
    Person.state: sp.Piecewise(
        (moves((LATENT, caught)), sp.Eq(state, SUSCEPTIBLE)),
        (
            moves(
                (PRESYMPTOMATIC, latent_exit * Person.symptomatic),
                (ASYMPTOMATIC, latent_exit),
            ),
            sp.Eq(state, LATENT),
        ),
        (moves((SYMPTOMATIC, 1 / presymptomatic_days)), sp.Eq(state, PRESYMPTOMATIC)),
        (
            moves(
                (DYING, Person.fatality / infectious_days),
                (REMOVED, 1 / infectious_days),
            ),
            sp.Eq(state, SYMPTOMATIC),
        ),
        (moves((REMOVED, 1 / infectious_days)), sp.Eq(state, ASYMPTOMATIC)),
        (moves((DEAD, 1 / death_days)), sp.Eq(state, DYING)),
        (moves((REPORTED, 1 / notification_days)), sp.Eq(state, DEAD)),
        (state, True),
    ),
}
# Pangallo et al. (2024) SI §2.3, hiring and firing at the micro level: the paper draws
# the workers uniformly at random from the industry's lists; a fixed-shape rule draws
# each one with the chance that matches that count in expectation, closing the gap from
# the people's own count
job, home_based, employed = Person.industry, Person.wfh, Person.employed
target = home_based * job.fromhome + (1 - home_based) * job.inperson
staff = home_based * job.staff_fromhome + (1 - home_based) * job.staff_inperson
jobs = home_based * job.fromhome0 + (1 - home_based) * job.inperson0
flip = sp.Piecewise(
    ((target - staff) / (jobs - staff), (target > staff) & sp.Eq(employed, 0)),
    ((staff - target) / staff, (target < staff) & sp.Eq(employed, 1)),
    (0, True),
)
job_draws = {
    Person.employed: sp.Piecewise(
        (1 - employed, Uniform("u", 0, 1) < flip), (employed, True)
    )
}


def settle(
    cells: pl.DataFrame,
    density: float = 0.8,  # AMBER's p['density']: per-cell occupancy probability
    n_groups: int = 2,  # AMBER's p['n_groups']
    seed: int = 0,
) -> pl.DataFrame:
    """The Person population: each cell settled with probability *density* by one person
    (``settler_<cell>``), of a group drawn uniformly among *n_groups*."""
    rng = np.random.default_rng(seed)
    settled = rng.random(cells.height) < density
    group = np.floor(rng.random(cells.height) * n_groups)
    return pl.DataFrame(
        {
            "id": ("settler_" + cells["id"]).filter(pl.Series(settled)),
            "group": group[settled],
            "location": cells["id"].filter(pl.Series(settled)),
        }
    )


# Pangallo et al. (2024) SI Table 5, by age band 0-9, 10-19, ..., 70-79, 80+
SYMPTOMATIC_BY_AGE = (0.181, 0.181, 0.225, 0.225, 0.300, 0.300, 0.360, 0.360, 0.646)
IFR_BY_AGE = (0.00161, 0.00695, 0.0309, 0.0844, 0.161, 0.595, 1.93, 4.28, 7.80)  # %
# effective contacts a day per person in each layer, kappa, Mistry et al. (2021), as
# Pangallo et al. (2024) SI §1 uses them
KAPPA = {"home": 4.11, "school": 11.41, "work": 8.07, "community": 2.79}


def contact_population(
    people: pl.DataFrame,
    latent: int = 10,  # Pangallo et al. (2024) SI §4.1: 10 latent individuals at first
    seed: int = 0,
    kappa: dict = KAPPA,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The ``Person`` and ``Place`` frames from a synthetic population: ``id``,
    ``household``, ``tract``, ``county``, ``age``, ``worker``, ``wfh`` and ``industry``
    (null for non-workers), as Pangallo et al. (2024) model them.

    Age gives SI Table 5's susceptibility (0.56 under 19), chance of symptoms, and
    fatality, the IFR over that chance (Table 5's correction).  A person meets others at
    home and, children under 18, at a school of their tract (eqs. 17-18: weight 1 over
    the others there).  The paper's community and workplace layers come from mobile
    phone traces it cannot share; here the community is the person's tract and the
    workplace their industry in their county, every pair weighing alike.  Each layer is
    scaled by kappa over its people's mean weighted contacts (eq. 14).  Everyone starts
    susceptible and employed if a worker, but *latent* people drawn at random.

    ponytail: tract and county-industry pools stand in for the mobility data; a public
    source of where New Yorkers meet and work would replace them.
    """
    layers = {
        "home": people.select("id", place=pl.format("home:{}", "household")),
        "school": people.filter(pl.col("age") < 18).select(
            "id", place=pl.format("school:{}", "tract")
        ),
        "community": people.select("id", place=pl.format("community:{}", "tract")),
        "work": people.filter(pl.col("worker") == 1).select(
            "id", place=pl.format("work:{}:{}", "county", "industry")
        ),
    }
    others = pl.len().over("place") - 1
    each = {"home": 1 / others, "school": 1 / others}  # eqs. 17-18; else 1 per pair
    members = pl.concat(
        [
            frame.with_columns(layer=pl.lit(name), pair=each.get(name, pl.lit(1.0)))
            .with_columns(pl.when(others > 0).then("pair").otherwise(0.0))
            .with_columns(
                weight=kappa[name] * pl.col("pair") / (others * pl.col("pair")).mean()
            )
            for name, frame in layers.items()
        ]
    )
    links = members.pivot("layer", index="id", values="place")
    places = (
        members.group_by("place").agg(pl.col("weight").first()).rename({"place": "id"})
    )
    band = (pl.col("age") // 10).clip(0, 8).cast(pl.Int64)
    symptomatic = pl.lit(pl.Series(SYMPTOMATIC_BY_AGE)).gather(band)
    sick = people.sample(latent, seed=seed)["id"]
    persons = people.select(
        "id",
        "industry",
        susceptibility=pl.when(pl.col("age") < 19).then(0.56).otherwise(1.0),
        symptomatic=symptomatic,
        fatality=pl.lit(pl.Series(IFR_BY_AGE)).gather(band) / 100 / symptomatic,
        wfh=pl.col("wfh").cast(pl.Float64),
        employed=pl.col("worker").cast(pl.Float64),
        state=pl.when(pl.col("id").is_in(sick.implode()))
        .then(float(LATENT))
        .otherwise(float(SUSCEPTIBLE)),
    ).join(links, on="id", how="left", maintain_order="left")
    return persons, places.with_columns(mass=pl.lit(0.0))
