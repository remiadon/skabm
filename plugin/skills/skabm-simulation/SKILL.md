---
name: skabm-simulation
description: Taking a user's request ("simulate the COVID trade-off in New York", "an app for this paper") to a working skabm simulation with a simple Voilà app — pick and scope the source paper, build its model with skabm-core, load the paper's own data, run it on the JAX or SPARQL engine, and show it in a Voilà notebook the user serves themselves. Use when the user asks for a new simulation, use case or app built on skabm. Loads skabm-core for the model; polars for data; maplib only for the SPARQL engine; dataviz before drawing charts.
---

# skabm-simulation: from a request to a Voilà app

A simulation is a model, written with the **skabm-core** skill, plus a Voilà notebook
that runs it: `voila app.ipynb` on the user's own machine or server. It is the paper's
model on the paper's own region and data, and the app says plainly where it departs from
the paper. There is no hosting budget to fit: size the run to what the question needs.

## Workflow

1. **Pick the source.** A well-documented, peer-reviewed paper, run on the region and
   data it states, unless the user asks otherwise. Credit the original paper; an
   open-source implementation is a link.
2. **Scope it, and report before building.** Report:
   - the equations;
   - which values are published, and which are only calibrated (those stay required);
   - each data source, public or proprietary, with its access terms and cost checked
     today, and anything unchecked labelled unverified;
   - what the app leaves out.
   Prefer the authors' replication package to rebuilding inputs. Get the user's go.
3. **Compose the model** from `skabm.behaviour` (its README lists the mechanisms by
   agent): `RULES = [...]` in the paper's order in a `model.py` next to the app, with the
   paper's values passed by name in `params=`. If skabm's showcase already runs the paper
   (`notebooks/` at the release tag, github.com/remiadon/skabm), start from its `RULES`
   and its data loading. Build the opening state with the behaviours' polars functions
   (`skabm/behaviour/README.md`, Opening state) and `skabm.space`. Write a new rule dict (the
   skabm-core skill) only for a mechanism no behaviour covers, and only over the agents
   and fields skabm ships (`skabm.template.SCHEMA`). If the request needs an agent or
   field skabm lacks, stop and report the gap as a feature request for skabm; don't
   declare one yourself. Add a few tests of what the paper reports.
4. **Pick the engine.**
   - JAX, `jax.jit(dsl.jax_tick(RULES))` stepped from `dsl.arrays(frames, RULES)`: for
     large populations, or when gradients matter.
   - SPARQL, `RDFSimulator` (maplib skill): when the rules need `total_by`,
     `running_sum` or `pick`, or the user wants to query or edit the world as a graph.
5. **Write the app**: one Voilà notebook, kept short.
   - A markdown cell: the paper, what the app shows, and where it departs from the paper.
   - A code cell that loads data, using `skabm.player.fetch` for files inside a remote zip.
   - A few widgets for the questions the user wants answered.
   - One function that runs the model and draws the result.
   Draw step by step with `skabm.player.Player` only if the user wants to watch the run
   unfold. Use lonboard for maps, sending geometry once and changing colours after that.
   Load the dataviz skill before the first chart.
6. **Verify.**
   - `jupyter execute app.ipynb` runs clean.
   - `voila app.ipynb` opens, and a run works from its widgets.
   - One headline number is compared with the paper's own result, and the gap is stated
     in the app.
