# SO2 Facility-z/h0 AERMOD/PINN Dataset

This folder contains the SO2 version of the Houston 2021 facility-z/h0 AERMOD/PINN dataset.

## Contents

- `scenarios_houston_2021_facility_z_h0_so2/`: parsed PINN training shards.
- `normalization_metadata_facility_z_h0_so2.json`: normalization metadata for the 13-column shard schema.
- `facility_z_h0_so2_parity_summary.json`: parity and row-count QA summary.
- `facility_z_h0_so2_decay_metadata.json`: AERMOD SO2 chemistry/configuration metadata.
- `so2_inert_comparison_summary.json`: SO2 decay-vs-inert comparison QA.
- `run1_top20_facility_srcgroup_members_2021_so2.csv`: fixed-geometry SO2 source-member table.
- `so2_fixed_geometry_allocation_2021.csv`: SO2 allocation summary against the fixed facility-z/h0 geometry.
- `so2_zero_facility_decision_2021.csv`: zero/near-zero facility QA table.

## Dataset Summary

- Pollutant: SO2
- Facilities: 20 (`FAC001` through `FAC020`)
- Selected meteorological hours: 110
- AERMOD runs: 2,200
- Spatial receptors: 4,637
- Receptor-height points: 23,185
- Parsed rows: 51,007,000
- Parsed shards: 40
- Shard schema: one `data` array per `.npz`, 13 `float32` columns

## Chemistry

AERMOD v24142 regulatory-default urban SO2 treatment:

- `MODELOPT CONC DFAULT`
- `POLLUTID SO2 H1H`
- `URBANSRC ALL`
- 4-hour urban SO2 half-life, equivalent decay coefficient `4.813522087221842e-05 s^-1`

## QA

Strict QA passed with zero errors. Facility identity, source geometry, receptor map, met-hour selection, and run manifest match the existing facility-z/h0 baseline. The SO2 decay-vs-inert comparison showed all matched nonzero SO2-decay concentrations below the inert counterpart.
