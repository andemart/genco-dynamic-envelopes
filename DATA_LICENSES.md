# Data provenance and licences

The Apache-2.0 licence in [`LICENSE`](LICENSE) covers the project code and documentation. It does not cover third-party data in this repository, which keep their own terms.

| Path | What it is | Source and terms |
|---|---|---|
| `data/feeder_topology/ENWL1/` | OpenDSS files of ENWL LV network 1 as a three-phase four-wire model: line codes with 4×4 phase-and-neutral matrices from BS 5467 cable tables, loads, coordinates, master file | Electricity North West (ENWL) LV network model in a four-wire representation. **Upstream distribution and licence: to be confirmed.** Modifications in this repository: the neutral is earthed only at the transformer (`Reactor.grndTR`; the source-side earth is commented out), and load shapes are replaced by the household profiles below at run time. |
| `data/load_profiles/*.xlsx` | Legend workbook for the Fluvius digital-meter dataset | Fluvius. Terms: Fluvius open-data licence, https://opendata.fluvius.be/p/licentieopendatafluvius (redistribution of this workbook to be confirmed). |
| Fluvius meter profiles (not in the repository) | 2,400 anonymised residential digital meters, quarter-hourly, 2024, eight technology groups | Fluvius Open Data: https://opendata.fluvius.be/explore/assets/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/ . Download so the CSVs sit at `data/load_profiles/<folder>/P6269_Open_Data_<group>.csv`. |
| `data_out/` (not in the repository) | Derived run files, four-wire labels, features, LV checkpoints | Derived from the Fluvius meters. Not yet redistributed while the licence terms for meter-level derivatives are checked; SHA-256 hashes in `results/run_manifest_*.json`. |
| Official GENCO checkpoint (not in the repository) | `gridfm/genco-opf-datakit`, case118 | https://huggingface.co/gridfm/genco-opf-datakit, under its own licence. |
