# Data source and preparation

The experiment uses the Ausgrid Solar Home Electricity Data, originally published by Ausgrid and described by Ratnam et al., *International Journal of Sustainable Energy* 36(8), 787–806, doi:10.1080/14786451.2015.1100196.

- Government catalogue: https://data.gov.au/data/dataset/nsw-solar-home-electricty-data
- Qualified preservation copy: https://github.com/pierre-haessig/ausgrid-solar-data
- Archive download: https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip
- Stated source license: Creative Commons Attribution 3.0 Australia, https://creativecommons.org/licenses/by/3.0/au/
- ZIP SHA-256: `5a766f52b6c8b3b72730380f4422e478934bc94640a4b089dd0e0e3c055c5d82`

Credit **Ausgrid** as original data provider and **Pierre Haessig** for the preservation copy. This repository does not claim their endorsement. The exact source-file bytes have not been authenticated by the original publisher. The source adjudication records that limitation; verification of a mirror hash is not publisher authentication.

Run `python reproduce.py fetch-data`. If the public host is unavailable, obtain that exact ZIP independently and use `python reproduce.py fetch-data --from-file /path/to/archive.zip`. Different ZIP bytes are rejected; do not bypass the check or substitute a newly cleaned dataset silently.

The loader uses the 2011–2012 and 2012–2013 CSV members and verifies each member hash. Half-hour kWh are end-labelled; midnight closes the preceding day. Whole-site load is measured general consumption plus controlled load (GC + CL), and PV is the separate gross-generation channel (GG). Missing CL is not filled with zero. Invalid, incomplete and unresolved clock-transition windows remain excluded. Eligibility and all registered history/test dates are in the supplied metadata; no controller outcomes determine this screen.

The selected 28-day histories precede continuous July 2012 and January 2013 30-day tests. The source uses Sydney civil time; retained windows avoid unresolved DST transitions. Meter receipts are modeled at interval end, not historical delivery receipts. Local load/PV power is reconstructed as constant within each half hour. Synthetic tariff prices are known schedules.

Reference summaries and selected traces are derived from these data: GC and CL were summed, the fixed time windows selected, and modeled controllers/batteries applied. Those transformations are not additional household measurements. Their data attribution remains separate from the MIT license on the software.

The metadata JSON files are historical records. Fields such as `controllers_run: false` in the original pre-execution screen describe the date of that record; they are not the current execution status.
