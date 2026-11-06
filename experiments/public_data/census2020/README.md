# Census 2020: a small, historical neighborhood table

Source: Singapore Department of Statistics, Resident Population by Planning Area/Subzone of Residence and Type of Dwelling (Census 2020).

- Official dataset: https://data.gov.sg/datasets/d_7f243956483d5901f237e6f87b096636/view
- Release chronology: https://www.singstat.gov.sg/-/media/files/publications/cop2020/admin/cop2020admin.pdf
- Public API response archived as `census2020_dwelling_raw.json`; retrieval URL, size and SHA-256 are in `download_manifest.json`.
- Original statistical release: 18 June 2021. The portal's later creation date is not the original release date.
- 388 rows: one national total, 55 planning-area totals, 332 subzones. The archived JSON is 117,860 bytes. No individual records are downloaded.

These figures count **residents by dwelling type**, not dwellings, rented units, households, population density or tenant income. A dash denotes nil/negligible/not significant; `NA` denotes missing information. Published values are rounded. Shares need not add to precisely one.

`work/prepare_census_features.py` joins the provided block metadata to the provided rental records by normalized address, then uses exact planning-area/subzone keys. It does not fuzzily match or drop unmatched rows. The preparation audit reports all 150,000 training rows matched. Original planning-area/subzone metadata is a supplied static snapshot; that limitation remains even though the census is historical.

The table was available at each of the four 2023–2024 model-fitting origins. Its use for early-2021 training observations means information available when fitting the later model, not information available at each individual approval date. No later census snapshot is substituted.

Control design: the same GPU model with the original 13 fields, with only two geographic labels added, and with those labels plus housing-composition features. Because composition is constant within each subzone, it provides a numerical description for sharing across places; it is not new within-subzone information. Improvement must be demonstrated rather than assumed.
