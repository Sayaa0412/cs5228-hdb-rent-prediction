# Feature and source contract

## Model inputs

Base reference (13): numerical MONTH_INDEX, FLOOR_AREA_SQM, LEASE_AGE, LATITUDE, LONGITUDE, MAX_FLOOR, BUILDING_AGE; categorical TOWN, FLAT_TYPE, FLAT_MODEL, REGION, STREET, ADDRESS. BLOCK is retained in ADDRESS, not treated as an ordered number. FURNISHED and FEE are constants and excluded. Rent is the target.

Static market inputs (12): FLOOR_AREA_SQM, LATITUDE, LONGITUDE, MAX_FLOOR, LEASE_START_YEAR, COMPLETION_YEAR, TOWN, FLAT_TYPE, FLAT_MODEL, REGION, STREET, ADDRESS. Month is removed; evolving ages become fixed years.

Exact-history reference adds three fields per lag: mean log-rent anomaly, log(1+count), and source-month-unavailable indicator. Lags 12 and 24 give 19 total fields through forecast horizon 12. The separately fitted 24-only profile has 16 total fields and is used at horizons 13–24. A missing group in an observable month differs from a source month beyond the origin. No evaluation label updates history.

Final market (15): the 12 static fields plus 24-month normalized mean rent, log count and source availability. This mean uses address-excluded market scales; it is not the reference's log-anomaly feature. The matched boosted market control uses the same inputs. Ridge uses numerical median imputation and standardization, one-hot categories with unknown levels ignored, alpha=10, lsqr tolerance=1e-6, max_iter=5000 and scale-squared sample weights normalized to mean one.

Enriched learner control (25): base13 + PLANNING_AREA + SUBZONE + CENSUS_LOG_RESIDENTS, CENSUS_HDB_SHARE, CENSUS_SMALL_WITHIN_HDB, CENSUS_3ROOM_WITHIN_HDB, CENSUS_4ROOM_WITHIN_HDB, CENSUS_5ROOM_WITHIN_HDB, CENSUS_CONDO_SHARE, CENSUS_LANDED_SHARE, CENSUS_OTHER_SHARE, CENSUS_MISSING. The attached feature_contract.json retains the original full contract.

## Source audits

| Source | Historical availability and join | Coverage/use |
| --- | --- | --- |
| Course HDB block data | Normalized town/block/street address, unique lookup keys; coordinates, height, completion and region | Retained base inputs; height is not the rented unit's floor. Snapshot metadata can be retrospective for early transactions. |
| HDB Property Information | Release 2022-04-04, 12,495 unique address keys; uses, total units, room-type counts | 150,000/150,000 training and 50,000/50,000 test rows matched. Not retained in final model. Catalogue: https://data.gov.sg/datasets/d_17f5382f26140b1fdae0ba2ef6239d2f/view |
| Census 2020 | Original publication 2021-06-18; later portal creation 2024-10-22 is not the original release. Exact normalized planning-area/subzone join; no fuzzy match | All 150,000 training rows matched. Resident composition, not dwelling-unit counts. Pre-publication training rows have a retrospective-information limitation. Used in enriched controls, not final. https://data.gov.sg/datasets/d_7f243956483d5901f237e6f87b096636/view |
| MRT opening history | Announced releases and physical-station normalization; operational status changes from next full month after opening | Origin-known announcements only. Distance/graph controls, not final. Full dated links: transport_release_rules.json. |
| Quarterly HDB resale medians | Town × flat type, latest released quarter, first full month after publication; query index frozen at origin | Four evaluation-window coverage: 93.15%, 91.08%, 91.31%, 91.98%. Jan–Apr 2021 missing earlier releases are not backfilled. 14 quarters/2,184 cells reconciled with dated annexes; not final. See resale_release_calendar.csv. |

The complete experiment summary includes additional historical resale-layout and private-market controls. Source publication and query-label exclusion are different issues: a pre-origin release need not have been known at each older fitting transaction. No new source data were downloaded for this editorial revision.
