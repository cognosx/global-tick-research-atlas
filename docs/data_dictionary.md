# Data dictionary

One section per table. Fill in as each layer is frozen.

## Data layers

| Layer | Content | Shared? |
| --- | --- | --- |
| L0 raw | Source exports, harvest date, query | No (licences) |
| L1 clean | One row per record, encoding fixed, normalized DOI/year/type/language | IDs only |
| L2 canonical | One row per work after dedup, tick-relevance label, validation flag | Yes (Zenodo) |
| L3 enriched | Linked tables below | Yes (Zenodo) |
| L4 analysis | Small tables per paper, figures | Yes (GitHub) |

## L3 tables

| Table | Key | Main columns | Source |
| --- | --- | --- | --- |
| works | oa_id | title, year, type, language, venue, doi, pmid, is_tick, theme_climate | OpenAlex + sources |
| authorships | oa_id, author_position | author_id, orcid, institution_ror, country, is_first, is_last | OpenAlex, ROR |
| funding | oa_id, funder_id | funder_name, funder_country, grant_id | OpenAlex, Crossref, Europe PMC |
| topics | oa_id, topic_id | topic_label, theme_group, probability | BERTopic + expert taxonomy |
| entities | oa_id, entity_id | tick_species, pathogen, host, location, method | scispaCy + LLM consensus |
| venues | issn_l | doaj, scimago_quartile, publisher, country | DOAJ, Scimago |
| oa_status | oa_id | oa_final, license | Unpaywall |
| citations | oa_id | cited_by_count, fwci, citation_percentile | OpenAlex, OpenCitations |
| fulltext_sections | oa_id | has_data_availability, has_ethics, has_funding, has_conflicts | Open full texts |

## Column template

| Column | Type | Meaning | Source | Coverage | Known limits |
| --- | --- | --- | --- | --- | --- |
| | | | | | |
