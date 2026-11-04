# General SERS Analysis Database

Last updated: November 4, 2026

This database stores individual spectra, task labels, and molecular structures in a consistent directory layout. Each `Data_id` normally represents one specific analysis task. Allowed task values are:

- `Classification`
- `Single-Component Quantification`
- `Multi-Component Quantification`

These values describe the final analysis objective. Internal workflows, including classification followed by quantification, are selected by subsequent algorithms rather than prescribed by the database.

Public-data imports are recorded below. Actual records are listed in `content.csv`; source-specific eligibility limits are described below and in each dataset's `metadata.json`. Ordinary Raman is explicitly marked as auxiliary and must be excluded for SERS-only experiments.

## 1. Directory layout

```text
General_SERS_Analysis_Database/
  content.csv
  README.md
  data/
    <Data_id>/
      metadata.json                    Recommended provenance and experiment metadata
      Class_1_0_1_000001/               Example classification group
        spectrum_000001.csv
        spectrum_000002.csv
      OR Quan_1.0_0.0_2.5_000001/       Example quantification group
        spectrum_000001.csv
        spectrum_000002.csv
  molecules/
    <molecule_name>.smi
    structure_sources.json             Recommended structure provenance record
```

The names and concentrations above are illustrative, not actual records. Classification and quantification groups belong to separate task datasets rather than being mixed under one `Data_id`.

## 2. Index fields

Keep exactly these ten field names and their order:

```csv
Data_id,task,modality,n_classes,molecules,quant_unit,quant_datatype,n_group,n_spectra,data_source
```

| Field | Definition and conventions |
|---|---|
| `Data_id` | Unique task-dataset identifier, exactly matching its directory under `data/`. Read it as a string to preserve leading zeros. |
| `task` | One of the three case-sensitive values listed above. Do not use internal operations such as denoising or unmixing as task values. |
| `modality` | Measurement/signal-generation category: `Raman`, `direct-SERS`, `indirect-SERS`, `mixed`, or `unknown`. Use the operational definitions below; do not infer from task labels. |
| `n_classes` | Number of distinct substances or molecules involved in the task, **not the number of possible mixture-composition classes**. |
| `molecules` | Ordered molecule identifiers separated by `_`. The number of identifiers equals `n_classes`. Each identifier matches its `.smi` filename stem where a molecular structure is available. |
| `quant_unit` | Quantitative target unit, such as `uM`, `nM`, or `carbons`. For classification, use the literal string `null`. Define the concentration basis in the metadata. |
| `quant_datatype` | Target numerical type, such as `int` or `double`. For classification, use the literal string `null`. Carbon counts may use `int`; concentrations normally use `double`. |
| `n_group` | Number of sample-group directories. Groups with identical labels may have different indices. Do not count metadata files. |
| `n_spectra` | Total valid individual-spectrum CSV files across all groups, excluding metadata, summary spectra, and duplicate backups. |
| `data_source` | Source DOI, publication URL, or original repository URL. Multiple links may be separated by `;`. Store detailed provenance and conversion records in `metadata.json`. |

For DA, E, and NE, `n_classes` is 3 even if classification distinguishes eight compositions: a blank, three single-component compositions, three binary compositions, and one ternary composition.

If one source supports both classification and quantification, create separate task identifiers while recording their common source study. These are not independent research sources and should be held out together when evaluating generalization to new sources.

### Modality definitions

This is an operational classification by the origin of the measured analytical signal. The exact case-sensitive values are:

| Value | Meaning | Assignment rule |
| --- | --- | --- |
| `Raman` | Conventional Raman spectra without surface enhancement. | Includes the four-sugar dataset. Do not equate low signal intensity with absence of enhancement. |
| `direct-SERS` | The enhanced Raman signatures of the target analyte itself are the analytical readout. | Functionalized capture/enrichment substrates can still be direct-SERS. The mere presence of a ligand or modified particle does not establish an indirect assay. |
| `indirect-SERS` | The target is inferred from another signal-bearing chemical species, such as a Raman reporter, reaction product or derivatization product. | This database explicitly includes product-based chemical readout here as an operational convention; literature terminology may differ. Record the precise mechanism. |
| `mixed` | One indexed task intentionally contains more than one category. | Record the category per group or per spectrum. Use separate task records where appropriate, retaining shared-sample links. |
| `unknown` | Available sources do not establish which category applies. | Document the missing evidence; do not automatically assign direct-SERS because a paper calls its method SERS. |

Blank/control spectra inherit their assay's modality, even though the analyte is absent. Modality is independent of `task`: classification or quantification can use any supported modality. A dataset need not contain every modality.

`molecules` continues to identify the **analytical targets**. For indirect-SERS, add `signal_species`, reporter/derivative identities and verified structures, and the target-to-signal mechanism to `metadata.json`. Do not replace a target's SMILES with a reporter's structure under the same molecule identifier. Unknown reaction products remain unknown. Experimental pH, substrate/ligand identity, enrichment and derivatization conditions belong in metadata regardless of category.

The direct/indirect distinction by analyte-versus-reporter signal is discussed in [Biomedical SERS](https://doi.org/10.1039/D4CS00090K); the treatment of derivatization above is this database's explicit convention.

### Molecule identifiers

Use stable filename-compatible identifiers such as `DA`, `E`, and `NE`, with full chemical names and aliases in the structure provenance record. Identifiers must not contain `_`, which is the separator, or characters prohibited in Windows filenames. Use the same identifier consistently for the same chemical entity.

### Single-component and multi-component quantification

Retain the specified task names, and separately document quantitative targets and sample composition. Quantifying one analyte in a complex mixture does not imply that labels are available for every constituent. Describe target and non-target positions in `metadata.json`; do not invent zero concentrations for unmeasured constituents.

## 3. Group names and labels

The final group index is a separate underscore-delimited field:

```text
Classification:
Class_<bool1>_<bool2>_..._<boolN>_<index>

Quantification:
Quan_<value1>_<value2>_..._<valueN>_<index>
```

`N` equals `n_classes`. Label order must exactly follow `molecules`; do not reorder alphabetically or by concentration. A six-digit index unique within each task, such as `000001`, is recommended.

### Classification

Each Boolean label is `0` or `1`. For `molecules = DA_E_NE`:

- `Class_1_0_1_000001`: DA and NE are present; E is absent.
- `Class_0_0_0_000002`: a blank group.
- `Class_1_0_1_000003`: another group with the same composition as the first; keep them separate to preserve provenance.

These labels encode the final composition without requiring an eight-class classifier or three separate presence/absence models.

### Quantification

Write numerical labels with `.` as the decimal separator. Units belong in `content.csv`, not directory names. Scientific notation is allowed when necessary, but formatting should be consistent within an import.

For `molecules = DA_E_NE`, `quant_unit = uM`, and `quant_datatype = double`, `Quan_1.0_0.0_2.5_000001` represents 1.0, 0.0, and 2.5 micromolar, respectively. Distinguish stock-solution, post-mixing, and final measurement concentrations. Prefer the final sample concentration required by the prediction task, and preserve the conversion basis.

Integer attributes such as carbon-chain length remain quantitative targets. Use `quant_unit = carbons` and `quant_datatype = int`, and specify whether the label counts all molecular carbons or only a designated chain. An integer target is not automatically a classification label.

**Distinguish zero from unknown.** An absent component may have a zero concentration when absence is established. Missing labels, unmeasured targets, and inapplicable attributes must not be replaced with zero or inferred from spectral intensity. If retained, such positions may use `null`, with the reason documented in `metadata.json`. Training requires label-validity masks; `null` must not be converted directly to a number. If missing labels are unsupported, exclude affected groups from the corresponding quantitative training task until they can be handled correctly.

### Group identity and independence

A group represents an identifiable sample or a clearly defined set of repeated measurements from the source. Fifty replicate spectra from one concentration-combination sample may form one group. Independently prepared samples with identical concentrations must have different indices. Do not merge spectra solely because their labels match.

If a physical sample appears across instruments, acquisition rounds, or datasets, retain a shared sample identifier in `metadata.json`. Directory grouping does not always establish statistical independence. Related records from the same sample must not leak across training and test partitions. Document unrecoverable grouping information rather than inventing `n_group`.

## 4. Individual-spectrum CSV files

Save each spectrum as a separate UTF-8 CSV with exactly two columns and this case-sensitive header:

```csv
RamanShift,intensity
```

- Express Raman shift in inverse centimeters (`cm^-1`). Convert wavelength axes using a documented excitation wavelength; do not merely rename the column.
- Use numerical values, with one row per coordinate and matching coordinate/intensity lengths. Flag missing or infinite values rather than silently replacing them with zero.
- Ascending Raman-shift order is recommended. Reorder intensities together with coordinates and record the transformation.
- Format conversion must not automatically smooth, baseline-correct, normalize, or interpolate spectra. Preserve the original grid; subsequent algorithms select preprocessing and resampling.
- If only preprocessed spectra are available, preserve their values and identify their status. Do not describe them as instrument-raw data. Record derivation relationships between raw and processed versions; they are not independent samples.
- Do not automatically clip negative intensities; first check whether baseline correction produced them.
- Resolve duplicate coordinates, stitched scans, and multichannel data explicitly rather than silently averaging them.
- Do not duplicate average spectra to inflate `n_spectra`.

Preserve original downloads in their collection location. Map standardized spectra back to source files, worksheets, or columns through `metadata.json`.

This schema uses Raman shift. Other modalities, such as infrared spectroscopy, must not be imported as if they were Raman data without an explicit schema revision. Existing IR candidates remain in the separate candidate collection.

## 5. Molecular structure files

**Use SMILES with the `.smi` extension.** Each filename matches its identifier in `content.csv`, for example `DA.smi`. Each UTF-8 plain-text file contains one verified SMILES string on a single line, without a header or explanatory text. Do not insert unverified structures as database examples.

Use **canonical isomeric SMILES** generated by a consistent toolkit version where possible. Canonicalization standardizes serialization within the toolkit; isomeric SMILES retains known stereochemical information. See the [RDKit MolToSmiles documentation](https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html). Canonical strings depend on toolkit and version and are not universally identical across implementations.

SMILES can be parsed into an atom-and-bond graph for descriptors, fingerprints, or graph neural networks. Learned feature vectors are not required at collection time. Molecular graphs do not fully encode SERS adsorption orientation or reaction conditions; preserve these in experimental metadata.

### What a SMILES file records and how to use it

SMILES means *Simplified Molecular Input Line Entry System*. It represents a molecular structure as a line of plain text: atoms, bonds, branches and rings, plus charge, isotope and stereochemistry when specified. It is not a spectrum or a concentration label. For example, ethanol can be written as `CCO`; `C` denotes carbon and `O` oxygen, and ordinary attached hydrogens are generally implicit. Benzene can be written as `c1ccccc1`, where the matching digits indicate ring closure, not the number of carbon atoms. See the [Daylight specification](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html).

With [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html), a SMILES string can be parsed into a molecule, checked, drawn, and converted into numerical representations:

| Representation | Plain-language meaning | Possible use in this project |
| --- | --- | --- |
| Molecular descriptors | Named numerical properties such as molecular weight, ring count and hydrogen-bond donor count. | Describe chemical tasks with features that chemists can interpret. |
| Molecular fingerprint | A vector encoding local structural patterns. It is a structural summary, not a measured Raman fingerprint. | Compare structural similarity and describe how similar the target compounds in a task are. |
| Molecular graph | Atoms become nodes and bonds become edges. A graph neural network (GNN) learns numerical features from this graph. | Optional learned structure features when sufficient task/structure diversity is available; not required for using SMILES. |

Recommended starting point: retain verified structures, then use descriptors and fingerprints as transparent baseline features. With the current limited number of source families and verified molecular structures, do not assume that a newly trained GNN will generalize. Structure features supplement the spectra and experimental metadata; they do not uniquely determine SERS peak intensities or prove peak-to-structure assignments.

SMILES does not supply a measured 3D conformation, adsorption orientation, substrate surface state, pH, concentration or spectrum. Generating a 3D conformer from it is a separate modeling step. Protonation, tautomerism, derivatization and metal coordination can change the relevant chemical species and must be recorded when established. Different valid strings can represent the same molecule; compare parsed/standardized structures rather than raw text alone.

Structure handling rules:

1. Preserve the source, original SMILES, standardized SMILES, toolkit version, and transformation history.
2. Retain known stereochemistry. Do not invent unspecified stereochemistry or represent a racemic sample as one confirmed enantiomer.
3. Record whether the reagent is a parent compound, salt, solvate, or derivative. Counterion removal, charge changes, and tautomer standardization require explicit documented rules.
4. A dot (`.`) may separate disconnected fragments, such as counterions. It does not encode concentrations or mixture ratios. See the [Daylight theory manual](https://daylight.com/dayhtml/doc/theory/index.pdf).
5. Use distinct identifiers for different stereoisomers, derivatives, or chemical entities rather than overwriting structures under the same name.
6. Store derivative structures only when their identities are supported. Do not infer an unknown product from reagent names and record it as established fact.
7. Bacteria, minerals, complex polymers, and undefined mixtures may lack unique small-molecule SMILES. Mark structures as unavailable or not applicable. Do not invent structures or substitute a monomer for an entire material without an explicit representation policy. Missing SMILES does not by itself prevent other uses of the spectra.

The recommended `structure_sources.json` records full names, aliases, sources, structure status, stereochemical notes, and standardization rules. It does not add columns to `content.csv`.

## 6. Recommended task metadata

Keep the index compact. Store the following in `metadata.json` under each `Data_id`:

- Publication DOI, repository record, version, download date, and data license. Record software and experimental-data licenses separately.
- Original-to-standardized file mappings, conversion-script version, and checksums.
- Relationships between group indices, physical samples, independent preparations, substrate batches, dates, instruments, and technical repeats.
- Target definitions, unit conversions, label provenance, and distinctions among known zero, missing, and inapplicable values.
- Known excitation wavelength, power, integration time, substrate, functionalization, matrix, pH, derivatization, storage time, and other conditions. Explicitly mark unknown values.
- Raw or preprocessed status and prior processing. Original algorithms and code are reproduction references, not mandatory workflows.
- Shared samples and source studies across task identifiers.
- Supported uses, permissible splits, and limitations. For example, data without independent substrate batches cannot by themselves establish cross-batch generalization.

These records prevent sample leakage and confusion between source effects and chemical differences without prescribing an internal analysis workflow.

## 7. Ingestion checks

Before accepting a dataset, verify that:

- The index has ten correctly named columns and unique `Data_id` values.
- Each task matches an allowed value.
- Every modality is allowed and matches the corresponding dataset metadata; unknown or mixed categories include an explanation.
- The molecule-identifier count equals `n_classes`.
- Group labels have the correct count, order, and numerical type.
- Group indices are unique within the task.
- `n_group` matches the group-directory count.
- `n_spectra` matches the valid individual-spectrum CSV count.
- Spectral coordinates and intensities are readable numerical data.
- Structure availability is explicit.
- Provenance and sample-group relationships are traceable.

The original empty index was backed up to `../公开光谱任务库/backups/content_before_first_import_20261104.csv`. Formal ingestion removed blank rows and trailing empty columns; the original import contained nine fields. The modality update adds a tenth field after `task`; the previous index is backed up in `../公开光谱任务库/backups/content_before_modality_20261104.csv`.

## 8. Collection status and acquisition routes

Snapshot: November 4, 2026, after the second collection round. The index contains **10 task datasets**, **31,835 spectral CSVs** (20,451 SERS and 11,384 conventional Raman), from **7 source-study/data families**. Task IDs, spectra, map pixels and independent preparations are different units.

| Data_id | Dataset | Source | Modality | Groups | Spectra |
| --- | --- | --- | --- | ---: | ---: |
| GSAD0001 | PAH mixtures | PNAS | direct-SERS | 32 | 684 |
| GSAD0002 | Adenine interlaboratory | Analytical Chemistry | direct-SERS | 1744 | 3,516 |
| GSAD0003 | Four-sugar mixtures | Author repository | Raman | 245 | 9,800 |
| GSAD0004 | Pesticide mixtures, public subset | ACS Applied Materials & Interfaces | direct-SERS | 11 | 33 |
| GSAD0005 | Methanol in ethanol/water | Nature Communications | Raman | 9 | 72 |
| GSAD0006 | Serine calibration | Nature Communications | Raman | 6 | 16 |
| GSAD0007 | Glucose calibration | Nature Communications | Raman | 9 | 28 |
| GSAD0008 | Polymer single/multicomponent maps | Nature Communications | direct-SERS | 51 | 14,811 |
| GSAD0009 | Polymer reference maps | Nature Communications | Raman | 12 | 1,468 |
| GSAD0010 | CV source-labelled concentration/background | Nature Machine Intelligence | direct-SERS | 7 | 1,407 |

The new six task records add 17,802 spectra. Three calibration tasks share one Nature Communications instrument study. The two polymer tasks share another Nature Communications study. The CV task is derived from one public validation dataset cited by the Nature Machine Intelligence DSCF paper. These are three additional source families, not six independent publications.

Fourteen molecular SMILES files have PubChem provenance. Five polymers have explicit `not_applicable_unique_small_molecule` records; CV has an `unavailable` identity/structure record. No monomer is substituted for a polymer and no guessed expansion of CV is linked to a chemical graph.

### Eligibility limits that affect training


- **GSAD0001:** two incomplete intensity series were excluded in full, with source positions recorded in metadata. The remaining tables support component-presence labels. Absolute quantitative labels and whether the deposited intensities are detector-native raw data have not been verified.
- **GSAD0002:** the 3,516 original TXT files match the author label table. However, concentrations for `cAg@785` and `sAg@532` disagree between the CSV and SI Table S1. SI Table S8 is consistent with the CSV grouping, so the source itself needs clarification. All original labels are retained. **Exclude the 1,401 spectra in these two methods from quantitative training until resolved.** Per-group `label_review_status` identifies them. The other 2,115 spectra are consistent with the checked SI table; this is a source-consistency check, not independent validation of concentration accuracy. Two byte-identical spectral clusters also span nominal batch groups; their four records must remain in the same validation fold (see `identical_spectrum_clusters` in metadata).
- **GSAD0003:** concentration labels use the author-declared nominal 1 M stocks and recorded mixing volumes. All 40 measurements from a well share one group, including high/low integration times. Sample age also differs between conditions. This dataset is not SERS.
- **GSAD0004:** the downloaded repository contains only 33 spectra from two pesticide pairs and a control. Per-component ppm labels reproduce the author's `Dataset.py` convention: directory concentration multiplied by 0.5 for a 50:50 mixture. The ppm mass basis, preparation independence and pre-deposition processing are unresolved. This is a small pilot task, not the entire paper dataset.


- **GSAD0005:** 72 measured spectra at nine methanol volume fractions, under two source-sheet acquisition configurations. Preserve the different grids and keep repeated concentration conditions together. Independent preparation IDs are unavailable.
- **GSAD0006:** 16 source CSV measurements at six serine concentration conditions. Concentrations are specified in g/L by source filenames; stereochemistry is not established, so the reference SMILES is connectivity-only.
- **GSAD0007:** 28 source CSV measurements at nine glucose concentration conditions. Original percentage labels are retained. The percentage basis is not verified; do not automatically convert to g/L or molarity or compare absolute errors across units.
- **GSAD0008:** 14,811 native JWS spectra from 51 source map/sample folders. Labels identify five polymers at the sample level; a pixel may contain no detectable target. Avoid random pixel splitting. Original SPC and environmental/concentration branches are retained outside the standardized subset.
- **GSAD0009:** 1,468 native JWS conventional Raman reference spectra in 12 folders. Same source family/materials as GSAD0008; not an independent external study.
- **GSAD0010:** 1,407 author-exported nobase TXT spectra from seven maps. Filenames specify CV at 1e-9 M, 1e-10 M or blank. CV's full identity and any preparation dilution require clarification. There are only two nonzero concentration levels. This is a small nominal-label/background challenge, not a broad validated quantitative curve.


Exact-duplicate audit: the 31,835 CSV records contain 30,414 distinct standardized CSV byte hashes. Original records and aliases are retained. GSAD0008 has 1,228 repeated exports beyond its 13,583 distinct hashes; 504 duplicate clusters span source folders. Its 51 folders are linked into 49 duplicate-connected split blocks, recorded per group in metadata. GSAD0009 has 189 repeated exports beyond 1,279 distinct hashes. No conflicting-label duplicate clusters were found within either new polymer task. Train/test separation must respect these links. Hash uniqueness does not establish sample independence.

### Native-file decoding and validation

Native JASCO files are OLE containers. Their `DataInfo` stream supplies the number of points, first/last spectral coordinates and coordinate increment; `Y-Data` contains float32 intensities. The original authors' `jwslib.py` and conversion code were inspected to establish this mapping. The importer verifies header lengths, endpoints and finite intensities, then writes each spectrum using its original calibrated grid. It does not run author baseline correction, normalization, spike removal, augmentation or model training. Source headers remain traceable in the per-spectrum manifest.

For the CV TXT files, the first row supplies the actual Raman coordinates. The first column of body rows supplies map positions and is retained as metadata. It must not be mistaken for the Raman coordinate axis.

`validation_report.json` records numerical/structural checks. Passing those checks does not resolve chemical-identity, percentage-basis, independence or processing-history limitations. Original downloads remain unchanged under `../公开光谱任务库/downloads/` and every standardized spectrum has a source mapping and checksum.

### High-impact sources already downloaded but not counted in the index

- **Nature Communications 2025 receptor recommendation:** both complete data ZIPs and code are downloaded and repository-MD5 verified. The paper describes map averaging and preprocessing. Main matrices have 704 numbered features without explicit physical axes, inconsistent with a literal application of the paper/code descriptions. Mixture files contain coordinates but composition units still need the supplementary table. No guessed axis or concentration is inserted. This is a highly relevant processed/averaged auxiliary source for structural analogues, not a confirmed detector-native raw collection. [Paper](https://doi.org/10.1038/s41467-025-62519-x), [data](https://zenodo.org/records/15830084).
- **Nature Machine Intelligence 2025 DSCF prepared-solution quantification:** 35 PT records have been downloaded and MD5 checked. Mixture identifiers alone do not specify concentrations or complete chemical identity. Verify axes and the SI composition table before conversion. The phrase “synthetic solution” refers to a prepared-solution source here and must not automatically be interpreted as computer-generated spectra. [Data](https://figshare.com/articles/dataset/28107281).
- **Nature Communications 2026 precision culturomics:** two source archives are downloaded, including 300 raw bacterial-cell example spectra. Fitted molecular coefficients are not independent chemical concentration ground truth; cell conditions are not molecule-presence vectors. The eight known amino-acid mixture benchmark spectra/labels still need to be located or requested. [Paper](https://doi.org/10.1038/s41467-026-74582-z).
- **Nature Communications 2024 lipid-AuNP**, **Nature Biomedical Engineering 2025 pH/PSA navigation**, and **Nature Communications 2026 swarming probes:** figure-source tables/arrays have been downloaded (the last source partially). Plotted mean curves, derived images and simulation curves are not automatically counted as original individual experimental spectra. See the acquisition queue for exact links and gaps.

### Download/access work that remains

`acquisition_status.csv` is the single acquisition queue. It distinguishes standardized, downloaded-but-unresolved, partially downloaded and author-request sources. No author requests or emails have been sent.

1. **Cambridge seven-molecule low-frequency SERS, Nature Communications 2024:** the public 702 MB original archive is identified, but several HTTP/SSL transfer attempts failed. Partial chunks are preserved for resuming; no incomplete archive is counted. [Repository](https://doi.org/10.17863/CAM.110008).
2. **K18 calibration, Nature Communications 2025:** original CC-BY-4.0 calibration ZIPs and README are publicly listed, but automated file downloads returned HTTP 403. Open the [university repository](https://pureportal.strath.ac.uk/en/datasets/data-for-a-point-of-care-diagnostic-for-drug-induced-liver-injury/) in a browser and download calibration files A and B plus `Raw_SERS_Readme.txt`. Patient data are not needed to start the calibration task.
3. **JACS 2025 chemical space of 38 molecules:** exceptionally relevant to functional-group/carbon-chain generalization, but no full machine-readable raw archive was verified. Obtain individual spectra and structure/label/preparation tables from the authors. [Paper](https://doi.org/10.1021/jacs.4c15804).
4. **Ling cerebroside taxonomy (Nature Communications 2024), pure-to-mixture transfer (Angew 2025), and structural-analogue recognition (Angew 2023):** keep these as priority raw-data requests. Code or spectral plots alone do not establish access to the required per-measurement data.
5. **Nature 2024 digital colloid-enhanced Raman counting:** publisher Source data are linked, while other relevant data are by request. Ask for complete event spectra and concentration standards rather than treating figure points as spectra. [Paper](https://doi.org/10.1038/s41586-024-07218-1).

An effective request asks for individual spectra with coordinates, per-spectrum labels/units, independent preparation/sample IDs, technical-repeat/map and substrate-batch IDs, instrument/matrix/derivatization conditions, prior processing, original split membership and reuse terms.

### What this collection can support

The collection now supports substantially broader reader, grouping, preprocessing and candidate-workflow experiments, with molecular quantification and polymer mixture recognition. It still contains few independent source families and does not establish that a universal workflow-design model can be trained. Additional small-molecule structural-analogue and genuine multicomponent quantitative datasets remain the most valuable additions. More map pixels cannot substitute for more independent chemistry tasks.
