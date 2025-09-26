# CONNExIN TEAM
A repository of the CONNExIN Functional MRI Team (0S)

## WEEK ONE ASSIGNMENT
### TEAM DETAILS 
**Team Lead:**
- **Name:** Maxwell Adu
- **Affiliation:** Komfo Anokye Teaching Hospital, Kumasi-Ghana 

### Team Members 
1.
   - **Name:** Abdul Rashid Karim
   - **Affiliation:** Spectra Health Interventional And Imaging Center, kumas-Ghana
2.
   - **Name:** Ireneaus Nyame 
   - **Affiliation:** University of Cape Coast, Cape Coast-Ghana
3.
   - **Name:** Djinkou Noukeu Frank Barthes
   - **Affiliation:** Faculty of Medicine and Biomedical Sciences, University of Yaoude 1, Yaounde-Cameroon 
4.
   - **Name:** Meram Mahmoud Elyan 
   - **Affiliation:** Systems and Biomedical Engineering, Cairo University-Egypt
5.
   - **Name:** Debborah Chepkurui
   - **Affiliation:** Kenyatta University Teaching Referral and Research Hospital, Nairobi-Kenya 



# Scripts Bash
# Script 1 — DICOM ➜ BIDS + validation (Neurodesk)
#!/bin/bash
set -euo pipefail

## ── 0) Project layout (single consistent root) ────────────────────────────────
export PROJECT=/neurodesktop-storage/Functional_MRI_OS
export BIDS_DIR=${PROJECT}/bids
export SRC_DICOM=${PROJECT}/sourcedata/dicom
export DERIV=${PROJECT}/derivatives
mkdir -p "${BIDS_DIR}" "${SRC_DICOM}" "${DERIV}" "${PROJECT}/code"

### Example: we’ll process sub-002 ses-01
SUBS=("sub-002")
SESS=("ses-01")

## ── 1) dcm2niix ───────────────────────────────────────────────────────────────
echo "[info] Loading dcm2niix…"
ml dcm2niix
dcm2niix -h >/dev/null

### Convert per subject/session into a temp dropbox, then move into BIDS layout.
for s in "${SUBS[@]}"; do
  for se in "${SESS[@]}"; do
    in_dir="${SRC_DICOM}/${s}/${se}"
    [ -d "${in_dir}" ] || { echo "[warn] Missing ${in_dir} – skipping."; continue; }

    tmp="${PROJECT}/tmp_${s}_${se}"
    rm -rf "${tmp}"; mkdir -p "${tmp}"

    echo "[info] dcm2niix: ${s} ${se}"
    # -b y: sidecars; -z y: gzip; -ba y: anonymize basic; -f: include sub/ses/protocol/run in file name
    dcm2niix -b y -z y -ba y -o "${tmp}" -f "${s}_${se}_%p_run-%2d" "${in_dir}"

    # ── 2) Move into BIDS folders (simple rules; edit the patterns to match your scanner names)
    dest_anat="${BIDS_DIR}/${s}/${se}/anat"
    dest_func="${BIDS_DIR}/${s}/${se}/func"
    mkdir -p "${dest_anat}" "${dest_func}"

    # Heuristics: adjust to your protocol names if needed (e.g., MPRAGE, T1-MPRAGE, REST, rs-fMRI, BOLD)
    shopt -s nocaseglob
    for f in "${tmp}"/*; do
      base=$(basename "$f")
      case "$base" in
        *T1*|*MPRAGE*)
          # Rename to BIDS: sub-XXX_ses-YY_T1w.{nii.gz,json}
          ext="${base##*.}"; stem="${base%.*}"
          # handle .nii.gz vs .json pairs
          if [[ "$base" == *.nii.gz ]]; then
            mv "$f" "${dest_anat}/${s}_${se}_T1w.nii.gz"
          elif [[ "$ext" == "json" ]]; then
            mv "$f" "${dest_anat}/${s}_${se}_T1w.json"
          fi
          ;;
        *REST*|*rs-fMRI*|*BOLD*|*rest*)
          # Rename to BIDS task-rest
          if [[ "$base" == *.nii.gz ]]; then
            mv "$f" "${dest_func}/${s}_${se}_task-rest_bold.nii.gz"
          elif [[ "$base" == *.json" ]]; then
            mv "$f" "${dest_func}/${s}_${se}_task-rest_bold.json"
          fi
          ;;
        *)
          # leave other series for later mapping (fieldmaps, FLAIR, etc.)
          ;;
      esac
    done
    shopt -u nocaseglob
    rm -rf "${tmp}"
  done
done

## ── 3) Minimal BIDS side files ────────────────────────────────────────────────
if [ ! -f "${BIDS_DIR}/dataset_description.json" ]; then
  cat > "${BIDS_DIR}/dataset_description.json" <<'JSON'
{
  "Name": "Functional MRI Dataset",
  "BIDSVersion": "1.8.0",
  "DatasetType": "raw",
  "Authors": ["Your Name", "Collaborator Name"]
}
JSON
fi

### optional participants.tsv (edit as needed)
if [ ! -f "${BIDS_DIR}/participants.tsv" ]; then
  printf "participant_id\tage\tsex\n" > "${BIDS_DIR}/participants.tsv"
  for s in "${SUBS[@]}"; do printf "%s\tNA\tNA\n" "${s}" >> "${BIDS_DIR}/participants.tsv"; done
fi

## ── 4) Validate BIDS ──────────────────────────────────────────────────────────
echo "[info] Validating with bids-validator module…"
if command -v bids-validator >/dev/null 2>&1; then
  bids-validator "${BIDS_DIR}" --ignoreWarnings --verbose
else
  ### Neurodesk has a bids-validator module; load it if not already in PATH
  ml bids-validator
  bids-validator "${BIDS_DIR}" --ignoreWarnings --verbose
fi

echo "[done] BIDS ready at ${BIDS_DIR}"
tree -L 3 "${BIDS_DIR}" || ls -R "${BIDS_DIR}"


# Script 2 — MRIQC (participant then group)
#!/bin/bash
set -euo pipefail

export PROJECT=/neurodesktop-storage/Functional_MRI_OS
export BIDS_DIR=${PROJECT}/bids
export MRIQC_OUT=${PROJECT}/derivatives/mriqc
mkdir -p "${MRIQC_OUT}"

ml mriqc
mriqc --version

### Edit labels as needed
PARTS=("sub-002")

echo "[info] MRIQC participant level…"
mriqc "${BIDS_DIR}" "${MRIQC_OUT}" participant \
  --participant-label "${PARTS[@]}" \
  --n_procs 8 --omp-nthreads 4 --mem_gb 16 \
  --fd_thres 0.3 \
  --verbose-reports

echo "[info] MRIQC group level…"
mriqc "${BIDS_DIR}" "${MRIQC_OUT}" group \
  --n_procs 8 --omp-nthreads 4 --mem_gb 16

echo "[info] Serving reports on http://localhost:8000"
cd "${MRIQC_OUT}"
python3 -m http.server 8000 &


# Script 3 — fMRIPrep (Apptainer, consistent paths)
#!/bin/bash
set -euo pipefail

### ── 0) Paths & resources ──────────────────────────────────────────────────────
export PROJECT=/neurodesktop-storage/Functional_MRI_OS
export BIDS_DIR=${PROJECT}/bids
export DERIV=${PROJECT}/derivatives
export OUT_DIR=${DERIV}/fmriprep
export WORK_DIR=${PROJECT}/work
export FS_LICENSE=/neurodesktop-storage/licenses/freesurfer/license.txt   export FS_SUBJECTS_DIR=${DERIV}/freesurfer
mkdir -p "${OUT_DIR}" "${WORK_DIR}" "${FS_SUBJECTS_DIR}"

[ -d "${BIDS_DIR}" ] || { echo "[error] ${BIDS_DIR} not found."; exit 1; }
[ -f "${FS_LICENSE}" ] || { echo "[error] FreeSurfer license missing at ${FS_LICENSE}"; exit 1; }

### Optional: keep Apptainer caches on the big disk
export APPTAINER_CACHEDIR=${PROJECT}/.apptainer_cache
export APPTAINER_TMPDIR=${PROJECT}/.apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

### ── 1) Choose subjects ────────────────────────────────────────────────────────
PARTS=("sub-002")  # match your BIDS labels exactly (e.g., sub-002)

### ── 2) Run fMRIPrep (latest image from Docker Hub) ────────────────────────────
echo "[info] Starting fMRIPrep…"
apptainer run --cleanenv \
  -B "${BIDS_DIR}":/data:ro \
  -B "${OUT_DIR}":/out \
  -B "${WORK_DIR}":/work \
  -B "${FS_LICENSE}":/opt/freesurfer/license.txt \
  -B "${FS_SUBJECTS_DIR}":/fs-subjects \
  docker://nipreps/fmriprep:latest \
  /data /out participant \
  --participant-label "${PARTS[@]}" \
  --fs-license-file /opt/freesurfer/license.txt \
  --fs-subjects-dir /fs-subjects \
  --work-dir /work \
  --nthreads 8 --omp-nthreads 4 --mem 32GB \
  --use-syn-sdc \
  --dummy-scans 0 \
  --output-spaces MNI152NLin2009cAsym:res-2 anat fsnative fsaverage \
  --cifti-output  \
  --use-aroma

echo "[done] fMRIPrep finished. Outputs -> ${OUT_DIR}"

echo "[info] Serving HTML reports on http://localhost:8000"
cd "${OUT_DIR}"
python3 -m http.server 8000 &


# Scritp 4- Analysis 

## Analysis: longitudinal measurement of local spontaneous brain activity (ALFF, fALFF, ReHo) across sessions

#!/bin/bash
set -euo pipefail

######################################################################
CONFIG — edit these to your dataset
######################################################################
PROJECT=/neurodesktop-storage/Functional_MRI_OS
DERIV=${PROJECT}/derivatives
FMRIPREP=${DERIV}/fmriprep                    # fMRIPrep output root
OUT_METRICS=${DERIV}/localfluct               # where we write ALFF/fALFF/ReHo
OUT_ROI=${DERIV}/localfluct_roi               # ROI CSV lives here
WORK=${PROJECT}/work                          # scratch

### Subjects and sessions to process (space-separated)
SUBJECTS=("sub-002")                          # e.g., sub-002 sub-003 ...
SESSIONS=("ses-01" "ses-02" "ses-03")         # e.g., ses-01 ses-02 ses-03

### Atlas for ROI extraction (FSL Harvard-Oxford cortical maxprob, 2mm)
ATLAS=${FSLDIR}/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz
ATLAS_NAME="Harvard-Oxford Cortical Structural Atlas"   # for atlasquery

### Denoising/band-pass settings
FD_CENSOR=0.5        # scrub if FD > 0.5 mm
BP_LO=0.01           # band-pass low (Hz)
BP_HI=0.08           # band-pass high (Hz)

### ReHo neighborhood (27 = 3x3x3 minus center)
REHO_N=27

### Resources (adjust to your Neurodesk VM)
NPROCS=8
OMP_NTHREADS=4
MEM_GB=12

### Export threads for AFNI/OpenMP tools
export OMP_NUM_THREADS=${OMP_NTHREADS}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${OMP_NTHREADS}
export AFNI_NIFTI_TYPE_WARN=NO

######################################################################Checks for tools (AFNI & FSL commands)
######################################################################
need() { command -v "$1" >/dev/null 2>&1 || { echo "[error] Missing '$1' in PATH"; exit 1; }; }
need 3dTproject
need 3dReHo
need fslstats
need flirt
need python
need atlasquery

RSFC_AVAILABLE=0
if command -v 3dRSFC >/dev/null 2>&1; then RSFC_AVAILABLE=1; fi

mkdir -p "${OUT_METRICS}" "${OUT_ROI}" "${WORK}"

######################################################################
ROI name lookup (ID -> Name) for Harvard-Oxford Cortical atlas
NOTE: In the maxprob label image, IDs start at 1 and follow this order.
######################################################################
HO_MAP=${OUT_ROI}/harvardoxford_cort_id2name.tsv
if [ ! -s "${HO_MAP}" ]; then
  ### Create a TSV with: id<tab>roi_name
  atlasquery -a "${ATLAS_NAME}" -l \
    | nl -ba \
    | sed 's/^\s*//' \
    | awk '{id=$1; $1=""; sub(/^ /,""); print id "\t" $0}' > "${HO_MAP}" || true
fi

######################################################################
Final CSV header
######################################################################
CSV=${OUT_ROI}/roi_means.tsv
echo -e "subject\tsession\trun\ttr_sec\tmeanFD\tmetric\troi_id\troi_name\tmean\tvoxels" > "${CSV}"

######################################################################
Iterate subjects/sessions/runs
######################################################################
for SUB in "${SUBJECTS[@]}"; do
  for SES in "${SESSIONS[@]}"; do
    FUNC_DIR=${FMRIPREP}/${SUB}/${SES}/func
    if [ ! -d "${FUNC_DIR}" ]; then
      echo "[warn] ${FUNC_DIR} not found; skipping ${SUB} ${SES}"
      continue
    fi

    mapfile -t RUNS < <(ls ${FUNC_DIR}/${SUB}_${SES}_task-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz 2>/dev/null || true)
    if [ ${#RUNS[@]} -eq 0 ]; then
      echo "[warn] No runs found for ${SUB} ${SES}; skipping"
      continue
    fi

    for BOLD in "${RUNS[@]}"; do
      BASE=$(basename "${BOLD}")
      RUN=$(echo "${BASE}" | sed -n 's/.*_task-[^_]\+_\(run-[^_]\+\).*/\1/p'); RUN=${RUN:-run-01}
      MASK=${BOLD/_desc-preproc_bold/_desc-brain_mask}
      CONF=${BOLD/_space-*/}_desc-confounds_timeseries.tsv

      OUT_DIR=${OUT_METRICS}/${SUB}/${SES}/${RUN}
      WRK_DIR=${WORK}/${SUB}/${SES}/${RUN}
      mkdir -p "${OUT_DIR}" "${WRK_DIR}"

      echo "[info] ${SUB} ${SES} ${RUN} :: denoise + band-pass"

      #1) Confounds (12 motion + top 5 aCompCor)
      python - "${CONF}" "${WRK_DIR}/confounds.1D" <<'PY'
import sys, pandas as pd, numpy as np
tsv, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(tsv, sep='\t')
base = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z',
        'trans_x_derivative1','trans_y_derivative1','trans_z_derivative1',
        'rot_x_derivative1','rot_y_derivative1','rot_z_derivative1']
acc = [c for c in df.columns if c.startswith('a_comp_cor_')][:5]
cols = [c for c in base if c in df.columns] + acc
X = df[cols].fillna(0.0).values if cols else np.zeros((len(df),1))
np.savetxt(out, X, fmt='%.6f')
PY

      #2) Censor vector (drop first 2 TRs; censor FD > threshold)
      python - "${CONF}" "${WRK_DIR}/censor.1D" "${FD_CENSOR}" <<'PY'
import sys, pandas as pd, numpy as np
tsv, out, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
df = pd.read_csv(tsv, sep='\t')
fd = df.get('framewise_displacement', pd.Series([0]*len(df))).fillna(0).values
keep = (fd <= thr).astype(int); keep[:2] = 0
np.savetxt(out, keep[np.newaxis,:], fmt='%d')
PY

      # 3) Denoise + band-pass
      3dTproject -input "${BOLD}" -mask "${MASK}" \
                 -ort "${WRK_DIR}/confounds.1D" \
                 -censor "${WRK_DIR}/censor.1D" \
                 -passband ${BP_LO} ${BP_HI} \
                 -polort 2 \
                 -prefix "${OUT_DIR}/bold_clean.nii.gz" >/dev/null

      # TR & mean FD
      TR=$(3dinfo -tr "${OUT_DIR}/bold_clean.nii.gz" | tr -d '[:space:]')
      MEANFD=$(python - "${CONF}" <<'PY'
import sys,pandas as pd, numpy as np
df=pd.read_csv(sys.argv[1], sep='\t'); x=df.get('framewise_displacement')
print(float(np.nanmean(x)) if x is not None else 0.0)
PY
)

      # 4) ReHo
      echo "[info] ${SUB} ${SES} ${RUN} :: ReHo"
      3dReHo -inset "${OUT_DIR}/bold_clean.nii.gz" -mask "${MASK}" -nneigh ${REHO_N} \
             -prefix "${OUT_DIR}/reho.nii.gz" >/dev/null

      # 5) ALFF / fALFF
      if [ ${RSFC_AVAILABLE} -eq 1 ]; then
        echo "[info] ${SUB} ${SES} ${RUN} :: ALFF/fALFF via 3dRSFC"
        3dRSFC -input "${OUT_DIR}/bold_clean.nii.gz" -mask "${MASK}" \
               -band ${BP_LO} ${BP_HI} -alff -f_alff -prefix "${OUT_DIR}/alff" >/dev/null
        mv "${OUT_DIR}/alff_ALFF.nii.gz"  "${OUT_DIR}/alff.nii.gz"
        mv "${OUT_DIR}/alff_fALFF.nii.gz" "${OUT_DIR}/falff.nii.gz"
      else
        echo "[info] ${SUB} ${SES} ${RUN} :: ALFF/fALFF via Python fallback"
        python - <<PY
import numpy as np, nibabel as nib
from numpy.fft import rfft, rfftfreq
bold_path=r"${OUT_DIR}/bold_clean.nii.gz"; mask_path=r"${MASK}"
img=nib.load(bold_path); data=img.get_fdata(); mask=nib.load(mask_path).get_fdata().astype(bool)
TR=img.header.get_zooms()[3]; T=data.shape[3]
freqs=rfftfreq(T,d=TR); lo,hi=${BP_LO},${BP_HI}
band=(freqs>=lo)&(freqs<=hi); total=freqs>0
alff=np.zeros(data.shape[:3],np.float32); falff=np.zeros_like(alff)
ts=data[mask]; ts=ts-ts.mean(axis=1,keepdims=True); spec=np.abs(rfft(ts,axis=1))**2
bandpow=spec[:,band].sum(axis=1); totpow=spec[:,total].sum(axis=1)+1e-8
alff_vals=np.sqrt(bandpow); falff_vals=bandpow/totpow
alff[mask]=alff_vals; falff[mask]=falff_vals
nib.save(nib.Nifti1Image(alff,img.affine,img.header), r"${OUT_DIR}/alff.nii.gz")
nib.save(nib.Nifti1Image(falff,img.affine,img.header), r"${OUT_DIR}/falff.nii.gz")
PY
      fi

      # 6) ROI extraction (resample atlas → label list → means → CSV with names)
      flirt -in "${ATLAS}" -ref "${OUT_DIR}/alff.nii.gz" \
            -applyxfm -usesqform -interp nearestneighbour \
            -out "${OUT_DIR}/atlas_resamp.nii.gz"

      python - "${OUT_DIR}/atlas_resamp.nii.gz" "${OUT_DIR}/labels.txt" <<'PY'
import sys, nibabel as nib, numpy as np
img=nib.load(sys.argv[1]); labs=np.unique(img.get_fdata().astype(int)); labs=labs[labs>0]
open(sys.argv[2],'w').write("\n".join(str(int(x)) for x in labs))
PY

      # Helper to append stats per metric (adds roi_name via HO_MAP)
      append_metric () {
        local METRIC="$1"
        local MAP="${OUT_DIR}/${METRIC}.nii.gz"
        [ -f "${MAP}" ] || return 0

        local MEANS COUNTS
        MEANS=$(fslstats "${MAP}" -K "${OUT_DIR}/atlas_resamp.nii.gz" "${OUT_DIR}/labels.txt" -M)
        COUNTS=$(fslstats "${OUT_DIR}/atlas_resamp.nii.gz" -K "${OUT_DIR}/atlas_resamp.nii.gz" "${OUT_DIR}/labels.txt" -V | awk '{print $1}')

        local i=0
        while read -r L; do
          i=$((i+1))
          local M VX NAME
          M=$(echo "${MEANS}"  | awk -v n=${i} '{print $n}')
          VX=$(echo "${COUNTS}"| awk -v n=${i} '{print $n}')
          NAME=$(awk -F'\t' -v id="${L}" '($1==id){print $2}' "${HO_MAP}")
          [ -z "${NAME}" ] && NAME="UNKNOWN_${L}"
          echo -e "${SUB}\t${SES}\t${RUN}\t${TR}\t${MEANFD}\t${METRIC}\t${L}\t${NAME}\t${M}\t${VX}" >> "${CSV}"
        done < "${OUT_DIR}/labels.txt"
      }

      append_metric "alff"
      append_metric "falff"
      append_metric "reho"

      echo "[done] ${SUB} ${SES} ${RUN}"
    done
  done
done

echo "[✓] ROI table written: ${CSV}"
echo "Columns: subject, session, run, tr_sec, meanFD, metric, roi_id, roi_name, mean, voxels"
echo "HO lookup: ${HO_MAP}"


