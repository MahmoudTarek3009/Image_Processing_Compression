# B14 - Robustness of Learned Compression to Corrupted/Adversarial Inputs

CSE 681 Digital Image Processing team project (catalog project 14).
Stress-tests a pretrained learned image codec (CompressAI) against image corruptions and small
adversarial perturbations, and compares its failure modes with JPEG / JPEG2000.

Team: TODO Student A, TODO Student B, TODO Student C

## Setup

Python 3.12 is required in practice: CompressAI has no prebuilt Windows wheels for 3.13.
Create the environment outside any OneDrive-synced folder.

```powershell
$env:UV_LINK_MODE = "copy"
uv venv C:\venvs\b14 --python 3.12
uv pip install --python C:\venvs\b14\Scripts\python.exe -r requirements.txt
```

## Data

Kodak (24 images, 768x512, free for research use):

```powershell
New-Item -ItemType Directory -Force data\kodak | Out-Null
1..24 | ForEach-Object {
    $n = "kodim{0:D2}.png" -f $_
    Invoke-WebRequest "https://r0k.us/graphics/kodak/kodak/$n" -OutFile "data\kodak\$n"
}
```

## Run the M1 demo

```powershell
C:\venvs\b14\Scripts\python.exe src\demo_corruptions.py
```

Prints bitrate (bpp), PSNR against the corrupted input, and the rate ratio versus the clean
image for the learned codec, JPEG and JPEG2000. Image, corruptions and severities are set at
the top of the script. Pretrained CompressAI weights are downloaded on first run.

## Reproducibility

- Fixed seed: `42` (`src/seed.py`). Call `set_seed()` at the start of every script and before
  every corruption call.
- Dependencies are listed in `requirements.txt`; pin exact versions with `uv pip freeze` once
  the environment works.
- `data/` is not versioned; use the download command above.

## AI-tool disclosure

TODO: one line stating where AI assistance was used (e.g. code scaffolding) and that all
results were run and verified by the team.
