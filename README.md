# XTTS-RVC-UI

This is a Fork of XTTS-RVC-UI that adds realtime typing, updates voice playback to happen right away if any changes are made in the interface, and adds additional temperature and repetition penality sliders to adjust your voice. Made it autoplay only the RVC output.

Note: You can also separately adjust the xtts model's config.json top_k and top_p settings for further tweaking before starting the start.bat. Here is what I am using for that (experimental):

"top_k": 70,
"top_p": 0.95,

Note2: When you finish typing if it didn't read the entire thing, you can simply press . or spacebar or backspace and it will read the entire sentence during a refresh (usually within 1 or 2 seconds). I have found the best results by using Dragon Naturally Speaking and my microphone. Having it type in the box for me and using a "custom dragon command" word "erase" to erase the box. My dragon step-by-step command is like this, Steps: Control + A, Backspace" when myCommand "erase" is spoken.

Original Repo Info:

This is a simple UI that utilize's [Coqui's XTTSv2](https://github.com/coqui-ai/TTS) paired with RVC functionality to improve output quality.

# Prerequisites

- Requires MSVC - VC 2022 C++ x64/x86 build tools.

# Usage

Clone this repository:

```
git clone https://github.com/Vali-98/XTTS-RVC-UI.git
```

It is recommended to create a venv.

Then, install the requirements:

```
pip install -r requirements.txt
```

If you have a CUDA device available, it is also recommended to install PyTorch with CUDA for faster conversions.

```
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

Then run `start.bat` , `start.sh` or simply `python app.py`

This will create the following folders within the project:

```
\models\xtts
\rvcs
\voices
```
- Relevant models will be downloaded into `\models`. This will be approximately ~2.27GB.
- You can manually add the desired XTTSv2 model files in `\models\xtts`.
- Place RVC models in `\rvcs`. Rename them as needed. If an **identically named** .index file exists in `\rvcs`, it will also be used.
- Place voice samples in `\voices`

