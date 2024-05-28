import torch
from TTS.api import TTS
import gradio as gr
from rvc import Config, load_hubert, get_vc, rvc_infer
import gc, os, sys, argparse, requests
from pathlib import Path

parser = argparse.ArgumentParser(
    prog='XTTS-RVC-UI',
    description='Gradio UI for XTTSv2 and RVC'
)

parser.add_argument('-s', '--silent', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

if args.silent:
    print('Enabling silent mode.')
    sys.stdout = open(os.devnull, 'w')

def download_models():
    rvc_files = ['hubert_base.pt', 'rmvpe.pt']

    for file in rvc_files:
        if not os.path.isfile(f'./models/{file}'):
            print(f'Downloading {file}')
            r = requests.get(f'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/{file}')
            with open(f'./models/{file}', 'wb') as f:
                f.write(r.content)

    xtts_files = ['vocab.json', 'config.json', 'dvae.path', 'mel_stats.pth', 'model.pth']

    for file in xtts_files:
        if not os.path.isfile(f'./models/xtts/{file}'):
            print(f'Downloading {file}')
            r = requests.get(f'https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/{file}')
            with open(f'./models/xtts/{file}', 'wb') as f:
                f.write(r.content)

[Path(_dir).mkdir(parents=True, exist_ok=True) for _dir in ['./models/xtts', './voices', './rvcs']]

download_models()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device: " + device)

config = Config(device, device != 'cpu')
hubert_model = load_hubert(device, config.is_half, "./models/hubert_base.pt")
tts = TTS(model_path="./models/xtts", config_path='./models/xtts/config.json').to(device)
voices = []
rvcs = []
langs = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]

def get_rvc_voices():
    global voices
    voices = os.listdir("./voices")
    global rvcs
    rvcs = list(filter(lambda x: x.endswith(".pth"), os.listdir("./rvcs")))
    return [rvcs, voices]

def runtts(rvc, voice, text, pitch_change, index_rate, temperature, repetition_penalty, language):
    try:
        if not text.strip():
            raise ValueError("Text input is required for synthesis.")
        
        # Ensure the TTS function uses the temperature and repetition penalty parameters
        audio = tts.tts_to_file(
            text=text,
            speaker_wav="./voices/" + voice,
            language=language,
            file_path="./output.wav",
            temperature=temperature,  # Add temperature here
            repetition_penalty=repetition_penalty  # Add repetition penalty here
        )
        
        voice_change(rvc, pitch_change, index_rate)
        return ["./output.wav", "./outputrvc.wav"]
    except Exception as e:
        print(f"Error in runtts: {e}")
        return [None, None]

def main():
    get_rvc_voices()
    print(rvcs)
    print(voices)
    interface = gr.Interface(
        fn=runtts,
        inputs=[
            gr.Dropdown(choices=rvcs, value=rvcs[0] if len(rvcs) > 0 else '', label='RVC model'),
            gr.Dropdown(choices=voices, value=voices[0] if len(voices) > 0 else '', label='Voice sample'),
            gr.Textbox(placeholder="Write here...", label='Text', elem_id="text_input"),
            gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="Pitch"),
            gr.Slider(minimum=0, maximum=1, value=0.75, step=0.05, label="Index Rate"),
            gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.001, label="Temperature"),
            gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.001, label="Repetition Penalty"),
            gr.Dropdown(choices=langs, value=langs[0], label='Language')
        ],
        outputs=[
            gr.Audio(label="TTS result", type="filepath", interactive=False),
            gr.Audio(label="RVC result", type="filepath", interactive=False, autoplay=True)
        ],
        live=True,
        title="XTTS RVC UI",
        description="XTTS and RVC integration"
    )

    js_code = """
    <script>
    let timeout = null;
    const inputElem = document.querySelector("#text_input input");
    inputElem.addEventListener('input', function() {
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            document.querySelector('button:contains("Submit")').click();
        }, 2000);
    });
    </script>
    """
    interface.launch(server_name="127.0.0.1", server_port=5000, quiet=True, share=False)

class RVC_Data:
    def __init__(self):
        self.current_model = {}
        self.cpt = {}
        self.version = {}
        self.net_g = {}
        self.tgt_sr = {}
        self.vc = {}

    def load_cpt(self, modelname, rvc_model_path):
        try:
            if self.current_model != modelname:
                print("Loading new model")
                del self.cpt, self.version, self.net_g, self.tgt_sr, self.vc
                self.cpt, self.version, self.net_g, self.tgt_sr, self.vc = get_vc(device, config.is_half, config, rvc_model_path)
                self.current_model = modelname
        except Exception as e:
            print(f"Error in load_cpt: {e}")
            input("Press Enter to continue...")

rvc_data = RVC_Data()

def voice_change(rvc, pitch_change, index_rate):
    try:
        modelname = os.path.splitext(rvc)[0]
        print("Using RVC model: " + modelname)
        rvc_model_path = "./rvcs/" + rvc
        rvc_index_path = "./rvcs/" + modelname + ".index" if os.path.isfile("./rvcs/" + modelname + ".index") and index_rate != 0 else ""

        if rvc_index_path != "":
            print("Index file found!")

        rvc_data.load_cpt(modelname, rvc_model_path)

        rvc_infer(
            index_path=rvc_index_path,
            index_rate=index_rate,
            input_path="./output.wav",
            output_path="./outputrvc.wav",
            pitch_change=pitch_change,
            f0_method="rmvpe",
            cpt=rvc_data.cpt,
            version=rvc_data.version,
            net_g=rvc_data.net_g,
            filter_radius=3,
            tgt_sr=rvc_data.tgt_sr,
            rms_mix_rate=0.25,
            protect=0,
            crepe_hop_length=0,
            vc=rvc_data.vc,
            hubert_model=hubert_model
        )
        gc.collect()
    except Exception as e:
        print(f"Error in voice_change: {e}")
        input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        input("Press Enter to exit...")
