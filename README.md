# 启动接口服务

python3 api.py

```
url接口地址: http://localhost:9880/?text=测试测试，这里是测试&speaker=中文女
```

```
字幕文件地址:http://localhost:9880/file/output.srt
```

```
音频文件地址:http://localhost:9880/file/output.wav
```

# CosyVoice
## 👉🏻 [CosyVoice Demos](https://fun-audio-llm.github.io/) 👈🏻
[[CosyVoice Paper](https://fun-audio-llm.github.io/pdf/CosyVoice_v1.pdf)][[CosyVoice Studio](https://www.modelscope.cn/studios/iic/CosyVoice-300M)][[CosyVoice Code](https://github.com/FunAudioLLM/CosyVoice)]

For `SenseVoice`, visit [SenseVoice repo](https://github.com/FunAudioLLM/SenseVoice) and [SenseVoice space](https://www.modelscope.cn/studios/iic/SenseVoice).

## Install

**References**
- https://blog.zhheo.com/p/e950.html
- https://www.soinside.com/question/KrTV2VQsaKq4v5YQyMkFba
- https://geek-docs.com/pytorch/pytorch-questions/277_pytorch_cannot_import_torch_audio_no_audio_backend_is_available.html

**Something to be installed on MacOS**
```sh
brew install sox git-lfs
```

**Clone and install**
- Clone the repo
``` sh
git clone https://github.com/v3ucn/CosyVoice_for_MacOs.git
# If you failed to clone submodule due to network failures, please run following command until success
cd CosyVoice_for_MacOs
git submodule update --init --recursive
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n cosyvoice python=3.8
conda activate cosyvoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

**Check torch related version**

``` sh
python check_version.py
```

**Check audio backend**
```sh
python check_backend.py
```
If return empty list, means not able to proceed audio.
Solution I used is, uninstall sox, librosa, soundfile, torch, torchaudio, torchvision, then reinstall them with version.
```sh
pip install sox librosa soundfile
pip install torch torchaudio torchvision
```


**Model download**
Create folder
```sh
mkdir -p pretrained_models
```
We strongly recommand that you download our pretrained `CosyVoice-300M` `CosyVoice-300M-SFT` `CosyVoice-300M-Instruct` model and `speech_kantts_ttsfrd` resource.

> If you are expert in this field, and you are only interested in training your own CosyVoice model from scratch, you can skip this step.


```sh
python download_models.py
```


**Basic Usage**

- First and important, add `third_party/AcademiCodec` and `third_party/Matcha-TTS` to `PYTHONPATH`:
``` sh
export PYTHONPATH=third_party/AcademiCodec;third_party/Matcha-TTS
```

**Sample**
- For zero_shot/cross_lingual inference, please use `CosyVoice-300M` model.
- For sft inference, please use `CosyVoice-300M-SFT` model.
- For instruct inference, please use `CosyVoice-300M-Instruct` model.
``` python
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice('speech_tts/CosyVoice-300M-SFT')
# sft usage
print(cosyvoice.list_avaliable_spks())
output = cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', '无')
torchaudio.save('./output/sft.wav', output['tts_speech'], 22050)

cosyvoice = CosyVoice('speech_tts/CosyVoice-300M')
# zero_shot usage
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
output = cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k)
torchaudio.save('./output/zero_shot.wav', output['tts_speech'], 22050)
# cross_lingual usage
prompt_speech_16k = load_wav('./asset/cross_lingual_prompt.wav', 16000)
output = cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.', prompt_speech_16k)
torchaudio.save('./output/cross_lingual.wav', output['tts_speech'], 22050)

#cosyvoice = CosyVoice('speech_tts/CosyVoice-300M-Instruct')
## instruct usage
#output = cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男', 'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.','无')
#torchaudio.save('./output/instruct.wav', output['tts_speech'], 22050)
```

**Start web demo**

You can use our web demo page to get familiar with CosyVoice quickly.
We support sft/zero_shot/cross_lingual/instruct inference in web demo.

Please see the demo website for details.

``` python
# change speech_tts/CosyVoice-300M-SFT for sft inference, or speech_tts/CosyVoice-300M-Instruct for instruct inference
python3 webui.py --port 9886 --model_dir ./pretrained_models/CosyVoice-300M
```
![PixPin_2024-07-07_15-00-18](https://github.com/v3ucn/CosyVoice_For_Windows/assets/1288038/7c6fa726-050a-4d54-9973-fe8c6a284ef3)


**Advanced Usage**

For advanced user, we have provided train and inference scripts in `examples/libritts/cosyvoice/run.sh`.
You can get familiar with CosyVoice following this recipie.

**Build for deployment**

Optionally, if you want to use grpc for service deployment,
you can run following steps. Otherwise, you can just ignore this step.

``` sh
cd runtime/python
docker build -t cosyvoice:v1.0 .
# change speech_tts/CosyVoice-300M to speech_tts/CosyVoice-300M-Instruct if you want to use instruct inference
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python && python3 server.py --port 50000 --max_conc 4 --model_dir speech_tts/CosyVoice-300M && sleep infinity"
python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
```

## Trouble shooting

- ImportError: There is no such entity as cosyvoice.utils.common.ras_sampling  

Reference: https://github.com/FunAudioLLM/CosyVoice/issues/325  
Solution: 
Locate model folder, such as pretrained_models/CosyVoice-300M, edit cosyvoice.yaml and comment below content:
```yaml
    #sampling: !name:cosyvoice.utils.common.ras_sampling
    #    top_p: 0.8
    #    top_k: 25
    #    win_size: 10
    #    tau_r: 0.1
```

- RuntimeError: Couldn't find appropriate backend to handle uri /Users/yorkhxli/git/CosyVoice_for_MacOs/音频输出/output.wav and format None  
Solution: Due to no any audio backend, plese refer to **Check audio backend**

## Discussion & Communication

You can directly discuss on [Github Issues](https://github.com/FunAudioLLM/CosyVoice/issues).

You can also scan the QR code to join our officla Dingding chat group.

<img src="./asset/dingding.png" width="250px">

## Acknowledge

1. We borrowed a lot of code from [FunASR](https://github.com/modelscope/FunASR).
2. We borrowed a lot of code from [FunCodec](https://github.com/modelscope/FunCodec).
3. We borrowed a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
4. We borrowed a lot of code from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
5. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet).

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.
