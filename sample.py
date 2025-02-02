from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import sys, os

# set environment variable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/third_party/AcademiCodec')
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

# check speaker
print("List out speaker:")
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
print(cosyvoice.list_avaliable_spks())

# sft usage
print("===== inference sft usage =====")
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
output = cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', '无')
torchaudio.save('./output/sft.wav', output['tts_speech'], 22050)

# zero_shot usage
print("===== zero shot usage =====")
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
output = cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k)
torchaudio.save('./output/zero_shot.wav', output['tts_speech'], 22050)

# cross_lingual usage
print("===== cross lingual usage =====")
prompt_speech_16k = load_wav('./asset/cross_lingual_prompt.wav', 16000)
output = cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.', prompt_speech_16k)
torchaudio.save('./output/cross_lingual.wav', output['tts_speech'], 22050)

# instruct usage
print("===== instruct usage =====")
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
output = cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男', 'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.','无')
torchaudio.save('./output/instruct.wav', output['tts_speech'], 22050)
