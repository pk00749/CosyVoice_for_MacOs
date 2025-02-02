#!/usr/bin/env python
#coding=utf-8
import os
import time
from typing import Generator
from tqdm import tqdm
from modelscope import snapshot_download
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from .frontend import CosyVoiceFrontEnd
from .model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.utils.class_utils import get_model_type


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def time_it(func):
  """
  这是一个装饰器，用来计算类方法运行的时长，单位秒.
  """
  def wrapper(self, *args, **kwargs):
    start_time = time.time()
    result = func(self, *args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    print(f"推理方法 {func.__name__} 运行时长: {duration:.4f} 秒")
    return result
  return wrapper


def ms_to_srt_time(ms):
    N = int(ms)
    hours, remainder = divmod(N, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    timesrt = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    # print(timesrt)
    return timesrt

class CosyVoice:

    def __init__(self, model_dir):
        instruct = True if '-Instruct' in model_dir else False
        self.model_dir = f"{ROOT_DIR}/{model_dir}"
        with open(f'{self.model_dir}/cosyvoice.yaml', 'r') as f:
            configs = load_hyperpyyaml(f)
        # assert get_model_type(configs) != CosyVoice2Model, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          instruct,
                                          configs['allowed_special'])
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'])
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        del configs

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    @time_it
    def inference_sft(self, tts_text, spk_id, new_dropdown, spk_mix="无",
                        w1=0.5, w2=0.5, token_max_n=30, token_min_n=20, merge_len=15):
        default_voices = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']

        # if new_dropdown != "无":
        #     spk_id = "中文女"
        tts_speeches = []
        audio_opt = []
        audio_samples = 0
        srtlines = []
        for i in self.frontend.text_normalize(tts_text,True,token_max_n,token_min_n,merge_len):
            print(i)
            spk_id = spk_id if spk_id in default_voices else "中文女"
            model_input = self.frontend.frontend_sft(i, spk_id)

            if new_dropdown != "无" or spk_id not in default_voices:
                if spk_id not in default_voices:
                    new_dropdown = spk_id
                # 加载数据
                print(f"读取pt:{new_dropdown}")
                newspk = torch.load(f'{ROOT_DIR}/voices/{new_dropdown}.pt', map_location=torch.device('cpu'))

                if spk_mix != "无":
                    print("融合音色:",spk_mix)
                    if spk_mix not in ["中文女","中文男","中文男","日语男","粤语女","粤语女","英文女","英文男","韩语女"]:
                        newspk_1 = torch.load(f'{ROOT_DIR}/voices/{spk_mix}.pt', map_location=torch.device('cpu'))
                    else:
                        newspk_1 = self.frontend.frontend_sft(i, spk_mix)

                    model_input["flow_embedding"] = (newspk["flow_embedding"] * w1) + (newspk_1["flow_embedding"] * w2)
                    # model_input["llm_embedding"] = (newspk["llm_embedding"] * w1) + (newspk_1["llm_embedding"] * w2)
                else:
                    model_input["flow_embedding"] = newspk["flow_embedding"] 
                    model_input["llm_embedding"] = newspk["llm_embedding"]

                model_input["llm_prompt_speech_token"] = newspk["llm_prompt_speech_token"]
                model_input["llm_prompt_speech_token_len"] = newspk["llm_prompt_speech_token_len"]

                model_input["flow_prompt_speech_token"] = newspk["flow_prompt_speech_token"]
                model_input["flow_prompt_speech_token_len"] = newspk["flow_prompt_speech_token_len"]

                model_input["prompt_speech_feat_len"] = newspk["prompt_speech_feat_len"]
                model_input["prompt_speech_feat"] = newspk["prompt_speech_feat"]
                model_input["prompt_text"] = newspk["prompt_text"]
                model_input["prompt_text_len"] = newspk["prompt_text_len"]

            model_output = self.model.inference(**model_input)
            print(model_output['tts_speech'])

            # 使用 .numpy() 方法将 tensor 转换为 numpy 数组
            numpy_array = model_output['tts_speech'].numpy()
            # 使用 np.ravel() 方法将多维数组展平成一维数组
            audio = numpy_array.ravel()
            print(audio)
            srtline_begin=ms_to_srt_time(audio_samples*1000.0 / 22050)
            audio_samples += audio.size
            srtline_end=ms_to_srt_time(audio_samples*1000.0 / 22050)
            audio_opt.append(audio)

            srtlines.append(f"{len(audio_opt):02d}\n")
            srtlines.append(srtline_begin+' --> '+srtline_end+"\n")
            srtlines.append(i.replace("、。","")+"\n\n")

            tts_speeches.append(model_output['tts_speech'])

        print(tts_speeches)
        audio_data = torch.concat(tts_speeches, dim=1)
        with open(f'{ROOT_DIR}/音频输出/output_sft.srt', 'w', encoding='utf-8') as f:
            f.writelines(srtlines)

        return {'tts_speech':audio_data}

    @time_it
    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            # 保存数据
            torch.save(model_input, f'{ROOT_DIR}/output_zero_shot.pt') 
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])

        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    @time_it
    def inference_cross_lingual(self, tts_text, prompt_speech_16k):
        if self.frontend.instruct is True:
            raise ValueError('{} do not support cross_lingual inference'.format(self.model_dir))
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])

        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    @time_it
    def inference_instruct(self, tts_text, spk_id, instruct_text,new_dropdown):
        if new_dropdown != "无":
            spk_id = "中文女"

        if self.frontend.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize_instruct(instruct_text, split=False)
        tts_speeches = []

        for i in self.frontend.text_normalize_instruct(tts_text, split=True):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)

            if new_dropdown != "无":
                # 加载数据
                print(f"读取pt:{new_dropdown}")
                newspk = torch.load(f'{ROOT_DIR}/voices/{new_dropdown}.pt')
                model_input["flow_embedding"] = newspk["flow_embedding"]
                model_input["llm_embedding"] = newspk["llm_embedding"]

                model_input["llm_prompt_speech_token"] = newspk["llm_prompt_speech_token"]
                model_input["llm_prompt_speech_token_len"] = newspk["llm_prompt_speech_token_len"]

                model_input["flow_prompt_speech_token"] = newspk["flow_prompt_speech_token"]
                model_input["flow_prompt_speech_token_len"] = newspk["flow_prompt_speech_token_len"]

                model_input["prompt_speech_feat_len"] = newspk["prompt_speech_feat_len"]
                model_input["prompt_speech_feat"] = newspk["prompt_speech_feat"]
                model_input["prompt_text"] = newspk["prompt_text"]
                model_input["prompt_text_len"] = newspk["prompt_text_len"]
            model_output = self.model.inference(**model_input)
            
            tts_speeches.append(model_output['tts_speech'])

        return {'tts_speech': torch.concat(tts_speeches, dim=1)}


class CosyVoice2(CosyVoice):

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                self.fp16)
        del configs

    def inference_instruct(self, *args, **kwargs):
        raise NotImplementedError('inference_instruct is not implemented for CosyVoice2!')

    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text) #, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text)): #, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text)): #, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()
