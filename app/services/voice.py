import os
import edge_tts
import asyncio
from loguru import logger
from datetime import datetime
from xml.sax.saxutils import unescape
from edge_tts import SubMaker
import time
from app.config import config
from app.utils import utils
from app.services import audio
from io import BytesIO
from pydub import AudioSegment


def get_all_azure_voices(filter_locals=None) -> list[str]:
    if filter_locals is None:
        filter_locals = ["zh-CN", "en-US", "zh-HK", "zh-TW", "vi-VN"]
    voices_str = """
Name: af-ZA-AdriNeural
Gender: Female

Name: af-ZA-WillemNeural
Gender: Male

Name: am-ET-AmehaNeural
Gender: Male

Name: am-ET-MekdesNeural
Gender: Female

Name: ar-AE-FatimaNeural
Gender: Female

Name: ar-AE-HamdanNeural
Gender: Male

Name: ar-BH-AliNeural
Gender: Male

Name: ar-BH-LailaNeural
Gender: Female

Name: ar-DZ-AminaNeural
Gender: Female

Name: ar-DZ-IsmaelNeural
Gender: Male

Name: ar-EG-SalmaNeural
Gender: Female

Name: ar-EG-ShakirNeural
Gender: Male

Name: ar-IQ-BasselNeural
Gender: Male

Name: ar-IQ-RanaNeural
Gender: Female

Name: ar-JO-SanaNeural
Gender: Female

Name: ar-JO-TaimNeural
Gender: Male

Name: ar-KW-FahedNeural
Gender: Male

Name: ar-KW-NouraNeural
Gender: Female

Name: ar-LB-LaylaNeural
Gender: Female

Name: ar-LB-RamiNeural
Gender: Male

Name: ar-LY-ImanNeural
Gender: Female

Name: ar-LY-OmarNeural
Gender: Male

Name: ar-MA-JamalNeural
Gender: Male

Name: ar-MA-MounaNeural
Gender: Female

Name: ar-OM-AbdullahNeural
Gender: Male

Name: ar-OM-AyshaNeural
Gender: Female

Name: ar-QA-AmalNeural
Gender: Female

Name: ar-QA-MoazNeural
Gender: Male

Name: ar-SA-HamedNeural
Gender: Male

Name: ar-SA-ZariyahNeural
Gender: Female

Name: ar-SY-AmanyNeural
Gender: Female

Name: ar-SY-LaithNeural
Gender: Male

Name: ar-TN-HediNeural
Gender: Male

Name: ar-TN-ReemNeural
Gender: Female

Name: ar-YE-MaryamNeural
Gender: Female

Name: ar-YE-SalehNeural
Gender: Male

Name: az-AZ-BabekNeural
Gender: Male

Name: az-AZ-BanuNeural
Gender: Female

Name: bg-BG-BorislavNeural
Gender: Male

Name: bg-BG-KalinaNeural
Gender: Female

Name: bn-BD-NabanitaNeural
Gender: Female

Name: bn-BD-PradeepNeural
Gender: Male

Name: bn-IN-BashkarNeural
Gender: Male

Name: bn-IN-TanishaaNeural
Gender: Female

Name: bs-BA-GoranNeural
Gender: Male

Name: bs-BA-VesnaNeural
Gender: Female

Name: ca-ES-EnricNeural
Gender: Male

Name: ca-ES-JoanaNeural
Gender: Female

Name: cs-CZ-AntoninNeural
Gender: Male

Name: cs-CZ-VlastaNeural
Gender: Female

Name: cy-GB-AledNeural
Gender: Male

Name: cy-GB-NiaNeural
Gender: Female

Name: da-DK-ChristelNeural
Gender: Female

Name: da-DK-JeppeNeural
Gender: Male

Name: de-AT-IngridNeural
Gender: Female

Name: de-AT-JonasNeural
Gender: Male

Name: de-CH-JanNeural
Gender: Male

Name: de-CH-LeniNeural
Gender: Female

Name: de-DE-AmalaNeural
Gender: Female

Name: de-DE-ConradNeural
Gender: Male

Name: de-DE-FlorianMultilingualNeural
Gender: Male

Name: de-DE-KatjaNeural
Gender: Female

Name: de-DE-KillianNeural
Gender: Male

Name: de-DE-SeraphinaMultilingualNeural
Gender: Female

Name: el-GR-AthinaNeural
Gender: Female

Name: el-GR-NestorasNeural
Gender: Male

Name: en-AU-NatashaNeural
Gender: Female

Name: en-AU-WilliamNeural
Gender: Male

Name: en-CA-ClaraNeural
Gender: Female

Name: en-CA-LiamNeural
Gender: Male

Name: en-GB-LibbyNeural
Gender: Female

Name: en-GB-MaisieNeural
Gender: Female

Name: en-GB-RyanNeural
Gender: Male

Name: en-GB-SoniaNeural
Gender: Female

Name: en-GB-ThomasNeural
Gender: Male

Name: en-HK-SamNeural
Gender: Male

Name: en-HK-YanNeural
Gender: Female

Name: en-IE-ConnorNeural
Gender: Male

Name: en-IE-EmilyNeural
Gender: Female

Name: en-IN-NeerjaExpressiveNeural
Gender: Female

Name: en-IN-NeerjaNeural
Gender: Female

Name: en-IN-PrabhatNeural
Gender: Male

Name: en-KE-AsiliaNeural
Gender: Female

Name: en-KE-ChilembaNeural
Gender: Male

Name: en-NG-AbeoNeural
Gender: Male

Name: en-NG-EzinneNeural
Gender: Female

Name: en-NZ-MitchellNeural
Gender: Male

Name: en-NZ-MollyNeural
Gender: Female

Name: en-PH-JamesNeural
Gender: Male

Name: en-PH-RosaNeural
Gender: Female

Name: en-SG-LunaNeural
Gender: Female

Name: en-SG-WayneNeural
Gender: Male

Name: en-TZ-ElimuNeural
Gender: Male

Name: en-TZ-ImaniNeural
Gender: Female

Name: en-US-AnaNeural
Gender: Female

Name: en-US-AndrewNeural
Gender: Male

Name: en-US-AriaNeural
Gender: Female

Name: en-US-AvaNeural
Gender: Female

Name: en-US-BrianNeural
Gender: Male

Name: en-US-ChristopherNeural
Gender: Male

Name: en-US-EmmaNeural
Gender: Female

Name: en-US-EricNeural
Gender: Male

Name: en-US-GuyNeural
Gender: Male

Name: en-US-JennyNeural
Gender: Female

Name: en-US-MichelleNeural
Gender: Female

Name: en-US-RogerNeural
Gender: Male

Name: en-US-SteffanNeural
Gender: Male

Name: en-ZA-LeahNeural
Gender: Female

Name: en-ZA-LukeNeural
Gender: Male

Name: es-AR-ElenaNeural
Gender: Female

Name: es-AR-TomasNeural
Gender: Male

Name: es-BO-MarceloNeural
Gender: Male

Name: es-BO-SofiaNeural
Gender: Female

Name: es-CL-CatalinaNeural
Gender: Female

Name: es-CL-LorenzoNeural
Gender: Male

Name: es-CO-GonzaloNeural
Gender: Male

Name: es-CO-SalomeNeural
Gender: Female

Name: es-CR-JuanNeural
Gender: Male

Name: es-CR-MariaNeural
Gender: Female

Name: es-CU-BelkysNeural
Gender: Female

Name: es-CU-ManuelNeural
Gender: Male

Name: es-DO-EmilioNeural
Gender: Male

Name: es-DO-RamonaNeural
Gender: Female

Name: es-EC-AndreaNeural
Gender: Female

Name: es-EC-LuisNeural
Gender: Male

Name: es-ES-AlvaroNeural
Gender: Male

Name: es-ES-ElviraNeural
Gender: Female

Name: es-ES-XimenaNeural
Gender: Female

Name: es-GQ-JavierNeural
Gender: Male

Name: es-GQ-TeresaNeural
Gender: Female

Name: es-GT-AndresNeural
Gender: Male

Name: es-GT-MartaNeural
Gender: Female

Name: es-HN-CarlosNeural
Gender: Male

Name: es-HN-KarlaNeural
Gender: Female

Name: es-MX-DaliaNeural
Gender: Female

Name: es-MX-JorgeNeural
Gender: Male

Name: es-NI-FedericoNeural
Gender: Male

Name: es-NI-YolandaNeural
Gender: Female

Name: es-PA-MargaritaNeural
Gender: Female

Name: es-PA-RobertoNeural
Gender: Male

Name: es-PE-AlexNeural
Gender: Male

Name: es-PE-CamilaNeural
Gender: Female

Name: es-PR-KarinaNeural
Gender: Female

Name: es-PR-VictorNeural
Gender: Male

Name: es-PY-MarioNeural
Gender: Male

Name: es-PY-TaniaNeural
Gender: Female

Name: es-SV-LorenaNeural
Gender: Female

Name: es-SV-RodrigoNeural
Gender: Male

Name: es-US-AlonsoNeural
Gender: Male

Name: es-US-PalomaNeural
Gender: Female

Name: es-UY-MateoNeural
Gender: Male

Name: es-UY-ValentinaNeural
Gender: Female

Name: es-VE-PaolaNeural
Gender: Female

Name: es-VE-SebastianNeural
Gender: Male

Name: et-EE-AnuNeural
Gender: Female

Name: et-EE-KertNeural
Gender: Male

Name: fa-IR-DilaraNeural
Gender: Female

Name: fa-IR-FaridNeural
Gender: Male

Name: fi-FI-HarriNeural
Gender: Male

Name: fi-FI-NooraNeural
Gender: Female

Name: fil-PH-AngeloNeural
Gender: Male

Name: fil-PH-BlessicaNeural
Gender: Female

Name: fr-BE-CharlineNeural
Gender: Female

Name: fr-BE-GerardNeural
Gender: Male

Name: fr-CA-AntoineNeural
Gender: Male

Name: fr-CA-JeanNeural
Gender: Male

Name: fr-CA-SylvieNeural
Gender: Female

Name: fr-CA-ThierryNeural
Gender: Male

Name: fr-CH-ArianeNeural
Gender: Female

Name: fr-CH-FabriceNeural
Gender: Male

Name: fr-FR-DeniseNeural
Gender: Female

Name: fr-FR-EloiseNeural
Gender: Female

Name: fr-FR-HenriNeural
Gender: Male

Name: fr-FR-RemyMultilingualNeural
Gender: Male

Name: fr-FR-VivienneMultilingualNeural
Gender: Female

Name: ga-IE-ColmNeural
Gender: Male

Name: ga-IE-OrlaNeural
Gender: Female

Name: gl-ES-RoiNeural
Gender: Male

Name: gl-ES-SabelaNeural
Gender: Female

Name: gu-IN-DhwaniNeural
Gender: Female

Name: gu-IN-NiranjanNeural
Gender: Male

Name: he-IL-AvriNeural
Gender: Male

Name: he-IL-HilaNeural
Gender: Female

Name: hi-IN-MadhurNeural
Gender: Male

Name: hi-IN-SwaraNeural
Gender: Female

Name: hr-HR-GabrijelaNeural
Gender: Female

Name: hr-HR-SreckoNeural
Gender: Male

Name: hu-HU-NoemiNeural
Gender: Female

Name: hu-HU-TamasNeural
Gender: Male

Name: id-ID-ArdiNeural
Gender: Male

Name: id-ID-GadisNeural
Gender: Female

Name: is-IS-GudrunNeural
Gender: Female

Name: is-IS-GunnarNeural
Gender: Male

Name: it-IT-DiegoNeural
Gender: Male

Name: it-IT-ElsaNeural
Gender: Female

Name: it-IT-GiuseppeNeural
Gender: Male

Name: it-IT-IsabellaNeural
Gender: Female

Name: ja-JP-KeitaNeural
Gender: Male

Name: ja-JP-NanamiNeural
Gender: Female

Name: jv-ID-DimasNeural
Gender: Male

Name: jv-ID-SitiNeural
Gender: Female

Name: ka-GE-EkaNeural
Gender: Female

Name: ka-GE-GiorgiNeural
Gender: Male

Name: kk-KZ-AigulNeural
Gender: Female

Name: kk-KZ-DauletNeural
Gender: Male

Name: km-KH-PisethNeural
Gender: Male

Name: km-KH-SreymomNeural
Gender: Female

Name: kn-IN-GaganNeural
Gender: Male

Name: kn-IN-SapnaNeural
Gender: Female

Name: ko-KR-HyunsuNeural
Gender: Male

Name: ko-KR-InJoonNeural
Gender: Male

Name: ko-KR-SunHiNeural
Gender: Female

Name: lo-LA-ChanthavongNeural
Gender: Male

Name: lo-LA-KeomanyNeural
Gender: Female

Name: lt-LT-LeonasNeural
Gender: Male

Name: lt-LT-OnaNeural
Gender: Female

Name: lv-LV-EveritaNeural
Gender: Female

Name: lv-LV-NilsNeural
Gender: Male

Name: mk-MK-AleksandarNeural
Gender: Male

Name: mk-MK-MarijaNeural
Gender: Female

Name: ml-IN-MidhunNeural
Gender: Male

Name: ml-IN-SobhanaNeural
Gender: Female

Name: mn-MN-BataaNeural
Gender: Male

Name: mn-MN-YesuiNeural
Gender: Female

Name: mr-IN-AarohiNeural
Gender: Female

Name: mr-IN-ManoharNeural
Gender: Male

Name: ms-MY-OsmanNeural
Gender: Male

Name: ms-MY-YasminNeural
Gender: Female

Name: mt-MT-GraceNeural
Gender: Female

Name: mt-MT-JosephNeural
Gender: Male

Name: my-MM-NilarNeural
Gender: Female

Name: my-MM-ThihaNeural
Gender: Male

Name: nb-NO-FinnNeural
Gender: Male

Name: nb-NO-PernilleNeural
Gender: Female

Name: ne-NP-HemkalaNeural
Gender: Female

Name: ne-NP-SagarNeural
Gender: Male

Name: nl-BE-ArnaudNeural
Gender: Male

Name: nl-BE-DenaNeural
Gender: Female

Name: nl-NL-ColetteNeural
Gender: Female

Name: nl-NL-FennaNeural
Gender: Female

Name: nl-NL-MaartenNeural
Gender: Male

Name: pl-PL-MarekNeural
Gender: Male

Name: pl-PL-ZofiaNeural
Gender: Female

Name: ps-AF-GulNawazNeural
Gender: Male

Name: ps-AF-LatifaNeural
Gender: Female

Name: pt-BR-AntonioNeural
Gender: Male

Name: pt-BR-FranciscaNeural
Gender: Female

Name: pt-BR-ThalitaNeural
Gender: Female

Name: pt-PT-DuarteNeural
Gender: Male

Name: pt-PT-RaquelNeural
Gender: Female

Name: ro-RO-AlinaNeural
Gender: Female

Name: ro-RO-EmilNeural
Gender: Male

Name: ru-RU-DmitryNeural
Gender: Male

Name: ru-RU-SvetlanaNeural
Gender: Female

Name: si-LK-SameeraNeural
Gender: Male

Name: si-LK-ThiliniNeural
Gender: Female

Name: sk-SK-LukasNeural
Gender: Male

Name: sk-SK-ViktoriaNeural
Gender: Female

Name: sl-SI-PetraNeural
Gender: Female

Name: sl-SI-RokNeural
Gender: Male

Name: so-SO-MuuseNeural
Gender: Male

Name: so-SO-UbaxNeural
Gender: Female

Name: sq-AL-AnilaNeural
Gender: Female

Name: sq-AL-IlirNeural
Gender: Male

Name: sr-RS-NicholasNeural
Gender: Male

Name: sr-RS-SophieNeural
Gender: Female

Name: su-ID-JajangNeural
Gender: Male

Name: su-ID-TutiNeural
Gender: Female

Name: sv-SE-MattiasNeural
Gender: Male

Name: sv-SE-SofieNeural
Gender: Female

Name: sw-KE-RafikiNeural
Gender: Male

Name: sw-KE-ZuriNeural
Gender: Female

Name: sw-TZ-DaudiNeural
Gender: Male

Name: sw-TZ-RehemaNeural
Gender: Female

Name: ta-IN-PallaviNeural
Gender: Female

Name: ta-IN-ValluvarNeural
Gender: Male

Name: ta-LK-KumarNeural
Gender: Male

Name: ta-LK-SaranyaNeural
Gender: Female

Name: ta-MY-KaniNeural
Gender: Female

Name: ta-MY-SuryaNeural
Gender: Male

Name: ta-SG-AnbuNeural
Gender: Male

Name: ta-SG-VenbaNeural
Gender: Female

Name: te-IN-MohanNeural
Gender: Male

Name: te-IN-ShrutiNeural
Gender: Female

Name: th-TH-NiwatNeural
Gender: Male

Name: th-TH-PremwadeeNeural
Gender: Female

Name: tr-TR-AhmetNeural
Gender: Male

Name: tr-TR-EmelNeural
Gender: Female

Name: uk-UA-OstapNeural
Gender: Male

Name: uk-UA-PolinaNeural
Gender: Female

Name: ur-IN-GulNeural
Gender: Female

Name: ur-IN-SalmanNeural
Gender: Male

Name: ur-PK-AsadNeural
Gender: Male

Name: ur-PK-UzmaNeural
Gender: Female

Name: uz-UZ-MadinaNeural
Gender: Female

Name: uz-UZ-SardorNeural
Gender: Male

Name: vi-VN-HoaiMyNeural
Gender: Female

Name: vi-VN-NamMinhNeural
Gender: Male

Name: zh-CN-XiaoxiaoNeural
Gender: Female

Name: zh-CN-XiaoyiNeural
Gender: Female

Name: zh-CN-YunjianNeural
Gender: Male

Name: zh-CN-YunxiNeural
Gender: Male

Name: zh-CN-YunxiaNeural
Gender: Male

Name: zh-CN-YunyangNeural
Gender: Male

Name: zh-CN-liaoning-XiaobeiNeural
Gender: Female

Name: zh-CN-shaanxi-XiaoniNeural
Gender: Female

Name: zh-HK-HiuGaaiNeural
Gender: Female

Name: zh-HK-HiuMaanNeural
Gender: Female

Name: zh-HK-WanLungNeural
Gender: Male

Name: zh-TW-HsiaoChenNeural
Gender: Female

Name: zh-TW-HsiaoYuNeural
Gender: Female

Name: zh-TW-YunJheNeural
Gender: Male

Name: zu-ZA-ThandoNeural
Gender: Female

Name: zu-ZA-ThembaNeural
Gender: Male


Name: en-US-AvaMultilingualNeural-V2
Gender: Female

Name: en-US-AndrewMultilingualNeural-V2
Gender: Male

Name: en-US-EmmaMultilingualNeural-V2
Gender: Female

Name: en-US-BrianMultilingualNeural-V2
Gender: Male

Name: de-DE-FlorianMultilingualNeural-V2
Gender: Male

Name: de-DE-SeraphinaMultilingualNeural-V2
Gender: Female

Name: fr-FR-RemyMultilingualNeural-V2
Gender: Male

Name: fr-FR-VivienneMultilingualNeural-V2
Gender: Female
    """.strip()
    voices = []
    name = ""
    for line in voices_str.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Name: "):
            name = line[6:].strip()
        if line.startswith("Gender: "):
            gender = line[8:].strip()
            if name and gender:
                # voices.append({
                #     "name": name,
                #     "gender": gender,
                # })
                if filter_locals:
                    for filter_local in filter_locals:
                        if name.lower().startswith(filter_local.lower()):
                            voices.append(f"{name}-{gender}")
                else:
                    voices.append(f"{name}-{gender}")
                name = ""
    voices.sort()
    return voices

def parse_voice_name(name: str):
    # zh-CN-XiaoyiNeural-Female
    # zh-CN-YunxiNeural-Male
    # zh-CN-XiaoxiaoMultilingualNeural-V2-Female
    name = name.replace("-Female", "").replace("-Male", "").strip()
    return name

def is_azure_v2_voice(voice_name: str):
    voice_name = parse_voice_name(voice_name)
    if voice_name.endswith("-V2"):
        return voice_name.replace("-V2", "").strip()
    return ""

def tts(text: str, voice_name: str, voice_rate: float, voice_pitch: float, voice_file: str,voice_volume:float,target_duration: float,
    max_rate_adjustments: int = 6,
    max_rate: float = 2.0) -> [SubMaker, None]:
    if is_azure_v2_voice(voice_name):
        return azure_tts_v2(text, voice_name, voice_file)
    return azure_tts_v1(text, voice_name, voice_rate, voice_pitch, voice_file,voice_volume,target_duration,max_rate_adjustments,max_rate)

def convert_rate_to_percent(rate: float) -> str:
    if rate == 1.0:
        return "+0%"
    percent = round((rate - 1.0) * 100)
    if percent > 0:
        return f"+{percent}%"
    else:
        return f"{percent}%"

def convert_volume_to_percent(volume: float) -> str:
    if volume == 1.0:
        return "+0%"
    percent = round((volume - 1.0) * 100)
    if percent > 0:
        return f"+{percent}%"
    else:
        return f"{percent}%"
    
def convert_pitch_to_percent(rate: float) -> str:
    if rate == 1.0:
        return "+0Hz"
    percent = round((rate - 1.0) * 100)
    if percent > 0:
        return f"+{percent}Hz"
    else:
        return f"{percent}Hz"

def azure_tts_v1(
    text: str, voice_name: str, voice_rate: float, voice_pitch: float, voice_file: str,voice_volume:float,
    target_duration: float,
    max_rate_adjustments: int = 6,
    max_rate: float = 2.0
) -> [SubMaker, None]:
    # get base params
    voice_name = parse_voice_name(voice_name)
    text = text.strip()
    volume_str = convert_volume_to_percent(voice_volume)
    pitch_str = convert_pitch_to_percent(voice_pitch)

    # get current_rate
    current_rate = voice_rate

    for i in range(max_rate_adjustments):
        try:
            logger.info(f"第 {i+1} 次使用 edge_tts 生成音频")

            async def _do() -> tuple[SubMaker, bytes]:
                communicate = edge_tts.Communicate(text, voice_name, rate=convert_rate_to_percent(current_rate), pitch=pitch_str,volume=volume_str)
                sub_maker = edge_tts.SubMaker()
                audio_data = bytes()
                
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                    elif chunk["type"] == "WordBoundary":
                        sub_maker.create_sub(
                            (chunk["offset"], chunk["duration"]), chunk["text"]
                        )
                return sub_maker, audio_data

            # judge if the audio file exists
            if os.path.exists(voice_file):
                logger.info(f"voice file exists, skip tts: {voice_file}")
                continue

            # get audio data
            sub_maker, audio_data = asyncio.run(_do())
            
            # judge if the audio data is valid
            if not sub_maker or not sub_maker.subs or not audio_data:
                logger.warning(f"failed, invalid data generated")
                time.sleep(1)
                continue

            # get new rate
            new_rate = get_next_voice_rate(audio_data,current_rate,target_duration,max_rate)
            if new_rate != current_rate and new_rate - current_rate > 0.01:
                current_rate = new_rate
                logger.info(f"new rate: {current_rate}")
                continue
            
            # write audio data to file
            with open(voice_file, "wb") as file:
                file.write(audio_data)
            logger.info(f"completed, output file: {voice_file}")
            return sub_maker
        except Exception as e:
            logger.error(f"生成音频文件时出错: {str(e)}")
            time.sleep(1)
    return None

def azure_tts_v2(text: str, voice_name: str, voice_file: str) -> [SubMaker, None]:
    voice_name = is_azure_v2_voice(voice_name)
    if not voice_name:
        logger.error(f"invalid voice name: {voice_name}")
        raise ValueError(f"invalid voice name: {voice_name}")
    text = text.strip()

    def _format_duration_to_offset(duration) -> int:
        if isinstance(duration, str):
            time_obj = datetime.strptime(duration, "%H:%M:%S.%f")
            milliseconds = (
                (time_obj.hour * 3600000)
                + (time_obj.minute * 60000)
                + (time_obj.second * 1000)
                + (time_obj.microsecond // 1000)
            )
            return milliseconds * 10000

        if isinstance(duration, int):
            return duration

        return 0

    for i in range(3):
        try:
            logger.info(f"start, voice name: {voice_name}, try: {i + 1}")

            import azure.cognitiveservices.speech as speechsdk

            sub_maker = SubMaker()

            def speech_synthesizer_word_boundary_cb(evt: speechsdk.SessionEventArgs):
                duration = _format_duration_to_offset(str(evt.duration))
                offset = _format_duration_to_offset(evt.audio_offset)
                sub_maker.subs.append(evt.text)
                sub_maker.offset.append((offset, offset + duration))

            # Creates an instance of a speech config with specified subscription key and service region.
            speech_key = config.azure.get("speech_key", "")
            service_region = config.azure.get("speech_region", "")
            audio_config = speechsdk.audio.AudioOutputConfig(
                filename=voice_file, use_default_speaker=True
            )
            speech_config = speechsdk.SpeechConfig(
                subscription=speech_key, region=service_region
            )
            speech_config.speech_synthesis_voice_name = voice_name
            # speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary,
            #                            value='true')
            speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestWordBoundary,
                value="true",
            )

            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
            )
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                audio_config=audio_config, speech_config=speech_config
            )
            speech_synthesizer.synthesis_word_boundary.connect(
                speech_synthesizer_word_boundary_cb
            )

            result = speech_synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.success(f"azure v2 speech synthesis succeeded: {voice_file}")
                return sub_maker
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error(
                    f"azure v2 speech synthesis canceled: {cancellation_details.reason}"
                )
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logger.error(
                        f"azure v2 speech synthesis error: {cancellation_details.error_details}"
                    )
            if i < 2:  # 如果不是最后一次重试，则等待1秒
                time.sleep(1)
            logger.info(f"completed, output file: {voice_file}")
        except Exception as e:
            logger.error(f"failed, error: {str(e)}")
            if i < 2:  # 如果不是最后一次重试，则等待1秒
                time.sleep(3)
    return None

def tts_multiple(out_path:str,subtitle_list: list, voice_name: str, voice_rate: float, voice_pitch: float, force_regenerate: bool = True,volume: float = 1.0) -> list:
    """
    根据JSON文件中的多段文本进行TTS转换
    
    :param out_path: 数据目录
    :param subtitle_list: 脚本列表 ->包含字幕序号、出现时间、和字幕文本的元组列表。
    :param voice_name: 语音名称
    :param voice_rate: 语音速率
    :param voice_pitch: 语音音调
    :param force_regenerate: 是否强制重新生成已存在的音频文件
    :param volume: 语音音量
    :return: 生成的音频文件列表
    """
    
    audio_files = []
    sub_maker_list = []

    for item in subtitle_list:
        index, timestamp_str, text = item

        # 将时间戳中的冒号替换为下划线
        timestamp = timestamp_str.split(' --> ')[0].replace(':', '_')
        audio_file = os.path.join(out_path, f"audio_{timestamp}.mp3")
        
        # 检查文件是否已存在，如存在且不强制重新生成，则跳过
        if os.path.exists(audio_file) and not force_regenerate:
            logger.info(f"音频文件已存在，跳过生成: {audio_file}")
            audio_files.append(audio_file)
            continue

        sub_times = timestamp_str.split(" --> ")
        start_time = utils.time_to_seconds(sub_times[0])
        end_time = utils.time_to_seconds(sub_times[1])
        duration = end_time - start_time
        sub_maker = tts(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_pitch=voice_pitch,
            voice_file=audio_file,
            voice_volume=volume,
            target_duration=duration,
            max_rate_adjustments=6,
            max_rate=2.0
        )

        if sub_maker is None:
            logger.error(f"无法为时间戳 {timestamp} 生成音频; "
                            f"如果您在中国，请使用VPN; "
                            f"或者使用其他 tts 引擎")
            continue

        audio_files.append(audio_file)
        sub_maker_list.append(sub_maker)
        logger.info(f"已生成音频文件: {audio_file}")

    return audio_files, sub_maker_list

def subtitle_to_voice(subtitles:list[tuple[int,str,str]],temp_path:str,voice_name:str,
                      voice_rate:float,voice_pitch:float,voice_volume:float,
                      out_path:str,
                      merged_subtitle_path:str) -> str:
    # get audios
    audio_files, sub_maker_list = tts_multiple(
        out_path=temp_path,
        subtitle_list=subtitles, 
        voice_name=voice_name,
        voice_rate=voice_rate,
        voice_pitch=voice_pitch,
        force_regenerate=True,
        volume=voice_volume
    )
    
    if audio_files is None:
        raise Exception("tts failed, audio_files is None")

    # merge audios
    final_audio = None
    try:
        # save final audio
        final_audio = audio.merge_audio_files(
            out_path=out_path,
            audio_files=audio_files,
            subtitle_list=subtitles,
            merged_subtitle_path=merged_subtitle_path
        )
    except Exception as e:
        raise Exception("merge audio failed:" + str(e))
    finally:
        # remove audio files
        if audio_files is not None:
            for audio_file in audio_files:
                if os.path.exists(audio_file):
                    os.remove(audio_file)

        del audio_files
        del sub_maker_list
    return final_audio

def get_next_voice_rate(audio_data,current_rate: float, target_duration: float,max_rate: float) -> float:
    with BytesIO(audio_data) as audio_stream:
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        actual_duration = len(audio) / 1000.0

    if actual_duration <= target_duration:
        return current_rate
    
    # get new rate
    adjustment_factor = actual_duration / target_duration
    new_rate = current_rate * adjustment_factor
    
    # limit new rate
    new_rate = min(new_rate, max_rate)

    return new_rate
