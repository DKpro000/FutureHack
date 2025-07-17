import os
import cv2
import torch
import librosa
import numpy as np
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    ClapModel,
    ClapProcessor,
    pipeline,
)
from torchvision import transforms, models
from torch import nn
import json
import tempfile
import uuid
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings("ignore")
import argparse

try:
    from imageio_ffmpeg import get_ffmpeg_exe
    FFMPEG_BIN = get_ffmpeg_exe()
    print(f"[INFO] Using bundled ffmpeg: {FFMPEG_BIN}")
except Exception:
    import shutil
    FFMPEG_BIN = shutil.which("ffmpeg")
    if FFMPEG_BIN:
        print(f"[INFO] Using system ffmpeg: {FFMPEG_BIN}")
    else:
        print("[WARNING] No ffmpeg binary found; audio extraction will fail.")

def _extract_audio_via_ffmpeg(video_path: str, target_sr: int = 48000) -> Tuple[np.ndarray, int]:
    """Extract mono float32 audio from a video using ffmpeg. Raises RuntimeError on failure."""
    if not FFMPEG_BIN:
        raise RuntimeError("No ffmpeg binary available.")
    import subprocess
    tmp_wav = os.path.join(tempfile.gettempdir(), f"tmpaudio_{uuid.uuid4().hex}.wav")
    cmd = [
        FFMPEG_BIN, "-y", "-i", video_path,
        "-ac", "1",
        "-ar", str(target_sr),
        "-vn",
        "-f", "wav", tmp_wav,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not os.path.exists(tmp_wav):
        msg = proc.stderr.decode(errors="ignore")[:500]
        raise RuntimeError(f"ffmpeg extraction failed (rc={proc.returncode}): {msg}")
    audio, sr = librosa.load(tmp_wav, sr=target_sr, mono=True)
    try:
        os.remove(tmp_wav)
    except OSError:
        pass
    return audio.astype(np.float32), sr


class VideoAnalysisSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        self._init_blip_model()
        self._init_clap_model()
        self._init_resnet_model()
        self._init_llm_model()

    def _init_blip_model(self):
        print("[INFO] Loading BLIP model...")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        self.blip_processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-opt-2.7b", use_fast=True
        )
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            quantization_config=quant_config,
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True,
        )
        self.blip_model.eval()

    def _init_clap_model(self):
        print("[INFO] Loading CLAP model...")
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(self.device)
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

        # NOTE: you can freely extend / edit these bark prompt templates.
        self.audio_prompts: Dict[str, List[str]] = {
            "Alert": [
                "sharp mid-pitch alert bark",
                "brief clear alarm bark",
                "short crisp warning bark",
                "fast staccato alert bark",
            ],
            "Territorial": [
                "deep sustained territorial bark",
                "low throaty guard bark",
                "prolonged defensive bark",
                "slow booming territorial bark",
            ],
            "Excited": [
                "high-pitched rapid excited yips",
                "quick playful bark",
                "series of lively yips",
                "fast bright excited bark",
            ],
            "Demand": [
                "steady attention-seeking bark",
                "regular rhythmic demand bark",
                "persistent repetitive request bark",
                "moderate-pitch insistence bark",
            ],
            "Fear": [
                "high-pitched trembling fearful bark",
                "quivering anxious bark",
                "shaky high anxious bark",
                "piercing nervous bark",
            ],
            "Aggressive": [
                "low guttural aggressive bark",
                "deep harsh threat bark",
                "rough menacing bark",
                "raspy growling attack bark",
            ],
            "Pain": [
                "single sharp pain yelp",
                "shrill acute pain bark",
                "sudden high pain yelp",
                "short piercing pain bark",
            ],
            "Lonely": [
                "slow spaced lonely bark",
                "mournful drawn-out bark",
                "distant monotone lonely bark",
                "long-interval melancholy bark",
            ],
            "Howl": [
                "long plaintive canine howl",
                "careless, spontaneous howl",
                "sustained mournful howl",
                "extended melodic dog howl",
            ],
        }

        all_prompts: List[str] = []
        prompt_classes: List[str] = []
        for cls, plist in self.audio_prompts.items():
            all_prompts.extend(plist)
            prompt_classes.extend([cls] * len(plist))
        self.prompt_texts = all_prompts
        self.prompt_cls = np.array(prompt_classes)

        # Cache text embeddings (normalized)
        txt_inputs = self.clap_processor(text=all_prompts, return_tensors="pt", padding=True)
        txt_inputs = {k: v.to(self.device) for k, v in txt_inputs.items()}
        with torch.no_grad():
            self.text_emb = torch.nn.functional.normalize(
                self.clap_model.get_text_features(**txt_inputs), dim=-1
            )

    def _init_resnet_model(self):
        print("[INFO] Loading ResNet18 model...")
        self.resnet_model = models.resnet18(pretrained=True)
        if os.path.exists('dog_emotion.pth'):
            try:
                checkpoint = torch.load('dog_emotion.pth', map_location=self.device)
                if 'fc.weight' in checkpoint:
                    num_classes = checkpoint['fc.weight'].shape[0]
                    print(f"[INFO] Detected {num_classes} emotion classes from checkpoint")
                else:
                    num_classes = 4
                    print(f"[INFO] Using default {num_classes} emotion classes")
                if num_classes == 4:
                    emotion_classes = ['angry','happy','relaxed','sad']
                elif num_classes == 5:
                    emotion_classes = ['angry','happy','relaxed','sad','calm']
                else:
                    emotion_classes = [f'emotion_{i}' for i in range(num_classes)]
                self.emotion_classes = emotion_classes
                self.resnet_model.fc = nn.Linear(self.resnet_model.fc.in_features, num_classes)
                self.resnet_model.load_state_dict(checkpoint)
                print("[INFO] Loaded pre-trained emotion model")
            except Exception as e:
                print(f"[WARNING] Error loading emotion model: {e}")
                print("[INFO] Using default emotion classes")
                self.emotion_classes = ['angry','happy','relaxed','sad']
                self.resnet_model.fc = nn.Linear(self.resnet_model.fc.in_features, 4)
        else:
            print("[INFO] No pre-trained emotion model found, using default classes")
            self.emotion_classes = ['angry','happy','relaxed','sad']
            self.resnet_model.fc = nn.Linear(self.resnet_model.fc.in_features, 4)
        self.resnet_model = self.resnet_model.to(self.device)
        self.resnet_model.eval()
        self.image_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def _init_llm_model(self):
        print("[INFO] Loading LLM model...")
        try:
            self.llm = pipeline(
                "text-generation", model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception:
            print("[WARNING] Could not load LLM model, using rule-based analysis")
            self.llm = None

    def extract_video_components(self, video_path: str, sample_rate: int = 48000) -> Tuple[List[np.ndarray], np.ndarray]:
        print(f"[INFO] Extracting components from {video_path}")
        cap = cv2.VideoCapture(video_path)
        frames: List[np.ndarray] = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print("[WARNING] Could not get total frame count, reading sequentially.")
            for _ in range(3):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        else:
            indices = [0, total_frames//2, total_frames-1]
            unique_indices = sorted(set(indices))
            for idx in unique_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"[WARNING] Could not read frame at index {idx}")
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        cap.release()
        print(f"[INFO] Extracted {len(frames)} frames")

        try:
            audio, sr = _extract_audio_via_ffmpeg(video_path, target_sr=sample_rate)
            print(f"[INFO] Extracted audio: {len(audio)} samples at {sr} Hz (ffmpeg).")
        except Exception as e:
            print(f"[WARNING] Could not extract audio via ffmpeg: {e}")
            audio = np.array([])
        return frames, audio

    def analyze_scenes(self, frames: List[np.ndarray]) -> List[str]:
        print("[INFO] Analyzing scenes with BLIP...")
        scene_descriptions: List[str] = []
        for i, frame in enumerate(frames):
            try:
                img = Image.fromarray(frame).convert("RGB")
                inputs = self.blip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self.blip_model.device) for k, v in inputs.items()}
                generated_ids = self.blip_model.generate(
                    **inputs,
                    max_new_tokens=80,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                )
                description = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                scene_descriptions.append(description)
                print(f"[INFO] Frame {i+1}: {description}")
            except Exception as e:
                print(f"[WARNING] Error analyzing frame {i+1}: {e}")
                scene_descriptions.append("Unable to analyze scene")
        return scene_descriptions

    def analyze_emotions(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        print("[INFO] Analyzing emotions with ResNet18...")
        results: List[Dict[str, Any]] = []
        for i, frame in enumerate(frames):
            try:
                img = Image.fromarray(frame).convert("RGB")
                img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.resnet_model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    pred_idx = torch.argmax(probs).item()
                    conf = probs[pred_idx].item()
                    result = {
                        'emotion': self.emotion_classes[pred_idx],
                        'confidence': conf,
                        'all_probabilities': {e: p.item() for e, p in zip(self.emotion_classes, probs)}
                    }
                    results.append(result)
                    print(f"[INFO] Frame {i+1}: {result['emotion']} ({conf:.3f})")
            except Exception as e:
                print(f"[WARNING] Error analyzing emotion for frame {i+1}: {e}")
                results.append({'emotion':'unknown','confidence':0.0,'all_probabilities':{}})
        return results

    def analyze_audio(self, audio: np.ndarray, sample_rate: int = 48000, top_k: int = 3) -> Dict[str, Any]:
        print("[INFO] Analyzing audio with CLAP...")
        if audio is None or len(audio) == 0:
            print("[INFO] No audio available; returning Silent.")
            return {
                'predicted_class':'Silent',
                'class_confidence':0.0,
                'predicted_phrase':'(no audio)',
                'confidence_phrase':0.0,
                'top_k_phrases':[],
                'all_probabilities_class':{},
                'all_probabilities_phrase':{}
            }

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak

        seg_seconds = 10
        seg_len = seg_seconds * sample_rate
        segments = [audio[i:i+seg_len] for i in range(0, len(audio), seg_len)]
        phrase_probs_list = []

        for seg in segments:
            if len(seg) < sample_rate:  # pad to at least 1s
                seg = np.pad(seg, (0, sample_rate-len(seg)), mode='constant')
            a_in = self.clap_processor(
                audios=[seg], sampling_rate=sample_rate,
                return_tensors="pt", padding=True
            )
            a_in = {k: v.to(self.device) for k, v in a_in.items()}
            with torch.no_grad():
                a_emb = torch.nn.functional.normalize(
                    self.clap_model.get_audio_features(**a_in), dim=-1
                )
            sim = (a_emb @ self.text_emb.T)  # [1, num_prompts]
            seg_prob = torch.softmax(sim, dim=-1)[0].cpu().numpy()  # [num_prompts]
            phrase_probs_list.append(seg_prob)

        if not phrase_probs_list:
            phrase_probs = np.zeros(len(self.prompt_texts), dtype=np.float32)
        else:
            phrase_probs = np.mean(np.vstack(phrase_probs_list), axis=0)

        cls_prob: Dict[str, float] = {}
        for cls in self.audio_prompts:
            mask = (self.prompt_cls == cls)
            cls_prob[cls] = float(np.mean(phrase_probs[mask])) if np.any(mask) else 0.0

        best_phrase_idx = int(np.argmax(phrase_probs)) if phrase_probs.size else -1
        if best_phrase_idx >= 0:
            best_phrase = self.prompt_texts[best_phrase_idx]
            best_phrase_cls = self.prompt_cls[best_phrase_idx]
            best_phrase_conf = float(phrase_probs[best_phrase_idx])
        else:
            best_phrase = '(unknown)'
            best_phrase_cls = 'unknown'
            best_phrase_conf = 0.0

        pred_cls = max(cls_prob, key=cls_prob.get) if cls_prob else 'unknown'
        cls_conf = cls_prob.get(pred_cls, 0.0)

        sort_idx = np.argsort(phrase_probs)[::-1]
        top_items = []
        for idx in sort_idx[:top_k]:
            top_items.append((self.prompt_texts[idx], self.prompt_cls[idx], float(phrase_probs[idx])))

        all_phrase_dict = {p: float(phrase_probs[i]) for i, p in enumerate(self.prompt_texts)}

        print(f"[INFO] Audio analysis: {best_phrase_cls} -> {best_phrase} ({best_phrase_conf:.3f})")

        return {
            'predicted_class': pred_cls,
            'class_confidence': cls_conf,
            'predicted_phrase': best_phrase,
            'confidence_phrase': best_phrase_conf,
            'top_k_phrases': top_items,
            'all_probabilities_class': cls_prob,
            'all_probabilities_phrase': all_phrase_dict,
        }

    def synthesize_analysis(self, scene_descriptions: List[str], emotion_results: List[Dict[str, Any]], audio_result: Dict[str, Any]) -> Dict[str, Any]:
        print("[INFO] Synthesizing analysis...")
        emotions = [r['emotion'] for r in emotion_results if r['emotion'] != 'unknown']
        most_common = max(set(emotions), key=emotions.count) if emotions else 'unknown'
        avg_conf = float(np.mean([r['confidence'] for r in emotion_results if r['confidence']>0])) if emotion_results else 0.0
        scene_keywords: List[str] = []
        stop = {'the','and','with','that','this'}
        for desc in scene_descriptions:
            for w in desc.split():
                wl = w.lower()
                if len(wl)>3 and wl not in stop:
                    scene_keywords.append(wl)
        analysis: Dict[str, Any] = {
            'dominant_emotion': most_common,
            'emotion_confidence': avg_conf,
            'audio_behavior': audio_result['predicted_class'],
            'audio_behavior_confidence': audio_result['class_confidence'],
            'audio_phrase': audio_result['predicted_phrase'],
            'audio_phrase_confidence': audio_result['confidence_phrase'],
            'scene_context': scene_descriptions,
            'scene_keywords': list(set(scene_keywords)),
            'detailed_emotions': emotion_results,
            'audio_details': audio_result,
        }
        analysis['dog_thoughts'] = self._generate_dog_thoughts(analysis)
        return analysis

    def _generate_dog_thoughts(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        emotion = analysis['dominant_emotion']
        audio_behavior = analysis['audio_behavior']
        scene_keywords = analysis['scene_keywords']
        t = {'mood':'neutral','likely_thoughts':[],'behavioral_interpretation':'','needs_attention':False}
        if emotion == 'happy' and audio_behavior in ['Excited','Demand']:
            t.update(mood='playful', likely_thoughts=['I want to play!','This is fun!','Pay attention to me!'],
                     behavioral_interpretation='The dog appears playful and seeking interaction.')
        elif emotion == 'sad' or audio_behavior in ['Lonely','Pain']:
            t.update(mood='distressed', likely_thoughts=['I feel lonely','I need comfort','Something is bothering me'],
                     behavioral_interpretation='The dog may be distressed and needs comfort or attention.',
                     needs_attention=True)
        elif audio_behavior in ['Alert','Territorial']:
            t.update(mood='vigilant', likely_thoughts=['Something is happening','I need to protect my territory','Alert! Someone is coming'],
                     behavioral_interpretation='The dog is alert to external stimuli.')
        elif emotion == 'angry' or audio_behavior == 'Aggressive':
            t.update(mood='defensive', likely_thoughts=['I feel threatened','Stay away from me','I need to defend myself'],
                     behavioral_interpretation='The dog shows defensive/aggressive behavior.',
                     needs_attention=True)
        elif emotion == 'calm':
            t.update(mood='relaxed', likely_thoughts=['I feel comfortable','Everything is peaceful','I am content'],
                     behavioral_interpretation='The dog seems relaxed.')
        elif emotion and emotion not in ['unknown','neutral']:
            t.update(mood=emotion, likely_thoughts=[f'I am feeling {emotion}'],
                     behavioral_interpretation=f'The dog is displaying {emotion} behavior.')
        else:
            t.update(mood='neutral', likely_thoughts=['Observing the environment'],
                     behavioral_interpretation='Neutral / observing.')
        if 'food' in scene_keywords or 'eating' in scene_keywords:
            t['likely_thoughts'].append('Food! I want some!')
        if 'person' in scene_keywords or 'human' in scene_keywords:
            t['likely_thoughts'].append('My human is here!')
        if 'outside' in scene_keywords or 'park' in scene_keywords:
            t['likely_thoughts'].append('Time for adventure!')
        return t

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        print(f"[INFO] Starting video analysis for: {video_path}")
        frames, audio = self.extract_video_components(video_path)
        scene_descriptions = self.analyze_scenes(frames)
        emotion_results = self.analyze_emotions(frames)
        audio_result = self.analyze_audio(audio)
        final = self.synthesize_analysis(scene_descriptions, emotion_results, audio_result)
        print("[INFO] Video analysis completed!")
        return final

    def save_results(self, results: Dict[str, Any], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze a dog video for emotion, bark type, and scene understanding.")
    parser.add_argument("video_path", type=str, help="Path to the video file (e.g., dog1.mp4)")
    parser.add_argument("--output", type=str, default="dog_analysis_results.json", help="Path to save analysis results")
    args = parser.parse_args()

    video_path = args.video_path
    output_path = args.output

    analyzer = VideoAnalysisSystem()

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        print("[INFO] Please provide a valid path.")
        return

    try:
        results = analyzer.analyze_video(video_path)
        print("\n" + "=" * 50)
        print("\U0001F415 DOG ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Mood: {results['dog_thoughts']['mood']}")
        print(f"Dominant Emotion: {results['dominant_emotion']}")
        print(f"Audio Behavior (class): {results['audio_behavior']} ({results['audio_behavior_confidence']:.3f})")
        print(f"Audio Phrase: {results['audio_phrase']} ({results['audio_phrase_confidence']:.3f})")
        print(f"Needs Attention: {results['dog_thoughts']['needs_attention']}")
        print(f"Behavioral Interpretation: {results['dog_thoughts']['behavioral_interpretation']}")
        print("\nLikely Thoughts:")
        for thought in results['dog_thoughts']['likely_thoughts']:
            print(f"  - {thought}")

        top_items = results['audio_details']['top_k_phrases']
        if top_items:
            print("\nTop Bark Phrase Matches:")
            for phrase, cls, prob in top_items:
                print(f"  - {phrase}  [{cls}]  prob={prob:.3f}")

        analyzer.save_results(results, output_path)
    except Exception as e:
        print(f"[ERROR] An error occurred during analysis: {e}")
        print("[INFO] Please check your video file and model files")

main()

import gc
for var in list(globals()):
    if not var.startswith("__") and var not in ["torch", "gc"]:
        del globals()[var]
gc.collect()
torch.cuda.empty_cache()