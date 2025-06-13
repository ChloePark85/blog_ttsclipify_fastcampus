import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import json
import hashlib
import os
from dotenv import load_dotenv
import time
import nltk
from nltk.tokenize import sent_tokenize
import random
import html
import cv2
import numpy as np
import moviepy.editor as mpy
import tempfile

st.set_page_config(page_title="블로그 AI 숏폼 생성기", page_icon="✨")

load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def extract_blog_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.naver.com"
    }

    try:
        res = requests.get(url, headers=headers)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        return [], [], [], f"요청 실패: {e}"

    # 네이버 블로그 iframe 확인
    frame_src = None
    frames = soup.find_all("iframe")
    for frame in frames:
        src = frame.get("src", "")
        if "PostView" in src:
            frame_src = "https://blog.naver.com" + src if not src.startswith("http") else src
            break

    # iframe이 있으면 iframe 내용 가져오기
    if frame_src:
        try:
            res = requests.get(frame_src, headers=headers)
            res.encoding = 'utf-8'
            soup = BeautifulSoup(res.text, "html.parser")
        except Exception as e:
            return [], [], [], f"iframe 요청 실패: {e}"

    # 본문 추출용 class 후보 리스트 (최신 및 구버전 모두 포함)
    container_classes = [
        "se-main-container",
        "se_component_wrap",
        "se-publishArea",
        "post_content",
        "postViewArea",
        "post-view",
        "view",
        "post-content",
        "entry-content",
        "se-module-content"
    ]

    container = None
    for cls in container_classes:
        container = soup.find("div", class_=cls)
        if container:
            break

    # 본문을 찾지 못했을 경우 article 태그도 확인
    if container is None:
        container = soup.find("article")

    # 여전히 본문을 찾지 못했을 경우 오류 반환
    if container is None:
        # 디버깅용 HTML 구조 확인
        st.write("HTML 구조 확인 (첫 500자):", soup.prettify()[:500])
        return [], [], [], "❗ 블로그 본문을 찾을 수 없습니다. 다른 글을 시도해 보세요."

    # 순서대로 콘텐츠 추출 (텍스트와 이미지의 원래 순서 유지)
    content_elements = []
    
    # 모든 하위 요소를 순서대로 탐색
    for element in container.find_all(recursive=True):
        # 텍스트 요소 확인
        if element.name in ["p", "span", "div", "h1", "h2", "h3", "h4"]:
            # 특정 클래스를 가진 태그는 건너뛰기 (이미지 컨테이너, 버튼 등)
            skip_classes = ["se-module-image", "se_image", "se-func-button", "se-file-button"]
            if element.get("class") and any(cls in element.get("class", []) for cls in skip_classes):
                continue
                
            text = element.get_text().strip()
            # 의미 있는 텍스트만 추가
            if text and len(text) > 5 and not re.match(r'^[\s\d.,:;!?]+$', text):
                # 중복 텍스트 확인
                if not any(item['type'] == 'text' and item['content'] == text for item in content_elements):
                    content_elements.append({
                        'type': 'text',
                        'content': text,
                        'position': len(content_elements)
                    })
        
        # 이미지 요소 확인
        elif element.name == "img":
            for attr in ["data-lazy-src", "src", "data-src"]:
                src = element.get(attr)
                if src:
                    # 상대 경로인 경우 절대 경로로 변환
                    if src.startswith("//"):
                        src = "https:" + src
                    # http로 시작하고 base64가 아닌 경우에만 추가
                    if src.startswith("http") and not src.startswith("data:"):
                        # 중복 이미지 확인
                        if not any(item['type'] == 'image' and item['content'] == src for item in content_elements):
                            content_elements.append({
                                'type': 'image',
                                'content': src,
                                'position': len(content_elements)
                            })
                    break
    
    # 이미지 컨테이너에서 추가 이미지 찾기 (위에서 놓친 것들)
    for img_container in container.find_all(["div"], class_=lambda c: c and any(x in str(c) for x in ["se-module-image", "se-image-container", "se_image", "__se_module_data"])):
        for img in img_container.find_all("img"):
            for attr in ["data-lazy-src", "src", "data-src"]:
                src = img.get(attr)
                if src:
                    if src.startswith("//"):
                        src = "https:" + src
                    if src.startswith("http") and not src.startswith("data:"):
                        # 중복 이미지 확인
                        if not any(item['type'] == 'image' and item['content'] == src for item in content_elements):
                            content_elements.append({
                                'type': 'image',
                                'content': src,
                                'position': len(content_elements)
                            })
                    break
    
    # 텍스트와 이미지 분리 (기존 호환성 유지)
    texts = [item['content'] for item in content_elements if item['type'] == 'text']
    image_urls = [item['content'] for item in content_elements if item['type'] == 'image']
    
    # 결과가 없을 경우 추가 디버깅 정보
    if not texts and not image_urls:
        st.write("HTML 구조 확인 (첫 500자):", soup.prettify()[:500])
        return [], [], [], "❗ 콘텐츠를 추출할 수 없습니다. URL을 확인해주세요."

    return texts, image_urls, content_elements, None

def download_image(url):
    """이미지 URL에서 이미지를 다운로드하여 base64 인코딩된 문자열로 반환"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://blog.naver.com/",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "sec-fetch-dest": "image",
        "sec-fetch-mode": "no-cors",
        "sec-fetch-site": "cross-site"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            try:
                img = Image.open(BytesIO(response.content))
                # RGBA 이미지를 RGB로 변환
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                # 이미지 포맷 변환 (JPEG로 통일)
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_str}"
            except Exception as e:
                print(f"이미지 처리 오류: {str(e)}")
                return None
        else:
            print(f"HTTP 오류: {response.status_code}")
            return None
    except Exception as e:
        print(f"요청 오류: {str(e)}")
        return None
    
def extract_interesting_pairs(content_elements, max_pairs=10):
    """텍스트와 이미지에서 위치 기반으로 흥미로운 쌍을 추출합니다."""
    
    # 필터링 키워드
    filter_keywords = [
        "게시물이 없습니다", "WRITE", "REVIEW", "댓글", "공감", "스크랩",
        "이웃추가", "맨위로", "목록", "이전", "다음", "HOME", "MENU"
    ]
    
    # 텍스트와 이미지 요소 분리 (위치 정보 유지)
    text_elements = []
    image_elements = []
    
    for element in content_elements:
        if element['type'] == 'text':
            text = element['content']
            
            # 키워드 필터링
            if any(keyword in text for keyword in filter_keywords):
                continue
                
            # 한국어 비율 체크 (50% 이상)
            korean_chars = len(re.findall(r'[가-힣]', text))
            if len(text) > 0 and korean_chars / len(text) < 0.5:
                continue
                
            # 길이 체크 (15-200자)
            if not (15 <= len(text) <= 200):
                continue
                
            # 공백 및 특수문자 비율 체크
            spaces = text.count(' ')
            special_chars = len(re.findall(r'[^\w\s가-힣]', text))
            if len(text) > 0 and (spaces / len(text) > 0.3 or special_chars / len(text) > 0.3):
                continue
                
            text_elements.append({
                'content': text,
                'position': element['position']
            })
        elif element['type'] == 'image':
            image_elements.append({
                'content': element['content'],
                'position': element['position']
            })
    
    # 텍스트 문장 분리
    sentence_elements = []
    for text_elem in text_elements:
        try:
            nltk.download('punkt', quiet=True)
            text_sentences = sent_tokenize(text_elem['content'])
        except:
            # NLTK 실패시 정규식 사용
            text_sentences = re.split(r'[.!?]+', text_elem['content'])
            text_sentences = [s.strip() for s in text_sentences if s.strip()]
        
        # 적절한 길이의 문장만 선택
        for sentence in text_sentences:
            if 20 <= len(sentence) <= 100:
                sentence_elements.append({
                    'content': sentence,
                    'position': text_elem['position']
                })
    
    # 위치 기반 매칭
    pairs = []
    used_positions = set()
    
    # 각 이미지에 대해 가장 가까운 문장 찾기
    for img_elem in image_elements:
        if len(pairs) >= max_pairs:
            break
            
        # 이미 사용된 위치의 텍스트는 제외
        available_sentences = [s for s in sentence_elements 
                            if s['position'] not in used_positions]
        
        if not available_sentences:
            continue
        
        # 가장 가까운 문장 찾기
        closest_sentence = min(available_sentences, 
                             key=lambda x: abs(x['position'] - img_elem['position']))
        
        pairs.append({
            'text': closest_sentence['content'],
            'image_url': img_elem['content']
        })
        used_positions.add(closest_sentence['position'])
    
    return pairs

def text_to_speech(text, voice_id="uyVNoMrnUku1dZyVEXwD", model_id="eleven_multilingual_v2"):
    """텍스트를 음성으로 변환합니다."""
    try:
        if not ELEVENLABS_API_KEY:
            st.warning("ElevenLabs API 키가 설정되지 않았습니다.")
            return None, {"duration": 10}

        # API 엔드포인트
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        # 요청 헤더
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }

        # 요청 데이터
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        # API 요청
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            # 오디오 데이터를 base64로 인코딩
            audio_base64 = base64.b64encode(response.content).decode()
            return audio_base64, {"duration": 10}  # 임시로 duration 10초 설정
        else:
            st.error(f"TTS API 오류: {response.status_code}")
            return None, {"duration": 10}

    except Exception as e:
        st.error(f"TTS 변환 실패: {str(e)}")
        return None, {"duration": 10}

def create_video_from_pairs(pairs, output_path='output.mp4'):
    """
    흥미로운 부분 쌍(이미지, 오디오, 텍스트) 리스트를 받아 9:16 비율 영상으로 생성합니다.
    - 이미지가 9:16이 아니면 검은 배경에 중앙 배치
    - 텍스트(한글) 자막 삽입
    - 오디오와 싱크 맞춤
    - 모든 쌍을 이어붙여 하나의 영상으로 만듦
    """
    W, H = 720, 1280  # 9:16 비율
    # macOS 한글 폰트 경로 최우선 추가
    font_candidates = [
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',  # macOS 기본 한글 폰트
        '/Library/Fonts/AppleGothic.ttf',
        '/Users/chloepark/Library/Fonts/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
        '/usr/share/fonts/truetype/malgun/MalgunGothic.ttf',
        '/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        './NanumGothic.ttf'
    ]
    font_path = None
    for path in font_candidates:
        if os.path.exists(path):
            font_path = path
            break
    font_size = 40
    clips = []
    audio_clips = []
    audio_paths = []

    # 1차 패스: 각 이미지별로 줄바꿈이 가능한 최소 폰트 크기 계산
    min_font_size_global = font_size
    all_lines = []
    font_sizes = []
    for idx, pair in enumerate(pairs):
        text = pair['text']
        min_font_size_local = font_size
        lines = []
        for font_size in range(font_size, 24 - 1, -2):
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                bbox = ImageDraw.Draw(Image.new('RGB', (W, H))).textbbox((0, 0), test_line, font=font)
                text_w = bbox[2] - bbox[0]
                if text_w > W - 2 * 40:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
            if len(lines) <= 3:
                min_font_size_local = font_size
                break
        all_lines.append(lines)
        font_sizes.append(min_font_size_local)
        if min_font_size_local < min_font_size_global:
            min_font_size_global = min_font_size_local

    # 2차 패스: 실제 자막 그리기(모든 이미지에 대해 min_font_size_global로 고정)
    for idx, pair in enumerate(pairs):
        # 1. 이미지 다운로드 및 9:16 비율 맞추기
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Referer": "https://blog.naver.com/",
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
                "Cache-Control": "no-cache",
                "sec-fetch-dest": "image",
                "sec-fetch-mode": "no-cors",
                "sec-fetch-site": "cross-site"
            }
            response = requests.get(pair['image_url'], headers=headers, timeout=10, allow_redirects=True)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            # 실패 시 검은 배경
            img = Image.new('RGB', (W, H), (0, 0, 0))

        # 이미지 리사이즈 및 중앙 배치
        img_ratio = img.width / img.height
        target_ratio = W / H
        if abs(img_ratio - target_ratio) > 0.01:
            # 검은 배경 생성
            bg = Image.new('RGB', (W, H), (0, 0, 0))
            # 이미지 크기 조정 (긴 쪽을 맞춤)
            if img_ratio > target_ratio:
                # 이미지가 더 넓음
                new_w = W
                new_h = int(W / img_ratio)
            else:
                # 이미지가 더 높음
                new_h = H
                new_w = int(H * img_ratio)
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            # 중앙 배치
            x = (W - new_w) // 2
            y = (H - new_h) // 2
            bg.paste(img_resized, (x, y))
            img = bg
        else:
            img = img.resize((W, H), Image.LANCZOS)

        # 2. 텍스트 자막 합성 (하단 1/3 지점, 크게, 그림자, 자동 줄바꿈)
        draw = ImageDraw.Draw(img)
        if font_path:
            font = ImageFont.truetype(font_path, min_font_size_global)
        else:
            font = ImageFont.load_default()
            print("❌ 한글 폰트를 찾을 수 없습니다. 반드시 '/System/Library/Fonts/AppleSDGothicNeo.ttc' 또는 한글 지원 폰트를 설치하세요!")
        text = pair['text']
        # 동일한 줄바꿈 로직 적용
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w > W - 2 * 40:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)
        lines = lines[:3]
        # 전체 자막 높이 계산
        line_heights = [draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines]
        total_text_height = sum(line_heights) + (len(lines) - 1) * 10
        y_pos = H - (H // 3) - (total_text_height // 2)
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x_pos = (W - text_w) // 2
            y_line = y_pos + sum(line_heights[:i]) + i * 10
            shadow_offsets = [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 2), (0, -2), (2, 0), (-2, 0)]
            for ox, oy in shadow_offsets:
                draw.text((x_pos + ox, y_line + oy), line, font=font, fill=(0, 0, 0))
            draw.text((x_pos, y_line), line, font=font, fill=(255, 255, 255))

        # 3. 오디오(base64) -> 임시 파일로 저장
        audio_data = pair.get('audio_data')
        if not audio_data:
            continue  # 오디오 없으면 스킵
        audio_bytes = base64.b64decode(audio_data)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as audio_file:
            audio_file.write(audio_bytes)
            audio_path = audio_file.name
        audio_clip = mpy.AudioFileClip(audio_path)
        duration = audio_clip.duration
        frame = np.array(img)
        video_clip = mpy.ImageClip(frame).set_duration(duration).set_audio(audio_clip)
        clips.append(video_clip)
        audio_clips.append(audio_clip)
        audio_paths.append(audio_path)

    if not clips:
        return None

    # 5. 모든 쌍의 영상 클립을 이어붙여 하나의 영상으로 합침
    final_clip = mpy.concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')
    final_clip.close()

    # 오디오 클립 및 임시 파일 정리 (write_videofile 이후)
    for ac in audio_clips:
        ac.close()
    for ap in audio_paths:
        try:
            os.remove(ap)
        except Exception:
            pass

    return output_path

def main():
    """메인 Streamlit 앱"""
    st.title("블로그 크롤링 테스트")
    st.markdown("네이버 블로그 글을 입력하면 텍스트와 이미지를 추출해드립니다!")
    
    # 세션 상태 초기화
    if 'texts' not in st.session_state:
        st.session_state.texts = None
    if 'image_urls' not in st.session_state:
        st.session_state.image_urls = None
    if 'content_elements' not in st.session_state:
        st.session_state.content_elements = None
    if 'show_interesting' not in st.session_state:
        st.session_state.show_interesting = False
    if 'error' not in st.session_state:
        st.session_state.error = None
    
    # 메인 인터페이스
    blog_url = st.text_input(
        "📝 블로그 URL을 입력하세요:",
        placeholder="https://blog.naver.com/example/123456789",
        help="네이버 블로그 포스트의 전체 URL을 입력해주세요."
    )
    
    if st.button("🚀 크롤링 시작", type="primary"):
        if not blog_url:
            st.error("블로그 URL을 입력해주세요!")
            return
        
        if "blog.naver.com" not in blog_url:
            st.error("현재는 네이버 블로그만 지원됩니다!")
            return
        
        # 크롤링 진행
        with st.spinner("블로그 내용을 분석하고 있습니다..."):
            texts, image_urls, content_elements, error = extract_blog_content(blog_url)
            
            # 세션 상태에 저장
            st.session_state.texts = texts
            st.session_state.image_urls = image_urls
            st.session_state.content_elements = content_elements
            st.session_state.error = error
            st.session_state.show_interesting = False  # 크롤링 시 흥미로운 부분 표시 초기화
    
    # 크롤링 결과가 있을 때만 표시
    if st.session_state.texts is not None and st.session_state.image_urls is not None:
        if st.session_state.error:
            st.error(f"오류 발생: {st.session_state.error}")
            return
        
        if not st.session_state.texts and not st.session_state.image_urls:
            st.warning("추출된 콘텐츠가 없습니다. 다른 블로그 글을 시도해보세요.")
            return
        
        # 결과 표시
        st.success(f"✅ 크롤링 완료! 텍스트 {len(st.session_state.texts)}개, 이미지 {len(st.session_state.image_urls)}개를 찾았습니다.")
        
        # 추출된 내용 미리보기
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 추출된 텍스트")
            for i, text in enumerate(st.session_state.texts):
                st.write(f"**텍스트 {i+1}:**")
                st.write(text)
                st.write("---")  # 구분선
        
        with col2:
            st.subheader("🖼️ 추출된 이미지")
            for i, img_url in enumerate(st.session_state.image_urls):
                st.write(f"**이미지 {i+1}:**")
                
                try:
                    # 네이버 블로그 이미지용 특별 헤더
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        "Referer": "https://blog.naver.com/",
                        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
                        "Cache-Control": "no-cache",
                        "sec-fetch-dest": "image",
                        "sec-fetch-mode": "no-cors",
                        "sec-fetch-site": "cross-site"
                    }
                    
                    # 이미지 요청
                    response = requests.get(img_url, headers=headers, timeout=10, allow_redirects=True)
                    
                    if response.status_code == 200:
                        img_data = BytesIO(response.content)
                        img = Image.open(img_data)
                        
                        # RGBA 이미지를 RGB로 변환
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                            
                        # 이미지를 메모리에 저장
                        img_buffer = BytesIO()
                        img.save(img_buffer, format='JPEG', quality=85)
                        img_buffer.seek(0)
                        
                        # Streamlit에 이미지 표시
                        st.image(img_buffer, width=300, caption=f"이미지 {i+1}")
                    else:
                        st.error(f"❌ HTTP 오류 ({response.status_code}): 이미지를 불러올 수 없습니다")
                        st.write(f"URL: {img_url}")
                except Exception as e:
                    st.error(f"❌ 이미지 로딩 실패: {str(e)}")
                    st.write(f"URL: {img_url}")
                st.write("---")
        
        # 흥미로운 부분 추출 버튼 (크롤링 결과 아래에 배치)
        if st.button("✨ 흥미로운 부분 추출", type="secondary"):
            st.session_state.show_interesting = True
        
        # 흥미로운 부분 표시 (세션 상태에 따라)
        if st.session_state.show_interesting:
            st.subheader("🎯 흥미로운 부분")
            interesting_pairs = extract_interesting_pairs(st.session_state.content_elements)

            # 각 쌍에 대해 TTS(audio_data) 미리 생성 및 저장
            for pair in interesting_pairs:
                if 'audio_data' not in pair or not pair.get('audio_data'):
                    audio_data, audio_info = text_to_speech(pair['text'])
                    pair['audio_data'] = audio_data

            # 음성 생성 상태 표시
            audio_status = st.empty()
            
            for i, pair in enumerate(interesting_pairs, 1):
                with st.container():
                    st.markdown(f"**📝 콘텐츠 {i}**")
                    st.write(pair['text'])
                    
                    # 음성 생성 중 표시
                    audio_status.info("🎵 음성을 생성하고 있습니다...")
                    
                    # TTS 생성 (이미 위에서 생성됨)
                    audio_data = pair.get('audio_data')
                    if audio_data:
                        # 음성 재생 컨트롤 표시
                        st.audio(f"data:audio/mpeg;base64,{audio_data}", format='audio/mp3')
                        audio_status.success("✅ 음성 생성 완료!")
                    else:
                        audio_status.error("❌ 음성 생성 실패")
                    
                    # 이미지 표시
                    try:
                        # 네이버 블로그 이미지용 특별 헤더
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                            "Referer": "https://blog.naver.com/",
                            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
                            "Cache-Control": "no-cache",
                            "sec-fetch-dest": "image",
                            "sec-fetch-mode": "no-cors",
                            "sec-fetch-site": "cross-site"
                        }
                        
                        # 이미지 요청
                        response = requests.get(pair['image_url'], headers=headers, timeout=10, allow_redirects=True)
                        
                        if response.status_code == 200:
                            img_data = BytesIO(response.content)
                            img = Image.open(img_data)
                            
                            # RGBA 이미지를 RGB로 변환
                            if img.mode == 'RGBA':
                                img = img.convert('RGB')
                                
                            # 이미지를 메모리에 저장
                            img_buffer = BytesIO()
                            img.save(img_buffer, format='JPEG', quality=85)
                            img_buffer.seek(0)
                            
                            # Streamlit에 이미지 표시
                            st.image(img_buffer, width=300, caption=f"이미지 {i}")
                        else:
                            st.error(f"❌ HTTP 오류 ({response.status_code}): 이미지를 불러올 수 없습니다")
                            st.write(f"URL: {pair['image_url']}")
                    except Exception as e:
                        st.error(f"❌ 이미지 로딩 실패: {str(e)}")
                        st.write(f"URL: {pair['image_url']}")
                    st.write("---")

            # 비디오 생성 버튼 추가
            if st.button("🎬 비디오 생성", type="primary"):
                with st.spinner("비디오를 생성 중입니다..."):
                    video_pairs = [p for p in interesting_pairs if p.get('audio_data')]
                    if not video_pairs:
                        st.error("오디오가 포함된 쌍이 없습니다. 비디오를 생성할 수 없습니다.")
                    else:
                        output_path = "output.mp4"
                        result = create_video_from_pairs(video_pairs, output_path=output_path)
                        if result:
                            with open(output_path, "rb") as f:
                                video_bytes = f.read()
                            st.session_state['video_bytes'] = video_bytes
                            st.success("비디오 생성이 완료되었습니다!")
                        else:
                            st.error("비디오 생성에 실패했습니다.")

            # 비디오가 세션에 있으면 항상 표시
            if 'video_bytes' in st.session_state and st.session_state['video_bytes']:
                st.video(st.session_state['video_bytes'])
                st.download_button(
                    label="비디오 다운로드",
                    data=st.session_state['video_bytes'],
                    file_name="output.mp4",
                    mime="video/mp4"
                )

if __name__ == "__main__":
    main()