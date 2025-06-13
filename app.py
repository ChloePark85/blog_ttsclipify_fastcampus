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
            
            # 음성 생성 상태 표시
            audio_status = st.empty()
            
            for i, pair in enumerate(interesting_pairs, 1):
                with st.container():
                    st.markdown(f"**📝 콘텐츠 {i}**")
                    st.write(pair['text'])
                    
                    # 음성 생성 중 표시
                    audio_status.info("🎵 음성을 생성하고 있습니다...")
                    
                    # TTS 생성
                    audio_data, audio_info = text_to_speech(pair['text'])
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

if __name__ == "__main__":
    main()