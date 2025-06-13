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
        "Referer": "https://blog.naver.com/"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            try:
                img = Image.open(BytesIO(response.content))
                # 이미지 포맷 변환 (PNG로 통일)
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return f"data:image/png;base64,{img_str}"
            except Exception:
                return None
        else:
            return None
    except Exception:
        return None
    
def main():
    """메인 Streamlit 앱"""
    st.title("블로그 크롤링 테스트")
    st.markdown("네이버 블로그 글을 입력하면 텍스트와 이미지를 추출해드립니다!")
    
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
        
        if error:
            st.error(f"오류 발생: {error}")
            return
        
        if not texts and not image_urls:
            st.warning("추출된 콘텐츠가 없습니다. 다른 블로그 글을 시도해보세요.")
            return
        
        # 결과 표시
        st.success(f"✅ 크롤링 완료! 텍스트 {len(texts)}개, 이미지 {len(image_urls)}개를 찾았습니다.")
        
        # 추출된 내용 미리보기
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 추출된 텍스트")
            for i, text in enumerate(texts):
                st.write(f"**텍스트 {i+1}:**")
                st.write(text)
                st.write("---")  # 구분선
        
        with col2:
            st.subheader("🖼️ 추출된 이미지")
            for i, img_url in enumerate(image_urls):
                st.write(f"**이미지 {i+1}:**")
                
                # 이미지 URL 표시
                st.write(f"URL: {img_url}")
                
                # 이미지 로딩 시도
                try:
                    # 네이버 블로그 이미지용 특별 헤더
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Referer': 'https://blog.naver.com/',
                        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                        'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
                        'Cache-Control': 'no-cache'
                    }
                    
                    # 이미지 요청
                    response = requests.get(img_url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        # PIL로 이미지 열기
                        img = Image.open(BytesIO(response.content))
                        st.image(img, width=300, caption=f"이미지 {i+1}")
                        st.success("✅ 이미지 로딩 성공")
                    else:
                        st.error(f"❌ HTTP 오류: {response.status_code}")
                        # 그래도 Streamlit 기본 방식으로 시도
                        st.image(img_url, width=300, caption=f"이미지 {i+1} (직접 로딩)")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ 네트워크 오류: {str(e)}")
                    # Streamlit 기본 방식으로 시도
                    try:
                        st.image(img_url, width=300, caption=f"이미지 {i+1} (직접 로딩)")
                    except:
                        st.error("❌ 이미지 로딩 완전 실패")
                        
                except Exception as e:
                    st.error(f"❌ 이미지 처리 오류: {str(e)}")
                    # Streamlit 기본 방식으로 시도
                    try:
                        st.image(img_url, width=300, caption=f"이미지 {i+1} (직접 로딩)")
                    except:
                        st.error("❌ 이미지 로딩 완전 실패")
                
                st.write("---")  # 구분선

if __name__ == "__main__":
    main()