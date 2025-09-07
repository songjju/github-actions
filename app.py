#!/usr/bin/env python3
"""
Flask 로또 예측 웹 애플리케이션
기존 main.py 시스템을 웹 인터페이스로 감싸는 래퍼
- 기능 1: 예측 번호 생성 버튼 (기존 main.py 활용)
- 기능 2: 이전 예측 기록 조회
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for, send_file, send_from_directory
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import random
import subprocess
import sys
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import re
import zipfile
import tempfile
import glob

# Flask 앱 초기화
app = Flask(__name__)
app.secret_key = 'lotto_prediction_secret_key_2025'

# 설정
MAIN_SCRIPT_PATH = 'main.py'  # 기존 main.py 경로
PREDICTIONS_HISTORY_PATH = 'outputs/predictions/prediction_history.txt'
PREDICTIONS_JSON_PATH = 'outputs/predictions/latest_predictions.json'
CSV_DATA_PATH = 'data/raw/lotto_results.csv'

DEBUG_MODE = os.getenv('LOTTO_DEBUG', 'false').lower() == 'true'
SHOW_SENSITIVE_INFO = os.getenv('LOTTO_SHOW_SENSITIVE', 'false').lower() == 'true'

VISUALIZATIONS_DIR = 'outputs/visualizations'

class FlaskLottoWrapper:
    """기존 main.py 시스템을 Flask로 감싸는 래퍼 클래스"""
    
    def __init__(self):
        self.ensure_directories()
    
    def ensure_directories(self):
        """필요한 디렉토리 생성"""
        dirs = ['outputs/predictions', 'outputs/reports', 'logs', 'data/raw']
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run_main_prediction(self, num_predictions=5):
        """기존 main.py 실행하여 예측 생성"""
        try:
            # main.py가 존재하는지 확인
            if not os.path.exists(MAIN_SCRIPT_PATH):
                return {"success": False, "error": "main.py 파일을 찾을 수 없습니다."}
            
            # main.py 실행
            cmd = [sys.executable, MAIN_SCRIPT_PATH, '--predictions', str(num_predictions)]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60,  # 60초 타임아웃
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                # 성공적으로 실행됨
                return {
                    "success": True, 
                    "output": result.stdout,
                    "predictions": self.parse_predictions_from_output(result.stdout)
                }
            else:
                # 실행 실패
                return {
                    "success": False, 
                    "error": f"main.py 실행 실패: {result.stderr}"
                }
        
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "예측 생성 시간이 초과되었습니다."}
        except Exception as e:
            return {"success": False, "error": f"예측 실행 중 오류 발생: {str(e)}"}
    
    def parse_predictions_from_output(self, output):
        """main.py 출력에서 예측 번호 파싱"""
        try:
            predictions = []
            
            # 실제 출력 형식에 맞는 패턴으로 수정
            # 예: "예측 1: [2, 3, 8, 14, 17, 18] (신뢰도: 0.492)"
            pattern = r'예측\s*(\d+):\s*\[([0-9, ]+)\]\s*\(신뢰도:\s*([\d.]+)\)'
            matches = re.findall(pattern, output)
            
            print(f"파싱 시도: {len(matches)}개 매치 발견")  # 디버그 출력
            
            for match in matches:
                prediction_id, numbers_str, confidence = match
                print(f"매치 {prediction_id}: numbers='{numbers_str}', confidence='{confidence}'")  # 디버그
                
                # 숫자 파싱 (공백과 쉼표로 분리)
                numbers = []
                for num_str in numbers_str.split(','):
                    try:
                        num = int(num_str.strip())
                        if 1 <= num <= 45:  # 유효한 로또 번호 범위
                            numbers.append(num)
                    except ValueError:
                        continue
                
                if len(numbers) == 6:  # 6개 번호가 정상적으로 파싱된 경우만
                    predictions.append({
                        'id': int(prediction_id),
                        'numbers': sorted(numbers),
                        'confidence': float(confidence),
                        'timestamp': datetime.now().isoformat(),
                        'method': 'genius_insight',
                        'explanation': '천재적 통찰 공식을 활용한 다차원적 분석'
                    })
                    print(f"예측 {prediction_id} 파싱 성공: {sorted(numbers)}")  # 디버그
                else:
                    print(f"예측 {prediction_id} 파싱 실패: 번호 개수 {len(numbers)}")  # 디버그
            
            # 파싱된 예측이 없으면 기본 예측 생성
            if not predictions:
                print("파싱 실패 - 기본 예측 생성")  # 디버그
                predictions = self.generate_fallback_predictions()
            
            print(f"최종 파싱 결과: {len(predictions)}개 예측")  # 디버그
            return predictions
            
        except Exception as e:
            print(f"예측 파싱 오류: {e}")
            return self.generate_fallback_predictions()
    
    def generate_fallback_predictions(self):
        """main.py 파싱 실패시 기본 예측 생성"""
        import random
        predictions = []
        
        for i in range(5):
            numbers = sorted(random.sample(range(1, 46), 6))
            predictions.append({
                'id': i + 1,
                'numbers': numbers,
                'confidence': random.uniform(0.4, 0.8),
                'timestamp': datetime.now().isoformat(),
                'method': 'fallback_random'
            })
        
        return predictions
    
    def load_prediction_history(self):
        """예측 기록 로딩 (텍스트 파일에서)"""
        try:
            if not os.path.exists(PREDICTIONS_HISTORY_PATH):
                return []
            
            history = []
            with open(PREDICTIONS_HISTORY_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                for line in lines[1:]:  # 헤더 스킵
                    line = line.strip()
                    if '|' in line:
                        parts = [part.strip() for part in line.split('|')]
                        if len(parts) >= 4:
                            date, pred_id, numbers_str, confidence_str = parts[:4]
                            
                            # 번호 파싱
                            numbers = [int(n.strip()) for n in numbers_str.split(',')]
                            
                            # 신뢰도 파싱
                            confidence = float(confidence_str.replace('%', '')) / 100.0
                            
                            history.append({
                                'date': date,
                                'id': pred_id,
                                'numbers': numbers,
                                'confidence': confidence,
                                'timestamp': f"{date} 00:00:00"
                            })
            
            return history[::-1]  # 최신순 정렬
            
        except Exception as e:
            print(f"히스토리 로딩 오류: {e}")
            return []
    
    def get_basic_stats(self):
        """기본 통계 정보 반환"""
        try:
            # CSV 파일이 있으면 로드
            if os.path.exists(CSV_DATA_PATH):
                df = pd.read_csv(CSV_DATA_PATH, encoding='utf-8')
                
                # 번호 컬럼 찾기
                number_cols = [col for col in df.columns if col.startswith('num')]
                
                if number_cols:
                    # 빈도 계산
                    all_numbers = []
                    for col in number_cols:
                        all_numbers.extend(df[col].dropna().tolist())
                    
                    frequency = {}
                    for i in range(1, 46):
                        frequency[i] = all_numbers.count(i)
                    
                    return {
                        'total_draws': len(df),
                        'frequency': frequency,
                        'most_frequent': max(frequency, key=frequency.get),
                        'least_frequent': min(frequency, key=frequency.get),
                        'avg_frequency': sum(frequency.values()) / len(frequency)
                    }
            
            # 기본 통계
            return {
                'total_draws': 0,
                'frequency': {i: 10 for i in range(1, 46)},
                'most_frequent': 7,
                'least_frequent': 13,
                'avg_frequency': 10
            }
            
        except Exception as e:
            print(f"통계 로딩 오류: {e}")
            return {
                'total_draws': 0,
                'frequency': {i: 10 for i in range(1, 46)},
                'most_frequent': 7,
                'least_frequent': 13,
                'avg_frequency': 10
            }

class Zodiac(Enum):
    ARIES = "aries"
    TAURUS = "taurus"
    GEMINI = "gemini"
    CANCER = "cancer"
    LEO = "leo"
    VIRGO = "virgo"
    LIBRA = "libra"
    SCORPIO = "scorpio"
    SAGITTARIUS = "sagittarius"
    CAPRICORN = "capricorn"
    AQUARIUS = "aquarius"
    PISCES = "pisces"

@dataclass
class LuckyElements:
    objects: List[str]
    numbers: List[int]
    colors: List[str]

class ZodiacLuckyManager:
    """별자리별 행운 요소 관리 클래스"""
    
    def __init__(self):
        self.zodiac_elements = {
            Zodiac.ARIES: LuckyElements(
                objects=["반지", "칼", "스포츠용품", "빨간 장미", "금속 액세서리"],
                numbers=[1, 9, 17, 25, 33],
                colors=["빨강", "주황", "황금", "진홍", "오렌지"]
            ),
            Zodiac.TAURUS: LuckyElements(
                objects=["목걸이", "화분", "쿠션", "향수", "예술품"],
                numbers=[2, 6, 14, 22, 30],
                colors=["초록", "분홍", "갈색", "연두", "에메랄드"]
            ),
            Zodiac.GEMINI: LuckyElements(
                objects=["책", "스마트폰", "펜", "거울", "열쇠"],
                numbers=[3, 12, 21, 39, 45],
                colors=["노랑", "은색", "하늘색", "라임", "레몬"]
            ),
            Zodiac.CANCER: LuckyElements(
                objects=["진주", "조개껍데기", "달 모양 장식", "물병", "가족사진"],
                numbers=[4, 7, 16, 25, 34],
                colors=["은색", "흰색", "연보라", "진주색", "달빛색"]
            ),
            Zodiac.LEO: LuckyElements(
                objects=["왕관", "해바라기", "황금 장신구", "태양 모양 장식", "다이아몬드"],
                numbers=[5, 19, 28, 37, 44],
                colors=["황금", "주황", "노랑", "골드", "황갈색"]
            ),
            Zodiac.VIRGO: LuckyElements(
                objects=["시계", "노트", "청소용품", "허브", "수정"],
                numbers=[6, 15, 24, 33, 42],
                colors=["베이지", "갈색", "연녹색", "아이보리", "모래색"]
            ),
            Zodiac.LIBRA: LuckyElements(
                objects=["저울", "꽃다발", "예술품", "향초", "실크 스카프"],
                numbers=[7, 16, 25, 34, 43],
                colors=["파스텔 블루", "분홍", "연보라", "라벤더", "하늘색"]
            ),
            Zodiac.SCORPIO: LuckyElements(
                objects=["가넷", "전갈 모양 장식", "검은 장갑", "신비로운 책", "붉은 와인"],
                numbers=[8, 13, 22, 31, 40],
                colors=["검정", "진빨강", "보라", "자주", "진홍"]
            ),
            Zodiac.SAGITTARIUS: LuckyElements(
                objects=["활", "지구본", "여행가방", "말 모양 장식", "나침반"],
                numbers=[9, 18, 27, 36, 45],
                colors=["보라", "터키옥", "남색", "진파랑", "로얄블루"]
            ),
            Zodiac.CAPRICORN: LuckyElements(
                objects=["시계", "산양 모양 장식", "가죽 지갑", "명함집", "펜"],
                numbers=[10, 19, 28, 37, 44],
                colors=["검정", "갈색", "회색", "다크 그린", "네이비"]
            ),
            Zodiac.AQUARIUS: LuckyElements(
                objects=["물병", "전자기기", "하늘색 구슬", "번개 모양 장식", "수정구"],
                numbers=[11, 20, 29, 38, 44],
                colors=["하늘색", "전기 파랑", "네온 색상", "터키옥", "청록"]
            ),
            Zodiac.PISCES: LuckyElements(
                objects=["물고기 모양 장식", "바다 조개", "수정구", "물 관련 장식", "비취"],
                numbers=[12, 21, 30, 39, 45],
                colors=["바다색", "연녹색", "청록", "라벤더", "아쿠아마린"]
            )
        }
        
        self.color_number_mapping = {
            "빨강": [1, 9, 17, 25, 33, 41],
            "주황": [2, 10, 18, 26, 34, 42],
            "노랑": [3, 11, 19, 27, 35, 43],
            "초록": [4, 12, 20, 28, 36, 44],
            "파랑": [5, 13, 21, 29, 37, 45],
            "보라": [6, 18, 30, 42],
            "분홍": [7, 14, 21, 28, 35],
            "하늘색": [3, 15, 27, 39],
            "검정": [8, 16, 24, 32, 40],
            "흰색": [7, 14, 21, 28, 35, 42],
            "회색": [11, 22, 33, 44],
            "은색": [2, 12, 22, 32, 42]
        }
        
        self.object_number_mapping = {
            "반지": [1, 10, 19, 28],
            "목걸이": [3, 12, 21, 30, 39],
            "왕관": [7, 17, 27, 37],
            "시계": [12, 24, 36],
            "책": [3, 13, 23, 33],
            "거울": [2, 11, 22, 44],
            "해바라기": [5, 15, 25, 35],
            "진주": [6, 16, 26, 36],
            "수정": [4, 14, 24, 34, 44],
            "활": [9, 18, 27, 36, 45],
            "저울": [7, 14, 21, 28, 35, 42],
            "물병": [11, 22, 33, 44],
            "물고기": [2, 12, 20, 29]
        }
    
    def get_zodiac_prediction(self, zodiac_str: str, user_objects: List[str],
                            user_numbers: List[int], user_colors: List[str],
                            base_prediction: List[int] = None) -> Tuple[List[int], Dict]:
        """별자리 기반 예측 생성"""
        
        try:
            zodiac = Zodiac(zodiac_str.lower())
        except ValueError:
            zodiac = Zodiac.ARIES
        
        # 기본 예측이 없으면 생성
        if base_prediction is None:
            base_prediction = random.sample(range(1, 46), 6)
        
        # 별자리 요소들
        zodiac_elements = self.zodiac_elements[zodiac]
        
        # 행운 번호 후보 수집
        lucky_candidates = set(zodiac_elements.numbers)
        
        # 사용자 입력 추가
        valid_user_numbers = [n for n in user_numbers if 1 <= n <= 45]
        lucky_candidates.update(valid_user_numbers)
        
        # 물건에서 연상되는 번호
        for obj in user_objects:
            if obj in self.object_number_mapping:
                lucky_candidates.update(self.object_number_mapping[obj])
        
        # 색상에서 연상되는 번호
        for color in user_colors:
            if color in self.color_number_mapping:
                lucky_candidates.update(self.color_number_mapping[color])
        
        # 최종 예측 생성
        final_prediction = self._create_final_prediction(
            base_prediction, list(lucky_candidates), valid_user_numbers
        )
        
        # 메타데이터
        metadata = {
            'zodiac': self._get_zodiac_korean_name(zodiac),
            'lucky_story': self._generate_lucky_story(
                zodiac, user_objects, valid_user_numbers, user_colors, final_prediction
            ),
            'confidence': self._calculate_confidence(user_objects, valid_user_numbers, user_colors),
            'element_analysis': self._analyze_elements(
                zodiac, user_objects, valid_user_numbers, user_colors, final_prediction
            )
        }
        
        return final_prediction, metadata
    
    def _create_final_prediction(self, base_prediction: List[int],
                               lucky_candidates: List[int],
                               user_direct_numbers: List[int]) -> List[int]:
        """최종 예측 번호 생성"""
        final = []
        
        # 사용자 직접 입력 우선 (최대 2개)
        direct_include = user_direct_numbers[:2]
        final.extend(direct_include)
        
        # 행운 후보에서 추가
        remaining_lucky = [n for n in lucky_candidates if n not in final]
        if remaining_lucky:
            additional_count = min(3, len(remaining_lucky))
            additional = random.sample(remaining_lucky, additional_count)
            final.extend(additional)
        
        # 부족한 경우 채우기
        while len(final) < 6:
            candidates = [n for n in range(1, 46) if n not in final]
            if candidates:
                final.append(random.choice(candidates))
        
        return sorted(final[:6])
    
    def _get_zodiac_korean_name(self, zodiac: Zodiac) -> str:
        """별자리 한국어 이름"""
        names = {
            Zodiac.ARIES: "양자리", Zodiac.TAURUS: "황소자리",
            Zodiac.GEMINI: "쌍둥이자리", Zodiac.CANCER: "게자리",
            Zodiac.LEO: "사자자리", Zodiac.VIRGO: "처녀자리",
            Zodiac.LIBRA: "천칭자리", Zodiac.SCORPIO: "전갈자리",
            Zodiac.SAGITTARIUS: "사수자리", Zodiac.CAPRICORN: "염소자리",
            Zodiac.AQUARIUS: "물병자리", Zodiac.PISCES: "물고기자리"
        }
        return names.get(zodiac, "알 수 없음")
    
    def _generate_lucky_story(self, zodiac: Zodiac, objects: List[str],
                            numbers: List[int], colors: List[str],
                            prediction: List[int]) -> str:
        """행운의 스토리 생성"""
        zodiac_name = self._get_zodiac_korean_name(zodiac)
        
        stories = [f"{zodiac_name}의 우주적 에너지가 번호 선택에 강한 영향을 미쳤습니다."]
        
        if objects:
            stories.append(f"선택하신 행운의 물건들({', '.join(objects[:2])})이 특별한 의미를 더했습니다.")
        
        if numbers:
            direct_influence = len(set(numbers) & set(prediction))
            if direct_influence > 0:
                stories.append(f"직접 선택하신 숫자 중 {direct_influence}개가 최종 예측에 포함되었습니다.")
        
        if colors:
            stories.append(f"행운의 색상들({', '.join(colors[:2])})이 에너지적 조화를 이뤘습니다.")
        
        return " ".join(stories)
    
    def _calculate_confidence(self, objects: List[str], numbers: List[int], colors: List[str]) -> float:
        """신뢰도 계산"""
        base_confidence = 0.75
        
        element_bonus = len(objects) * 0.02 + len(numbers) * 0.03 + len(colors) * 0.02
        element_types = sum([1 if objects else 0, 1 if numbers else 0, 1 if colors else 0])
        diversity_bonus = element_types * 0.05
        
        return min(0.95, base_confidence + element_bonus + diversity_bonus)
    
    def _analyze_elements(self, zodiac: Zodiac, objects: List[str],
                        numbers: List[int], colors: List[str],
                        prediction: List[int]) -> Dict:
        """요소별 분석"""
        return {
            'zodiac_influence': f"{self._get_zodiac_korean_name(zodiac)}의 영향",
            'object_count': len(objects),
            'number_count': len(numbers),
            'color_count': len(colors),
            'total_elements': len(objects) + len(numbers) + len(colors)
        }
    
# 전역 래퍼 인스턴스
lotto_wrapper = FlaskLottoWrapper()
zodiac_manager = None

def get_zodiac_manager():
    """별자리 매니저 싱글톤"""
    global zodiac_manager
    if zodiac_manager is None:
        zodiac_manager = ZodiacLuckyManager()
    return zodiac_manager

@app.route('/')
def index():
    """메인 페이지"""
    stats = lotto_wrapper.get_basic_stats()
    history = lotto_wrapper.load_prediction_history()
    return render_template('index.html', stats=stats, history_count=len(history), show_zodiac_option=True)

@app.route('/predict', methods=['POST'])
def predict():
    """예측 번호 생성 API - 기존 main.py 실행"""
    try:
        # 예측 개수 (기본값: 5)
        num_predictions = 5
        if request.is_json and 'count' in request.json:
            num_predictions = min(max(int(request.json['count']), 1), 10)
        
        # main.py 실행
        result = lotto_wrapper.run_main_prediction(num_predictions)
        
        if result['success']:
            return jsonify({
                'success': True,
                'predictions': result['predictions'],
                'message': 'AI 분석 완료! 천재적 공식으로 예측된 번호입니다.'
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'fallback': lotto_wrapper.generate_fallback_predictions()
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'예측 생성 중 오류 발생: {str(e)}',
            'fallback': lotto_wrapper.generate_fallback_predictions()
        }), 500

@app.route('/history')
def history():
    """예측 기록 조회 페이지"""
    try:
        # 직접 히스토리 파일 읽기 (lotto_wrapper가 제대로 작동하지 않을 경우 대비)
        predictions = load_prediction_history_direct()
        print(f"전달할 예측 데이터: {predictions}")  # 디버그용
        print(f"데이터 타입: {type(predictions)}")
        print(f"데이터 개수: {len(predictions) if predictions else 0}")
        
        # 데이터 정리 - JSON 직렬화 가능하도록 확인
        clean_predictions = []
        for i, pred in enumerate(predictions):
            try:
                clean_pred = {
                    'id': str(pred.get('id', i + 1)),
                    'numbers': pred.get('numbers', []),
                    'confidence': float(pred.get('confidence', 0.5)),
                    'timestamp': str(pred.get('timestamp', pred.get('date', ''))),
                    'date': str(pred.get('date', '')),
                    'method': str(pred.get('method', 'unknown')),
                    'explanation': str(pred.get('explanation', '통계 분석'))
                }
                
                # 숫자 배열 검증 및 정리
                if isinstance(clean_pred['numbers'], list) and len(clean_pred['numbers']) >= 6:
                    # 숫자만 추출하고 정렬
                    clean_pred['numbers'] = [int(n) for n in clean_pred['numbers'][:6] if isinstance(n, (int, str)) and str(n).isdigit()]
                    if len(clean_pred['numbers']) == 6:
                        clean_predictions.append(clean_pred)
                elif isinstance(clean_pred['numbers'], str):
                    # 문자열 형태의 번호를 파싱 (예: "1, 2, 3, 4, 5, 6")
                    try:
                        numbers_str = clean_pred['numbers'].replace('[', '').replace(']', '')
                        numbers = [int(x.strip()) for x in numbers_str.split(',') if x.strip().isdigit()]
                        if len(numbers) == 6:
                            clean_pred['numbers'] = numbers
                            clean_predictions.append(clean_pred)
                    except:
                        continue
                        
            except Exception as e:
                print(f"예측 데이터 정리 오류: {e}, 데이터: {pred}")
                continue
        
        print(f"정리된 예측 데이터: {len(clean_predictions)}개")
        
        # 빈 데이터인 경우 샘플 데이터 생성 (테스트용)
        if not clean_predictions:
            print("빈 히스토리 데이터 - 샘플 데이터 생성")
            clean_predictions = generate_sample_predictions()
        
        # JSON 문자열로 변환하여 템플릿에 전달
        import json
        predictions_json = json.dumps(clean_predictions, ensure_ascii=False, separators=(',', ':'))
        print(f"JSON 문자열 길이: {len(predictions_json)}")
        print(f"JSON 문자열 미리보기: {predictions_json[:200]}...")
        
        return render_template('history.html', 
                             predictions=clean_predictions, 
                             predictions_json=predictions_json)
        
    except Exception as e:
        print(f"히스토리 로딩 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류 발생 시에도 빈 데이터로 페이지 렌더링
        sample_predictions = generate_sample_predictions()
        predictions_json = json.dumps(sample_predictions, ensure_ascii=False)
        
        return render_template('history.html', 
                             predictions=sample_predictions, 
                             predictions_json=predictions_json)


def load_prediction_history_direct():
    """히스토리 파일을 직접 읽어서 파싱"""
    from pathlib import Path
    import re
    
    history_file = Path('outputs/predictions/prediction_history.txt')
    predictions = []
    
    try:
        if not history_file.exists():
            print(f"히스토리 파일이 존재하지 않음: {history_file}")
            return []
        
        with open(history_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"히스토리 파일에서 {len(lines)}줄 읽음")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or '날짜 | 시간 | ID | 예측번호 | 신뢰도' in line or '날짜 | ID | 예측번호 | 신뢰도' in line:
                continue  # 헤더나 빈 줄 건너뛰기
                
            try:
                # 파이프로 구분된 데이터 파싱
                parts = [part.strip() for part in line.split('|')]
                
                # 새 형식 (시간 포함): "날짜 | 시간 | ID | 예측번호 | 신뢰도"
                if len(parts) >= 5:
                    date_str = parts[0]
                    time_str = parts[1] 
                    id_str = parts[2]
                    numbers_str = parts[3]
                    confidence_str = parts[4]
                    timestamp = f"{date_str} {time_str}"
                    
                # 기존 형식 (시간 없음): "날짜 | ID | 예측번호 | 신뢰도"
                elif len(parts) >= 4:
                    date_str = parts[0]
                    id_str = parts[1]
                    numbers_str = parts[2]
                    confidence_str = parts[3]
                    timestamp = f"{date_str} 00:00:00"
                    
                else:
                    continue
                    
                # 번호 파싱 (예: "1, 2, 3, 4, 5, 6")
                numbers = [int(x.strip()) for x in numbers_str.split(',') if x.strip().isdigit()]
                
                # 신뢰도 파싱 (예: "57.63%")
                confidence_match = re.search(r'(\d+\.?\d*)%?', confidence_str)
                confidence = float(confidence_match.group(1)) / 100 if confidence_match else 0.5
                if confidence > 1:  # 100% 형태였다면
                    confidence = confidence / 100
                
                if len(numbers) == 6:
                    prediction = {
                        'id': id_str,
                        'date': date_str,
                        'timestamp': timestamp,
                        'numbers': sorted(numbers),
                        'confidence': confidence,
                        'method': 'genius_insight',
                        'explanation': '통계 분석 기반 예측'
                    }
                    predictions.append(prediction)
                    
            except Exception as e:
                print(f"라인 {i+1} 파싱 오류: {e}, 라인: {line}")
                continue
        
        print(f"총 {len(predictions)}개의 예측 파싱 완료")
        return predictions
        
    except Exception as e:
        print(f"히스토리 파일 읽기 오류: {e}")
        return []


def generate_sample_predictions():
    """테스트용 샘플 예측 데이터 생성"""
    import random
    from datetime import datetime, timedelta
    
    sample_predictions = []
    
    for i in range(5):  # 5개 샘플 생성
        # 랜덤 번호 생성
        numbers = sorted(random.sample(range(1, 46), 6))
        
        # 날짜 생성 (최근 5일)
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        
        prediction = {
            'id': str(i + 1),
            'numbers': numbers,
            'confidence': random.uniform(0.4, 0.8),
            'timestamp': f"{date} 00:00:00",
            'date': date,
            'method': random.choice(['genius_insight', 'frequency', 'random']),
            'explanation': f'샘플 예측 #{i + 1} - 통계 분석'
        }
        sample_predictions.append(prediction)
    
    print(f"샘플 예측 {len(sample_predictions)}개 생성")
    return sample_predictions

@app.route('/api/history')
def api_history():
    """예측 기록 API"""
    try:
        history = lotto_wrapper.load_prediction_history()
        return jsonify({
            'success': True,
            'predictions': history,
            'total': len(history)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/statistics')
def statistics():
    """통계 정보 페이지"""
    try:
        stats = lotto_wrapper.get_basic_stats()
        print(f"통계 데이터: {stats}")  # 디버그용
        
        # JSON 문자열로 안전하게 변환
        import json
        stats_json = json.dumps(stats, ensure_ascii=False, separators=(',', ':'))
        print(f"통계 JSON 길이: {len(stats_json)}")
        
        return render_template('statistics.html', 
                             stats=stats, 
                             stats_json=stats_json)
                             
    except Exception as e:
        print(f"통계 로딩 오류: {e}")
        default_stats = {
            'total_draws': 0,
            'frequency': {str(i): 10 for i in range(1, 46)},
            'most_frequent': 7,
            'least_frequent': 13,
            'avg_frequency': 10.0
        }
        stats_json = json.dumps(default_stats, ensure_ascii=False)
        
        return render_template('statistics.html', 
                             stats=default_stats, 
                             stats_json=stats_json)

def get_filtered_python_version():
    """Python 버전 정보 필터링"""
    version_info = sys.version_info
    return f"Python {version_info.major}.{version_info.minor}.x"

def get_prediction_count():
    """예측 기록 개수 반환"""
    try:
        if os.path.exists(PREDICTIONS_HISTORY_PATH):
            with open(PREDICTIONS_HISTORY_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return max(0, len(lines) - 1)  # 헤더 제외
        return 0
    except:
        return 0

def get_basic_status():
    """기본적인 시스템 정보만 반환 (프로덕션용)"""
    return {
        'main_py_exists': os.path.exists(MAIN_SCRIPT_PATH),
        'csv_data_exists': os.path.exists(CSV_DATA_PATH),
        'output_dir_exists': os.path.exists('outputs'),
        'predictions_dir_exists': os.path.exists('outputs/predictions'),
        'reports_dir_exists': os.path.exists('outputs/reports'),
        'logs_dir_exists': os.path.exists('logs'),
        'python_version': get_filtered_python_version(),
        'app_status': 'running',
        'total_predictions': get_prediction_count(),
        'environment': 'production' if not DEBUG_MODE else 'development',
        'last_check': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def get_detailed_status():
    """상세한 시스템 정보 반환 (개발용)"""
    basic_status = get_basic_status()
    
    # 개발 환경에서만 추가 정보 제공
    if DEBUG_MODE and SHOW_SENSITIVE_INFO:
        detailed_info = {
            'current_dir': os.getcwd(),
            'python_full_version': sys.version,
            'file_count': len(os.listdir('.')) if os.path.exists('.') else 0,
            'key_files': get_key_files_status(),
            'directory_structure': get_safe_directory_structure()
        }
        basic_status.update(detailed_info)
    
    return basic_status

def get_key_files_status():
    """주요 파일들의 상태만 확인"""
    key_files = [
        'main.py',
        'app.py', 
        'requirements.txt',
        'data/raw/lotto_results.csv',
        'src/prediction/ensemble_predictor.py',
        'src/utils/validator.py'
    ]
    
    status = {}
    for file_path in key_files:
        status[file_path] = {
            'exists': os.path.exists(file_path),
            'size': get_safe_file_size(file_path)
        }
    
    return status

def get_safe_file_size(file_path):
    """안전한 파일 크기 정보"""
    try:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size < 1024:
                return f"{size}B"
            elif size < 1024*1024:
                return f"{size//1024}KB"
            else:
                return f"{size//(1024*1024)}MB"
        return "N/A"
    except:
        return "Error"

def get_safe_directory_structure():
    """안전한 디렉토리 구조 정보"""
    important_dirs = [
        'src/',
        'data/',
        'outputs/',
        'logs/',
        'templates/',
        'static/'
    ]
    
    structure = {}
    for dir_path in important_dirs:
        if os.path.exists(dir_path):
            try:
                file_count = len([f for f in os.listdir(dir_path) 
                                if os.path.isfile(os.path.join(dir_path, f))])
                subdir_count = len([d for d in os.listdir(dir_path) 
                                  if os.path.isdir(os.path.join(dir_path, d))])
                structure[dir_path] = {
                    'files': file_count,
                    'subdirs': subdir_count
                }
            except:
                structure[dir_path] = {'files': 'Error', 'subdirs': 'Error'}
        else:
            structure[dir_path] = {'exists': False}
    
    return structure

@app.route('/system-check')
def system_check():
    """보안 강화된 시스템 상태 확인 페이지"""
    try:
        if DEBUG_MODE:
            print(f"[DEBUG] System check accessed in debug mode")
            print(f"[DEBUG] SHOW_SENSITIVE_INFO: {SHOW_SENSITIVE_INFO}")
        
        # 환경에 따른 정보 제공
        if DEBUG_MODE:
            status = get_detailed_status()
        else:
            status = get_basic_status()
        
        # 추가 보안 정보
        security_info = {
            'debug_mode': DEBUG_MODE,
            'sensitive_info_enabled': SHOW_SENSITIVE_INFO,
            'access_level': 'admin' if (DEBUG_MODE and SHOW_SENSITIVE_INFO) else 'basic'
        }
        
        return render_template('system_check.html', 
                             status=status, 
                             security=security_info)
        
    except Exception as e:
        print(f"System check error: {e}")
        # 오류 시 최소한의 정보만 제공
        minimal_status = {
            'app_status': 'error',
            'error_message': 'System check failed',
            'environment': 'unknown',
            'last_check': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        security_info = {
            'debug_mode': False,
            'sensitive_info_enabled': False,
            'access_level': 'minimal'
        }
        
        return render_template('system_check.html', 
                             status=minimal_status, 
                             security=security_info)

@app.route('/run-analysis', methods=['POST'])
def run_analysis():
    """전체 분석 실행 (고급 기능)"""
    try:
        # main.py --full-analysis 실행
        cmd = [sys.executable, MAIN_SCRIPT_PATH, '--full-analysis']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': '전체 분석이 완료되었습니다.',
                'output': result.stdout[:1000]  # 처음 1000자만
            })
        else:
            return jsonify({
                'success': False,
                'error': result.stderr
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': '분석 시간이 초과되었습니다.'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/zodiac')
def zodiac_page():
    """별자리 행운 요소 입력 페이지"""
    return render_template('zodiac.html')

# 별자리 예측 API (app.py에 추가)
@app.route('/api/zodiac-predict', methods=['POST'])
def api_zodiac_predict():
    """별자리 기반 예측 API"""
    try:
        data = request.get_json()
        
        # 입력 데이터 검증
        zodiac = data.get('zodiac', '').lower()
        user_objects = data.get('objects', [])[:3]  # 최대 3개
        user_numbers = [int(n) for n in data.get('numbers', []) if str(n).isdigit()][:5]  # 최대 5개
        user_colors = data.get('colors', [])[:3]  # 최대 3개
        
        if not zodiac:
            return jsonify({'success': False, 'error': '별자리를 선택해주세요.'})
        
        # 기존 예측 시스템과 연동하여 기본 예측 생성
        lotto_wrapper = FlaskLottoWrapper()
        base_result = lotto_wrapper.run_main_prediction(1)
        
        base_prediction = None
        if base_result.get('success') and base_result.get('predictions'):
            base_prediction = base_result['predictions'][0]['numbers']
        
        # 별자리 예측 생성
        manager = get_zodiac_manager()
        prediction, metadata = manager.get_zodiac_prediction(
            zodiac, user_objects, user_numbers, user_colors, base_prediction
        )
        
        # 응답 데이터
        response_data = {
            'success': True,
            'prediction': {
                'numbers': prediction,
                'confidence': metadata['confidence'],
                'zodiac': metadata['zodiac'],
                'lucky_story': metadata['lucky_story'],
                'element_analysis': metadata['element_analysis'],
                'timestamp': datetime.now().isoformat(),
                'method': 'zodiac_lucky'
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Zodiac prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': '별자리 예측 생성 중 오류가 발생했습니다.'
        }), 500

# 별자리 정보 API (app.py에 추가)
@app.route('/api/zodiac-info/<zodiac_name>')
def get_zodiac_info(zodiac_name):
    """특정 별자리 정보 반환"""
    try:
        manager = get_zodiac_manager()
        zodiac = Zodiac(zodiac_name.lower())
        elements = manager.zodiac_elements[zodiac]
        
        return jsonify({
            'success': True,
            'zodiac': manager._get_zodiac_korean_name(zodiac),
            'recommended_objects': elements.objects,
            'lucky_numbers': elements.numbers,
            'lucky_colors': elements.colors
        })
        
    except ValueError:
        return jsonify({'success': False, 'error': '유효하지 않은 별자리입니다.'})
    except Exception as e:
        app.logger.error(f"Zodiac info error: {str(e)}")
        return jsonify({'success': False, 'error': '정보를 가져올 수 없습니다.'})
    
@app.route('/health')
def health_check():
    '''Kubernetes 헬스체크용 엔드포인트'''
    try:
        # 기본 시스템 체크
        pod_name = os.getenv('POD_NAME', 'unknown')
        node_name = os.getenv('NODE_NAME', 'unknown')
        
        # CSV 데이터 존재 확인
        csv_exists = os.path.exists('/app/data/lotto_results.csv')
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'pod_name': pod_name,
            'node_name': node_name,
            'csv_data_available': csv_exists,
            'flask_port': 5000
        }
        
        return jsonify(health_status), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/ready')
def readiness_check():
    '''Kubernetes 준비상태 체크용 엔드포인트'''
    try:
        # CSV 데이터 로딩 가능 여부 확인
        csv_path = '/app/data/lotto_results.csv'
        if not os.path.exists(csv_path):
            return jsonify({
                'ready': False,
                'reason': 'CSV 데이터 파일 없음'
            }), 503
        
        # 기본 기능 동작 확인
        stats = lotto_wrapper.get_basic_stats()
        
        return jsonify({
            'ready': True,
            'csv_rows': stats.get('total_draws', 0),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'ready': False,
            'error': str(e)
        }), 503

@app.route('/metrics')
def metrics():
    '''Prometheus 메트릭 엔드포인트 (선택사항)'''
    try:
        pod_name = os.getenv('POD_NAME', 'unknown')
        
        # 기본 메트릭
        metrics_text = f'''# HELP lotto_predictions_total Total predictions generated
# TYPE lotto_predictions_total counter
lotto_predictions_total{{pod="{pod_name}"}} 0

# HELP lotto_app_info Application info
# TYPE lotto_app_info gauge  
lotto_app_info{{version="1.0.0",pod="{pod_name}"}} 1
'''
        
        return Response(metrics_text, mimetype='text/plain')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    print("🚀 Flask 로또 예측 웹 애플리케이션 시작")
    print(f"📁 작업 디렉토리: {os.getcwd()}")
    print(f"🐍 Python 버전: {sys.version}")
    print(f"📊 main.py 상태: {'✅ 존재' if os.path.exists(MAIN_SCRIPT_PATH) else '❌ 없음'}")
    print(f"📄 CSV 데이터: {'✅ 존재' if os.path.exists(CSV_DATA_PATH) else '❌ 없음'}")
    print("🌐 접속 주소: http://localhost:5000")
    print("🔍 시스템 확인: http://localhost:5000/system-check")
    
    # 필요한 디렉토리 생성
    lotto_wrapper.ensure_directories()
    
    # Flask 앱 실행
    app.run(debug=True, host='0.0.0.0', port=5000)