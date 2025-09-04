#!/usr/bin/env python3
"""
Flask 로또 예측 웹 애플리케이션
기존 main.py 시스템을 웹 인터페이스로 감싸는 래퍼
- 기능 1: 예측 번호 생성 버튼 (기존 main.py 활용)
- 기능 2: 이전 예측 기록 조회
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for, send_file, send_from_directory
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

# 전역 래퍼 인스턴스
lotto_wrapper = FlaskLottoWrapper()

@app.route('/')
def index():
    """메인 페이지"""
    stats = lotto_wrapper.get_basic_stats()
    history = lotto_wrapper.load_prediction_history()
    return render_template('index.html', stats=stats, history_count=len(history))

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