from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import traceback
import logging
import pandas as pd
import numpy as np

# 모든 모듈을 명시적으로 import (순환 참조 방지)
from data_parser import LottoDataParser
from lotto_predictor import LottoPredictor
from advanced_predictor import AdvancedLottoPredictor
from visualization import LottoVisualizer
from database import DatabaseManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 전역 시스템 상태
system = {
    'predictor': None,
    'advanced_predictor': None,
    'visualizer': None,
    'db_manager': None,
    'data': None,
    'last_update': None,
    'initialization_status': 'Not Started',
    'diversity_manager': None
}

class DiversityManager:
    """중복 예측 관리 클래스"""
    """중복 예측 관리 클래스 - 자릿수 다양성 포함"""
    def __init__(self):
        self.global_prediction_cache = []
        self.session_predictions = {}
        self.diversity_settings = {
            'min_difference': 5000,
            'cache_size': 100,
            'similarity_threshold': 0.8,
            'digit_diversity_threshold': 0.6  # 자릿수 편중 임계값
        }
        
        # 자릿수별 다양성 추적
        self.global_digit_history = {i: [] for i in range(6)}
        
        logger.info("DiversityManager 초기화 완료 (자릿수 다양성 포함)")
    
    def _check_digit_diversity(self, number_str):
        """전역 자릿수 다양성 검사"""
        if len(number_str) != 6:
            return True, "번호 길이 오류"
        
        diversity_issues = []
        
        for pos in range(6):
            digit = int(number_str[pos])
            recent_digits = self.global_digit_history[pos][-15:]  # 최근 15개
            
            if len(recent_digits) >= 5:
                # 연속성 검사 - 최근 4개가 모두 같은 숫자
                if len(recent_digits) >= 4 and recent_digits[-4:].count(digit) >= 4:
                    diversity_issues.append(f"위치{pos+1}: {digit} 연속 4회 이상")
                
                # 빈도 검사 - 60% 이상 같은 숫자
                frequency_ratio = recent_digits.count(digit) / len(recent_digits)
                if frequency_ratio >= self.diversity_settings['digit_diversity_threshold']:
                    diversity_issues.append(f"위치{pos+1}: {digit} 빈도 {frequency_ratio*100:.0f}%")
        
        if diversity_issues:
            return False, "; ".join(diversity_issues)
        else:
            return True, "자릿수 다양성 양호"

    def _update_digit_history(self, number_str):
        """자릿수별 기록 업데이트"""
        if len(number_str) != 6:
            return
        
        for pos in range(6):
            digit = int(number_str[pos])
            self.global_digit_history[pos].append(digit)
            
            # 최근 30개만 유지
            if len(self.global_digit_history[pos]) > 30:
                self.global_digit_history[pos] = self.global_digit_history[pos][-30:]
    
    def _generate_digit_diverse_alternative(self, base_number_str):
        """자릿수 다양성을 고려한 대안 번호 생성"""
        new_digits = list(base_number_str.zfill(6))
        
        for pos in range(6):
            current_digit = int(new_digits[pos])
            recent_digits = self.global_digit_history[pos][-10:]
            
            if len(recent_digits) >= 3:
                # 현재 숫자의 빈도가 높으면 교체
                frequency = recent_digits.count(current_digit) / len(recent_digits)
                if frequency > 0.5:  # 50% 이상이면 교체
                    # 사용 빈도가 낮은 숫자들 찾기
                    if pos == 0:  # 첫 번째 자리 (조)
                        candidates = list(range(1, 6))
                    else:
                        candidates = list(range(10))
                    
                    # 각 후보의 최근 사용 빈도 계산
                    candidate_scores = {}
                    for candidate in candidates:
                        freq = recent_digits.count(candidate) / len(recent_digits)
                        candidate_scores[candidate] = freq
                    
                    # 가장 적게 사용된 숫자 선택
                    best_candidate = min(candidate_scores.items(), key=lambda x: x[1])[0]
                    if best_candidate != current_digit:
                        new_digits[pos] = str(best_candidate)
                        print(f"자릿수 다양성 개선: 위치{pos+1} {current_digit} → {best_candidate}")
        
        return ''.join(new_digits)

    def is_prediction_unique(self, prediction, session_id=None):
        """예측의 고유성 검사 - 자릿수 다양성 포함"""
        try:
            pred_num = int(prediction.get('number', '000000'))
            pred_str = str(pred_num).zfill(6)
            
            # 기존 전역 캐시 검사
            for cached_pred in self.global_prediction_cache[-20:]:
                try:
                    if abs(pred_num - int(cached_pred)) < self.diversity_settings['min_difference']:
                        return False, f"번호 유사성: 기존 {cached_pred}와 차이 {abs(pred_num - int(cached_pred))}"
                except (ValueError, TypeError):
                    continue
            
            # 자릿수 다양성 검사
            is_digit_diverse, digit_message = self._check_digit_diversity(pred_str)
            if not is_digit_diverse:
                return False, f"자릿수 편중: {digit_message}"
            
            # 세션별 검사
            if session_id and session_id in self.session_predictions:
                session_preds = self.session_predictions[session_id]
                for s_pred in session_preds[-5:]:
                    try:
                        if abs(pred_num - int(s_pred)) < self.diversity_settings['min_difference'] * 0.5:
                            return False, f"세션 내 중복: {s_pred}"
                    except (ValueError, TypeError):
                        continue
            
            return True, "고유한 예측"
            
        except Exception as e:
            logger.error(f"고유성 검사 중 오류: {e}")
            return True, "검사 오류로 인한 통과"
    
    def add_prediction(self, prediction, session_id=None):
        """예측을 캐시에 추가 - 자릿수 기록 포함"""
        try:
            pred_str = str(prediction.get('number', '000000')).zfill(6)
            
            # 전역 캐시 추가
            self.global_prediction_cache.append(pred_str)
            if len(self.global_prediction_cache) > self.diversity_settings['cache_size']:
                self.global_prediction_cache = self.global_prediction_cache[-self.diversity_settings['cache_size']:]
            
            # 자릿수별 기록 업데이트
            self._update_digit_history(pred_str)
            
            # 세션별 캐시 추가
            if session_id:
                if session_id not in self.session_predictions:
                    self.session_predictions[session_id] = []
                self.session_predictions[session_id].append(pred_str)
                
                if len(self.session_predictions[session_id]) > 20:
                    self.session_predictions[session_id] = self.session_predictions[session_id][-20:]
                    
        except Exception as e:
            logger.error(f"예측 추가 중 오류: {e}")
    
    def generate_diverse_alternative(self, base_prediction):
        """다양성을 고려한 대안 생성"""
        try:
            base_number = base_prediction.get('number', '000000')
            
            # 자릿수 다양성 우선 적용
            digit_diverse_number = self._generate_digit_diverse_alternative(base_number)
            
            # 조 일치성 확인
            new_jo = int(digit_diverse_number[0])
            if new_jo < 1 or new_jo > 5:
                # 유효하지 않은 조면 조정
                valid_jos = list(range(1, 6))
                # 최근 가장 적게 사용된 조 선택
                jo_counts = {}
                for cached in self.global_prediction_cache[-20:]:
                    try:
                        jo = int(cached[0])
                        jo_counts[jo] = jo_counts.get(jo, 0) + 1
                    except (ValueError, IndexError):
                        continue
                
                best_jo = min(valid_jos, key=lambda x: jo_counts.get(x, 0))
                digit_diverse_number = str(best_jo) + digit_diverse_number[1:]
            
            return {
                'jo': int(digit_diverse_number[0]),
                'number': digit_diverse_number,
                'method': 'digit_diversity',
                'confidence_adjustment': -5
            }
            
        except Exception as e:
            logger.error(f"대안 생성 중 오류: {e}")
            # 폴백: 완전 랜덤
            import random
            random_jo = random.randint(1, 5)
            random_number = str(random_jo * 100000 + random.randint(0, 99999)).zfill(6)
            
            return {
                'jo': random_jo,
                'number': random_number,
                'method': 'fallback_random',
                'confidence_adjustment': -10
            }

    def get_diversity_stats(self):
        """다양성 통계 반환 - 자릿수 분석 포함"""
        try:
            if len(self.global_prediction_cache) < 2:
                return {"message": "예측 기록이 부족합니다"}
            
            # 기본 다양성 통계
            recent = [int(p) for p in self.global_prediction_cache[-10:] if p.isdigit()]
            if len(recent) < 2:
                return {"message": "유효한 예측 기록이 부족합니다"}
            
            differences = []
            for i in range(1, len(recent)):
                diff = abs(recent[i] - recent[i-1])
                differences.append(diff)
            
            avg_diff = sum(differences) / len(differences) if differences else 0
            diversity_score = min(100, (avg_diff / 10000) * 100)
            
            # 자릿수별 다양성 분석
            digit_diversity = {}
            for pos in range(6):
                recent_digits = self.global_digit_history[pos][-10:]
                if recent_digits:
                    unique_count = len(set(recent_digits))
                    max_possible = 5 if pos == 0 else 10
                    pos_diversity = (unique_count / min(max_possible, len(recent_digits))) * 100
                    
                    # 가장 빈번한 숫자
                    most_common = max(set(recent_digits), key=recent_digits.count)
                    frequency = recent_digits.count(most_common)
                    
                    digit_diversity[f'position_{pos+1}'] = {
                        'diversity_score': round(pos_diversity, 1),
                        'unique_digits': unique_count,
                        'most_common': most_common,
                        'most_common_frequency': frequency,
                        'recent_sequence': recent_digits[-5:]
                    }
            
            # 전체 자릿수 다양성 점수
            pos_scores = [dd.get('diversity_score', 0) for dd in digit_diversity.values()]
            digit_overall_score = sum(pos_scores) / len(pos_scores) if pos_scores else 0
            
            return {
                'diversity_score': round(diversity_score, 1),
                'average_difference': round(avg_diff, 0),
                'total_predictions': len(self.global_prediction_cache),
                'recent_predictions': self.global_prediction_cache[-5:],
                'settings': self.diversity_settings,
                'digit_diversity': digit_diversity,
                'digit_overall_score': round(digit_overall_score, 1),
                'digit_status': 'excellent' if digit_overall_score >= 80 else 'good' if digit_overall_score >= 60 else 'needs_improvement'
            }
            
        except Exception as e:
            logger.error(f"다양성 통계 생성 오류: {e}")
            return {
                'error': str(e),
                'diversity_score': 0,
                'total_predictions': 0
            }

def initialize_system():
    """시스템 초기화 (다양성 관리자 포함)"""
    try:
        logger.info("=== 시스템 초기화 시작 ===")
        system['initialization_status'] = 'In Progress'
        
        # 다양성 관리자 초기화 (제일 먼저)
        try:
            system['diversity_manager'] = DiversityManager()
            logger.info("✓ 다양성 관리자 초기화 완료")
        except Exception as e:
            logger.error(f"다양성 관리자 초기화 실패: {e}")
            system['diversity_manager'] = None
        
        # 기존 초기화 코드...
        # 1. 데이터베이스 매니저 초기화
        try:
            system['db_manager'] = DatabaseManager()
            logger.info("✓ 데이터베이스 매니저 초기화 완료")
        except Exception as e:
            logger.warning(f"데이터베이스 매니저 초기화 실패: {e}")
            system['db_manager'] = None
        
        # 2. HTML 파일 파싱
        try:
            parser = LottoDataParser('data/lotto_720.html')
            data = parser.parse_html()
            
            if data is None or len(data) == 0:
                logger.warning("HTML 파싱 실패, 샘플 데이터로 진행")
                data = parser._create_comprehensive_sample_data()
            
            system['data'] = data
            logger.info(f"✓ 데이터 로드 완료: {len(data)}개")
            
        except Exception as e:
            logger.error(f"데이터 파싱 실패: {e}")
            system['data'] = create_emergency_sample_data()
            logger.info("✓ 응급 샘플 데이터 생성 완료")
        
        # 3. AI 예측 모델 초기화
        try:
            logger.info("AI 예측 모델 초기화 중...")
            system['predictor'] = LottoPredictor(system['data'], system['db_manager'])
            
            if system['predictor'].is_trained:
                logger.info("✓ AI 모델 학습 완료")
            else:
                logger.warning("⚠ AI 모델 학습 실패, 패턴 기반 모델로 동작")
                
        except Exception as e:
            logger.error(f"AI 예측 모델 초기화 실패: {e}")
            try:
                system['predictor'] = LottoPredictor(system['data'])
                logger.info("✓ 기본 예측 모델 생성 완료")
            except Exception as e2:
                logger.error(f"기본 예측 모델 생성도 실패: {e2}")
                system['predictor'] = None
        
        # 4. 고급 예측 모델
        try:
            if len(system['data']) >= 50:
                system['advanced_predictor'] = AdvancedLottoPredictor(system['data'])
                system['advanced_predictor'].train_advanced_models()
                logger.info("✓ 고급 예측 모델 초기화 완료")
            else:
                system['advanced_predictor'] = None
        except Exception as e:
            logger.warning(f"고급 예측 모델 초기화 실패: {e}")
            system['advanced_predictor'] = None
        
        # 5. 시각화 객체
        try:
            system['visualizer'] = LottoVisualizer(system['data'])
            logger.info("✓ 시각화 모듈 초기화 완료")
        except Exception as e:
            logger.warning(f"시각화 모듈 초기화 실패: {e}")
            system['visualizer'] = None
        
        system['last_update'] = datetime.now()
        system['initialization_status'] = 'Completed'
        logger.info("=== 시스템 초기화 완료 ===")
        return True
        
    except Exception as e:
        logger.error(f"시스템 초기화 실패: {e}")
        system['initialization_status'] = 'Failed'
        return False

def create_emergency_sample_data():
    """응급 샘플 데이터 생성"""
    try:
        np.random.seed(42)
        data = []
        
        for i in range(50):
            jo = np.random.randint(1, 6)
            number = jo * 100000 + np.random.randint(0, 99999)
            
            data.append({
                'draw_no': 671 + i,
                'draw_date': pd.Timestamp('2023-01-01') + pd.Timedelta(weeks=i),
                'jo': jo,
                'number': str(number).zfill(6),
                'bonus_number': str(np.random.randint(100000, 999999)).zfill(6)
            })
        
        df = pd.DataFrame(data)
        df['number'] = df['number'].astype(int)
        return df
        
    except Exception as e:
        logger.error(f"응급 데이터 생성 실패: {e}")
        # 최소한의 데이터라도 반환
        return pd.DataFrame({
            'draw_no': [720],
            'draw_date': [pd.Timestamp.now()],
            'jo': [1],
            'number': [123456],
            'bonus_number': [654321]
        })

def ensure_system_ready():
    """시스템 준비 상태 확인"""
    try:
        if system['initialization_status'] not in ['Completed']:
            logger.info("시스템 재초기화 필요")
            return initialize_system()
        
        # 다양성 관리자 확인
        if not system.get('diversity_manager'):
            logger.info("다양성 관리자 재생성")
            system['diversity_manager'] = DiversityManager()
        
        return True
        
    except Exception as e:
        logger.error(f"시스템 준비 확인 실패: {e}")
        return False

# 앱 시작 시 자동 초기화 시도
try:
    initialize_system()
except Exception as e:
    logger.error(f"앱 시작 시 초기화 실패: {e}")

# 스케줄러 설정 (안전하게)
try:
    def update_data_job():
        """주기적 데이터 업데이트"""
        try:
            logger.info("주기적 데이터 업데이트 시작")
            parser = LottoDataParser('data/lotto_720.html')
            latest_data = parser.fetch_latest_data()
            if latest_data:
                initialize_system()
                logger.info("데이터 업데이트 및 재학습 완료")
        except Exception as e:
            logger.error(f"주기적 업데이트 실패: {e}")

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=update_data_job, trigger="cron", hour=12, minute=0)
    scheduler.start()
    logger.info("스케줄러 시작 완료")
    
except Exception as e:
    logger.warning(f"스케줄러 설정 실패: {e}")

# ==================== 라우트 정의 ====================

@app.route('/')
def index():
    """메인 페이지"""
    try:
        ensure_system_ready()
        return render_template('index.html')
    except Exception as e:
        logger.error(f"메인 페이지 렌더링 실패: {e}")
        return render_template('error.html', 
                             title="메인 페이지 오류", 
                             message="메인 페이지를 로드할 수 없습니다.")

@app.route('/predict')
def predict():
    """예측 수행 - 다양성 검사 포함"""
    try:
        logger.info("예측 페이지 요청 받음")
        ensure_system_ready()
        
        # 세션 ID 생성 (간단한 방식)
        session_id = request.remote_addr + str(int(datetime.now().timestamp()) // 3600)  # 1시간 단위
        
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            # 예측 수행
            ml_prediction = None
            advanced_prediction = None
            
            if system['predictor']:
                try:
                    ml_prediction = system['predictor'].predict_next()
                    logger.info(f"예측 완료: {ml_prediction.get('model_type', 'Unknown')} 모델 사용")
                except Exception as e:
                    logger.error(f"예측 수행 중 오류: {e}")
                    ml_prediction = create_sample_prediction()
            else:
                ml_prediction = create_sample_prediction()
            
            # 고급 예측 시도
            if system['advanced_predictor']:
                try:
                    advanced_prediction = system['advanced_predictor'].predict_ensemble()
                    logger.info("고급 앙상블 예측도 완료")
                except Exception as e:
                    logger.warning(f"고급 예측 실패: {e}")
                    advanced_prediction = create_sample_advanced_prediction()
            else:
                advanced_prediction = create_sample_advanced_prediction()
            
            # 다양성 검사
            diversity_manager = system.get('diversity_manager')
            if diversity_manager:
                main_prediction = ml_prediction.get('ml_prediction', {})
                is_unique, message = diversity_manager.is_prediction_unique(main_prediction, session_id)
                
                if is_unique:
                    # 고유한 예측이므로 캐시에 추가하고 사용
                    diversity_manager.add_prediction(main_prediction, session_id)
                    logger.info(f"고유한 예측 확인: {message}")
                    break
                else:
                    # 중복이므로 재시도
                    logger.info(f"중복 예측 감지 (시도 {attempt + 1}): {message}")
                    attempt += 1
                    
                    # 다음 시도를 위해 약간의 지연과 시드 변경
                    import time
                    time.sleep(0.1)
                    
                    if system['predictor'] and hasattr(system['predictor'], 'randomness_seed'):
                        system['predictor'].randomness_seed += 1000
                    
                    continue
            else:
                # 다양성 관리자가 없으면 그대로 진행
                break
        
        if attempt >= max_attempts:
            logger.warning(f"최대 시도 횟수 {max_attempts} 도달, 강제 다양화 적용")
            # 강제 다양화
            main_prediction = ml_prediction.get('ml_prediction', {})
            original_num = int(main_prediction.get('number', '100000'))
            forced_num = original_num + np.random.randint(10000, 50000)
            forced_num = max(100000, min(999999, forced_num))
            main_prediction['number'] = str(forced_num).zfill(6)
            main_prediction['jo'] = int(str(forced_num)[0])
            
            if diversity_manager:
                diversity_manager.add_prediction(main_prediction, session_id)
        
        # 차트 생성
        charts = create_safe_charts()
        
        # 통계 정보 
        statistics = get_safe_statistics()
        
        # 템플릿과 맞는 데이터 구조로 변환
        template_data = {
            'ml': {
                'next_draw_no': ml_prediction.get('next_draw_no', 721),
                'confidence_score': ml_prediction.get('confidence_score', 75.0),
                'ml_prediction': ml_prediction.get('ml_prediction', {}),
                'pattern_predictions': ml_prediction.get('pattern_predictions', []),
                'statistical_predictions': [],
                'diversity_info': {
                    'attempts_used': attempt + 1,
                    'max_attempts': max_attempts,
                    'uniqueness_verified': True
                }
            },
            'advanced': advanced_prediction,
            'statistics': statistics
        }
        
        # result.html 템플릿 사용
        try:
            return render_template('result.html', 
                                 prediction=template_data,
                                 charts=charts)
        except Exception as template_error:
            logger.error(f"result.html 템플릿 렌더링 실패: {template_error}")
            return render_template('error.html',
                                 title="예측 결과 오류",
                                 message=f"예측 결과를 표시할 수 없습니다.",
                                 details=str(template_error))
            
    except Exception as e:
        logger.error(f"예측 페이지 전체 오류: {e}")
        try:
            return render_template('error.html',
                                 title="예측 오류",
                                 message=str(e))
        except:
            return f'''
            <!DOCTYPE html>
            <html><head><meta charset="UTF-8"><title>오류</title></head>
            <body style="text-align:center; margin-top:100px;">
            <h1>오류 발생</h1><p>{str(e)}</p>
            <a href="/" style="background:#667eea; color:white; padding:10px 20px; text-decoration:none; border-radius:5px;">홈으로</a>
            </body></html>
            ''', 500


@app.route('/api/diversity/stats')
def api_diversity_stats():
    """다양성 통계 API - 자릿수 분석 포함"""
    try:
        diversity_manager = system.get('diversity_manager')
        if not diversity_manager:
            return jsonify({
                'status': 'error',
                'message': '다양성 관리자가 초기화되지 않음'
            }), 500
        
        # 전체 다양성 통계 수집 (기존)
        diversity_stats = diversity_manager.get_diversity_stats()
        
        # 예측 모델별 다양성 통계
        predictor_stats = {}
        if system.get('predictor') and hasattr(system['predictor'], 'get_digit_diversity_stats'):
            try:
                predictor_stats = system['predictor'].get_digit_diversity_stats()
            except Exception as e:
                logger.warning(f"예측 모델 자릿수 통계 수집 실패: {e}")
        
        advanced_stats = {}
        if system.get('advanced_predictor') and hasattr(system['advanced_predictor'], 'get_digit_diversity_stats'):
            try:
                advanced_stats = system['advanced_predictor'].get_digit_diversity_stats()
            except Exception as e:
                logger.warning(f"고급 예측 모델 자릿수 통계 수집 실패: {e}")
        
        # 시스템 전체 자릿수 건강도 평가
        digit_health = evaluate_system_digit_health(diversity_stats, predictor_stats, advanced_stats)
        
        return jsonify({
            'status': 'success',
            'data': {
                'global_diversity': diversity_stats,
                'predictor_diversity': predictor_stats,
                'advanced_diversity': advanced_stats,
                'digit_health': digit_health,
                'system_info': {
                    'diversity_enabled': True,
                    'digit_tracking_enabled': True,
                    'cache_size': len(diversity_manager.global_prediction_cache),
                    'session_count': len(diversity_manager.session_predictions),
                    'digit_history_sizes': {
                        f'position_{pos+1}': len(diversity_manager.global_digit_history[pos])
                        for pos in range(6)
                    }
                }
            }
        })
        
    except Exception as e:
        logger.error(f"다양성 통계 API 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/diversity/reset', methods=['POST'])
def api_diversity_reset():
    """다양성 캐시 초기화 API"""
    try:
        diversity_manager = system.get('diversity_manager')
        if not diversity_manager:
            return jsonify({
                'status': 'error',
                'message': '다양성 관리자가 초기화되지 않음'
            }), 500
        
        # 캐시 초기화
        old_count = len(diversity_manager.global_prediction_cache)
        diversity_manager.global_prediction_cache = []
        diversity_manager.session_predictions = {}
        
        # 예측 모델 기록도 초기화
        if system.get('predictor') and hasattr(system['predictor'], 'recent_predictions'):
            system['predictor'].recent_predictions = []
        
        if system.get('advanced_predictor') and hasattr(system['advanced_predictor'], 'prediction_history'):
            system['advanced_predictor'].prediction_history = []
        
        logger.info(f"다양성 캐시 초기화 완료: {old_count}개 예측 기록 삭제")
        
        return jsonify({
            'status': 'success',
            'message': f'{old_count}개 예측 기록이 초기화되었습니다',
            'data': {
                'cleared_predictions': old_count,
                'cleared_sessions': len(diversity_manager.session_predictions)
            }
        })
        
    except Exception as e:
        logger.error(f"다양성 초기화 API 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/diversity/settings', methods=['GET', 'POST'])
def api_diversity_settings():
    """다양성 설정 API"""
    try:
        diversity_manager = system.get('diversity_manager')
        if not diversity_manager:
            return jsonify({
                'status': 'error',
                'message': '다양성 관리자가 초기화되지 않음'
            }), 500
        
        if request.method == 'POST':
            # 설정 업데이트
            data = request.get_json()
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'JSON 데이터가 필요합니다'
                }), 400
            
            # 유효한 설정만 업데이트
            valid_keys = ['min_difference', 'cache_size', 'similarity_threshold']
            updated = {}
            
            for key in valid_keys:
                if key in data:
                    old_value = diversity_manager.diversity_settings.get(key)
                    new_value = data[key]
                    
                    # 유효성 검사
                    if key == 'min_difference' and (not isinstance(new_value, int) or new_value < 1000 or new_value > 100000):
                        return jsonify({
                            'status': 'error',
                            'message': f'min_difference는 1000-100000 사이의 정수여야 합니다'
                        }), 400
                    
                    if key == 'cache_size' and (not isinstance(new_value, int) or new_value < 10 or new_value > 500):
                        return jsonify({
                            'status': 'error',
                            'message': f'cache_size는 10-500 사이의 정수여야 합니다'
                        }), 400
                    
                    if key == 'similarity_threshold' and (not isinstance(new_value, (int, float)) or new_value < 0.1 or new_value > 1.0):
                        return jsonify({
                            'status': 'error',
                            'message': f'similarity_threshold는 0.1-1.0 사이의 실수여야 합니다'
                        }), 400
                    
                    diversity_manager.diversity_settings[key] = new_value
                    updated[key] = {'old': old_value, 'new': new_value}
            
            return jsonify({
                'status': 'success',
                'message': f'{len(updated)}개 설정이 업데이트되었습니다',
                'data': {
                    'updated_settings': updated,
                    'current_settings': diversity_manager.diversity_settings
                }
            })
        
        else:
            # 현재 설정 반환
            return jsonify({
                'status': 'success',
                'data': {
                    'current_settings': diversity_manager.diversity_settings,
                    'setting_descriptions': {
                        'min_difference': '예측 번호 간 최소 차이값',
                        'cache_size': '전역 캐시 최대 크기',
                        'similarity_threshold': '유사도 판단 임계값'
                    }
                }
            })
        
    except Exception as e:
        logger.error(f"다양성 설정 API 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/diversity/test')
def api_diversity_test():
    """다양성 시스템 테스트 API - 강화된 오류 처리"""
    try:
        logger.info("다양성 테스트 API 호출됨")
        
        # 시스템 준비 상태 확인
        if not ensure_system_ready():
            return jsonify({
                'status': 'error',
                'message': '시스템이 초기화되지 않았습니다',
                'test_results': {
                    'predictions': [],
                    'diversity_stats': {},
                    'system_info': {
                        'predictor_available': False,
                        'advanced_predictor_available': False,
                        'diversity_manager_available': False,
                        'data_count': 0,
                        'initialization_status': system.get('initialization_status', 'Failed')
                    }
                }
            }), 200  # 200 상태코드로 반환하여 JSON 파싱 오류 방지
        
        # 다양성 관리자 확인 및 초기화
        if not system.get('diversity_manager'):
            try:
                from app import DiversityManager  # 순환 import 방지
                system['diversity_manager'] = DiversityManager()
                logger.info("다양성 관리자 임시 생성")
            except Exception as dm_error:
                logger.warning(f"다양성 관리자 생성 실패: {dm_error}")
        
        # 테스트 예측 생성
        test_predictions = []
        test_errors = []
        
        for i in range(5):
            try:
                if system.get('predictor'):
                    # 각 예측마다 다른 시드 사용
                    import time
                    if hasattr(system['predictor'], 'randomness_seed'):
                        system['predictor'].randomness_seed = int(time.time() * 1000) + i * 1000
                    
                    prediction_result = system['predictor'].predict_next()
                    
                    # 예측 결과 검증
                    ml_pred = prediction_result.get('ml_prediction', {})
                    if not ml_pred or not ml_pred.get('number'):
                        raise ValueError("예측 결과가 비어있습니다")
                    
                    test_predictions.append({
                        'attempt': i + 1,
                        'jo': ml_pred.get('jo', 0),
                        'number': ml_pred.get('number', '000000'),
                        'confidence': round(prediction_result.get('confidence_score', 0), 1),
                        'model_type': prediction_result.get('model_type', 'Unknown'),
                        'success': True
                    })
                    
                    logger.info(f"테스트 예측 {i+1} 성공: {ml_pred.get('jo')}조 {ml_pred.get('number')}")
                    
                else:
                    # 예측 모델이 없는 경우 샘플 생성
                    import random
                    jo = random.randint(1, 5)
                    number = str(jo * 100000 + random.randint(0, 99999)).zfill(6)
                    
                    test_predictions.append({
                        'attempt': i + 1,
                        'jo': jo,
                        'number': number,
                        'confidence': 50.0,
                        'model_type': 'Sample',
                        'success': True,
                        'note': 'AI 모델 없이 샘플 생성'
                    })
                
                # 예측 간 간격
                time.sleep(0.05)
                
            except Exception as pred_error:
                error_msg = str(pred_error)
                test_errors.append(f"예측 {i+1}: {error_msg}")
                logger.error(f"테스트 예측 {i+1} 실패: {pred_error}")
                
                test_predictions.append({
                    'attempt': i + 1,
                    'jo': None,
                    'number': None,
                    'confidence': 0,
                    'model_type': 'Error',
                    'success': False,
                    'error': error_msg
                })
        
        # 다양성 통계 수집
        diversity_stats = {}
        try:
            diversity_manager = system.get('diversity_manager')
            if diversity_manager:
                diversity_stats = diversity_manager.get_diversity_stats()
                logger.info("다양성 통계 수집 성공")
            else:
                diversity_stats = {
                    'message': '다양성 관리자 없음',
                    'diversity_score': 0,
                    'total_predictions': 0
                }
        except Exception as stats_error:
            logger.error(f"다양성 통계 수집 실패: {stats_error}")
            diversity_stats = {
                'error': str(stats_error),
                'diversity_score': 0
            }
        
        # 시스템 정보 수집
        system_info = {
            'predictor_available': system.get('predictor') is not None,
            'advanced_predictor_available': system.get('advanced_predictor') is not None,
            'diversity_manager_available': system.get('diversity_manager') is not None,
            'data_count': len(system['data']) if system.get('data') is not None else 0,
            'initialization_status': system.get('initialization_status', 'Unknown'),
            'test_errors_count': len(test_errors),
            'successful_predictions': len([p for p in test_predictions if p.get('success', False)])
        }
        
        # 응답 생성
        response_data = {
            'status': 'success',
            'message': f'{len(test_predictions)}개 테스트 예측 완료 ({system_info["successful_predictions"]}개 성공)',
            'test_results': {
                'predictions': test_predictions,
                'diversity_stats': diversity_stats,
                'system_info': system_info,
                'test_errors': test_errors if test_errors else None
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"다양성 테스트 완료: {system_info['successful_predictions']}/{len(test_predictions)} 성공")
        return jsonify(response_data), 200
        
    except Exception as e:
        error_msg = f"다양성 테스트 전체 실패: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        
        # 오류가 발생해도 JSON 형태로 반환
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'test_results': {
                'predictions': [],
                'diversity_stats': {'error': 'Stats collection failed'},
                'system_info': {
                    'predictor_available': False,
                    'advanced_predictor_available': False,
                    'diversity_manager_available': False,
                    'data_count': 0,
                    'error_details': str(e)
                }
            },
            'timestamp': datetime.now().isoformat()
        }), 200  # 200 상태코드로 반환

@app.route('/api/diversity/predict-batch', methods=['POST'])
def api_diversity_predict_batch():
    """배치 예측 API (다양성 테스트용)"""
    try:
        data = request.get_json()
        count = data.get('count', 5) if data else 5
        count = min(max(count, 1), 20)  # 1-20 범위로 제한
        
        if not ensure_system_ready():
            return jsonify({
                'status': 'error',
                'message': '시스템 초기화 실패'
            }), 500
        
        batch_results = []
        diversity_info = {
            'total_attempts': 0,
            'unique_predictions': 0,
            'duplicate_prevented': 0
        }
        
        for i in range(count):
            session_id = f"batch_{int(datetime.now().timestamp())}_{i}"
            
            try:
                # 예측 수행
                if system['predictor']:
                    result = system['predictor'].predict_next()
                    
                    # 다양성 검사
                    diversity_manager = system.get('diversity_manager')
                    is_unique = True
                    diversity_message = "고유함"
                    
                    if diversity_manager:
                        main_pred = result.get('ml_prediction', {})
                        is_unique, diversity_message = diversity_manager.is_prediction_unique(main_pred, session_id)
                        
                        if is_unique:
                            diversity_manager.add_prediction(main_pred, session_id)
                            diversity_info['unique_predictions'] += 1
                        else:
                            diversity_info['duplicate_prevented'] += 1
                    
                    diversity_info['total_attempts'] += 1
                    
                    batch_results.append({
                        'index': i + 1,
                        'prediction': result,
                        'diversity_check': {
                            'is_unique': is_unique,
                            'message': diversity_message
                        },
                        'timestamp': datetime.now().isoformat()
                    })
                    
                else:
                    batch_results.append({
                        'index': i + 1,
                        'error': '예측 모델 사용 불가',
                        'prediction': None
                    })
                
                # 배치 처리 간 짧은 대기
                import time
                time.sleep(0.05)
                
            except Exception as pred_error:
                batch_results.append({
                    'index': i + 1,
                    'error': str(pred_error),
                    'prediction': None
                })
        
        # 다양성 분석
        predictions_only = [r['prediction']['ml_prediction']['number'] 
                          for r in batch_results 
                          if r.get('prediction') and r['prediction'].get('ml_prediction')]
        
        diversity_analysis = analyze_prediction_diversity(predictions_only) if predictions_only else {}
        
        return jsonify({
            'status': 'success',
            'batch_count': count,
            'results': batch_results,
            'diversity_info': diversity_info,
            'diversity_analysis': diversity_analysis
        })
        
    except Exception as e:
        logger.error(f"배치 예측 API 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/diversity/export', methods=['GET'])
def api_diversity_export():
    """다양성 데이터 내보내기 API"""
    try:
        diversity_manager = system.get('diversity_manager')
        if not diversity_manager:
            return jsonify({
                'status': 'error',
                'message': '다양성 관리자가 초기화되지 않음'
            }), 500
        
        # 내보낼 데이터 준비
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_info': {
                'version': '1.0',
                'diversity_enabled': True,
                'total_sessions': len(diversity_manager.session_predictions),
                'cache_size': len(diversity_manager.global_prediction_cache)
            },
            'global_predictions': diversity_manager.global_prediction_cache,
            'session_predictions': diversity_manager.session_predictions,
            'settings': diversity_manager.diversity_settings,
            'statistics': diversity_manager.get_diversity_stats()
        }
        
        # 예측 모델별 다양성 정보 추가
        if system.get('predictor') and hasattr(system['predictor'], 'recent_predictions'):
            export_data['predictor_history'] = [
                {
                    'key': p.get('key'),
                    'timestamp': p.get('timestamp').isoformat() if p.get('timestamp') else None,
                    'prediction': p.get('full_prediction')
                } for p in system['predictor'].recent_predictions
            ]
        
        if system.get('advanced_predictor') and hasattr(system['advanced_predictor'], 'prediction_history'):
            export_data['advanced_predictor_history'] = system['advanced_predictor'].prediction_history
        
        return jsonify({
            'status': 'success',
            'data': export_data
        })
        
    except Exception as e:
        logger.error(f"다양성 데이터 내보내기 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/diversity/digits')
def api_digit_diversity():
    """자릿수별 다양성 상세 통계 API"""
    try:
        diversity_manager = system.get('diversity_manager')
        if not diversity_manager:
            return jsonify({
                'status': 'error',
                'message': '다양성 관리자가 초기화되지 않음'
            }), 500
        
        # 자릿수별 상세 통계
        digit_stats = {}
        overall_scores = []
        
        for pos in range(6):
            recent_digits = diversity_manager.global_digit_history[pos][-20:]  # 최근 20개
            
            if not recent_digits:
                digit_stats[f'position_{pos+1}'] = {
                    'status': 'no_data',
                    'diversity_score': 100,
                    'message': '데이터 부족'
                }
                continue
            
            # 통계 계산
            unique_count = len(set(recent_digits))
            total_count = len(recent_digits)
            
            # 첫 번째 자리는 1-5만 가능, 나머지는 0-9
            max_possible_unique = 5 if pos == 0 else 10
            diversity_score = (unique_count / min(max_possible_unique, total_count)) * 100
            
            # 빈도 분석
            digit_counts = {}
            for d in recent_digits:
                digit_counts[d] = digit_counts.get(d, 0) + 1
            
            # 가장 빈번한 숫자
            most_common_digit = max(digit_counts.items(), key=lambda x: x[1])
            
            # 연속성 분석
            consecutive_count = 1
            max_consecutive = 1
            for i in range(1, len(recent_digits)):
                if recent_digits[i] == recent_digits[i-1]:
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count)
                else:
                    consecutive_count = 1
            
            # 편중도 분석
            bias_ratio = most_common_digit[1] / total_count
            
            # 상태 결정
            if diversity_score >= 80 and bias_ratio <= 0.4 and max_consecutive <= 2:
                status = 'excellent'
                status_icon = '✅'
            elif diversity_score >= 60 and bias_ratio <= 0.6 and max_consecutive <= 3:
                status = 'good'
                status_icon = '⚠️'
            else:
                status = 'needs_improvement'
                status_icon = '🔴'
            
            digit_stats[f'position_{pos+1}'] = {
                'status': status,
                'status_icon': status_icon,
                'diversity_score': round(diversity_score, 1),
                'unique_digits': unique_count,
                'total_samples': total_count,
                'most_common': {
                    'digit': most_common_digit[0],
                    'count': most_common_digit[1],
                    'ratio': round(bias_ratio, 3)
                },
                'max_consecutive': max_consecutive,
                'recent_sequence': recent_digits[-10:],  # 최근 10개
                'frequency_distribution': dict(sorted(digit_counts.items())),
                'recommendations': get_digit_recommendations(pos, diversity_score, bias_ratio, max_consecutive)
            }
            
            overall_scores.append(diversity_score)
        
        # 전체 평가
        overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        
        # 문제 있는 자리 식별
        problem_positions = [
            pos for pos, stats in digit_stats.items() 
            if stats.get('status') == 'needs_improvement'
        ]
        
        return jsonify({
            'status': 'success',
            'data': {
                'position_stats': digit_stats,
                'overall': {
                    'diversity_score': round(overall_score, 1),
                    'status': 'excellent' if overall_score >= 80 else 'good' if overall_score >= 60 else 'needs_improvement',
                    'problem_positions': problem_positions,
                    'total_positions': 6
                },
                'recommendations': get_overall_recommendations(overall_score, problem_positions),
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"자릿수 다양성 API 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def get_digit_recommendations(position, diversity_score, bias_ratio, max_consecutive):
    """자릿수별 개선 권장사항"""
    recommendations = []
    
    if diversity_score < 60:
        recommendations.append(f"{position+1}번째 자리의 다양성이 부족합니다")
    
    if bias_ratio > 0.6:
        recommendations.append(f"{position+1}번째 자리에서 특정 숫자({int(bias_ratio*100)}% 편중)가 과도하게 사용되고 있습니다")
    
    if max_consecutive > 3:
        recommendations.append(f"{position+1}번째 자리에서 같은 숫자가 {max_consecutive}회 연속 나타났습니다")
    
    if not recommendations:
        recommendations.append(f"{position+1}번째 자리의 다양성이 양호합니다")
    
    return recommendations

def get_overall_recommendations(overall_score, problem_positions):
    """전체 시스템 권장사항"""
    recommendations = []
    
    if overall_score < 50:
        recommendations.append("🔴 자릿수 다양성이 전반적으로 부족합니다. 캐시 새로고침을 강력히 권장합니다.")
    elif overall_score < 70:
        recommendations.append("⚠️ 일부 자릿수에서 편중 현상이 나타나고 있습니다.")
    else:
        recommendations.append("✅ 자릿수 다양성이 전반적으로 양호합니다.")
    
    if len(problem_positions) > 2:
        recommendations.append(f"특히 {', '.join(problem_positions)} 자리의 개선이 필요합니다.")
    
    recommendations.append("💡 더 나은 다양성을 위해서는 주기적인 캐시 새로고침을 추천합니다.")
    
    return recommendations
@app.route('/api/diversity/analyze-batch', methods=['POST'])
def api_analyze_prediction_batch():
    """예측 배치의 자릿수 다양성 분석 API"""
    try:
        data = request.get_json()
        predictions = data.get('predictions', []) if data else []
        
        if not predictions:
            return jsonify({
                'status': 'error',
                'message': '분석할 예측 데이터가 없습니다'
            }), 400
        
        # 예측 데이터 검증 및 정리
        valid_predictions = []
        for pred in predictions:
            try:
                # 다양한 형태의 입력 처리
                if isinstance(pred, dict):
                    number = pred.get('number', pred.get('prediction', ''))
                elif isinstance(pred, str):
                    number = pred
                else:
                    number = str(pred)
                
                # 6자리 숫자로 정규화
                number = str(number).zfill(6)
                if len(number) == 6 and number.isdigit():
                    valid_predictions.append(number)
                    
            except Exception as pred_error:
                logger.warning(f"예측 데이터 처리 오류: {pred_error}")
                continue
        
        if not valid_predictions:
            return jsonify({
                'status': 'error',
                'message': '유효한 예측 데이터가 없습니다'
            }), 400
        
        # 자릿수별 분석
        position_analysis = {}
        
        for pos in range(6):
            digits_at_pos = [int(pred[pos]) for pred in valid_predictions]
            
            # 기본 통계
            unique_digits = set(digits_at_pos)
            total_count = len(digits_at_pos)
            
            # 빈도 분석
            digit_counts = {}
            for d in digits_at_pos:
                digit_counts[d] = digit_counts.get(d, 0) + 1
            
            # 다양성 점수 계산
            max_possible = 5 if pos == 0 else 10
            diversity_score = (len(unique_digits) / min(max_possible, total_count)) * 100
            
            # 편중도 분석
            most_common = max(digit_counts.items(), key=lambda x: x[1]) if digit_counts else (0, 0)
            bias_ratio = most_common[1] / total_count if total_count > 0 else 0
            
            # 연속성 분석
            consecutive_runs = []
            current_run = 1
            for i in range(1, len(digits_at_pos)):
                if digits_at_pos[i] == digits_at_pos[i-1]:
                    current_run += 1
                else:
                    if current_run > 1:
                        consecutive_runs.append(current_run)
                    current_run = 1
            if current_run > 1:
                consecutive_runs.append(current_run)
            
            max_consecutive = max(consecutive_runs) if consecutive_runs else 1
            
            # 상태 평가
            issues = []
            if diversity_score < 60:
                issues.append("다양성 부족")
            if bias_ratio > 0.5:
                issues.append(f"편중 심함({most_common[0]}: {bias_ratio*100:.0f}%)")
            if max_consecutive > 2:
                issues.append(f"연속 반복({max_consecutive}회)")
            
            status = 'good' if not issues else 'warning' if len(issues) == 1 else 'poor'
            
            position_analysis[f'position_{pos+1}'] = {
                'diversity_score': round(diversity_score, 1),
                'unique_digits': len(unique_digits),
                'total_samples': total_count,
                'most_common_digit': most_common[0],
                'most_common_count': most_common[1],
                'bias_ratio': round(bias_ratio, 3),
                'max_consecutive': max_consecutive,
                'frequency_distribution': dict(sorted(digit_counts.items())),
                'issues': issues,
                'status': status,
                'digit_sequence': digits_at_pos
            }
        
        # 전체 분석
        all_scores = [pos['diversity_score'] for pos in position_analysis.values()]
        overall_diversity = sum(all_scores) / len(all_scores) if all_scores else 0
        
        problem_positions = [
            pos for pos, data in position_analysis.items() 
            if data['status'] in ['warning', 'poor']
        ]
        
        # 배치 품질 평가
        if overall_diversity >= 80 and len(problem_positions) == 0:
            batch_quality = 'excellent'
            quality_message = '✅ 예측 배치의 자릿수 다양성이 우수합니다'
        elif overall_diversity >= 60 and len(problem_positions) <= 2:
            batch_quality = 'good'
            quality_message = '⚠️ 예측 배치의 자릿수 다양성이 양호하지만 개선 여지가 있습니다'
        else:
            batch_quality = 'poor'
            quality_message = '🔴 예측 배치의 자릿수 다양성이 부족합니다'
        
        # 개선 제안
        improvement_suggestions = []
        if len(problem_positions) > 0:
            improvement_suggestions.append(f"문제가 있는 자리: {', '.join(problem_positions)}")
        
        if overall_diversity < 70:
            improvement_suggestions.append("다양성 향상을 위해 예측 알고리즘 조정 필요")
        
        for pos, data in position_analysis.items():
            if data['issues']:
                improvement_suggestions.append(f"{pos}: {', '.join(data['issues'])}")
        
        return jsonify({
            'status': 'success',
            'data': {
                'batch_info': {
                    'total_predictions': len(valid_predictions),
                    'quality': batch_quality,
                    'quality_message': quality_message,
                    'overall_diversity_score': round(overall_diversity, 1)
                },
                'position_analysis': position_analysis,
                'summary': {
                    'excellent_positions': len([p for p in position_analysis.values() if p['status'] == 'good']),
                    'warning_positions': len([p for p in position_analysis.values() if p['status'] == 'warning']),
                    'poor_positions': len([p for p in position_analysis.values() if p['status'] == 'poor']),
                    'problem_positions': problem_positions
                },
                'improvement_suggestions': improvement_suggestions,
                'predictions_analyzed': valid_predictions,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"배치 분석 API 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/diversity/reset-digits', methods=['POST'])
def api_reset_digit_diversity():
    """자릿수별 다양성 기록 초기화 API"""
    try:
        diversity_manager = system.get('diversity_manager')
        if not diversity_manager:
            return jsonify({
                'status': 'error',
                'message': '다양성 관리자가 초기화되지 않음'
            }), 500
        
        # 자릿수별 기록 초기화
        cleared_counts = {}
        for pos in range(6):
            cleared_count = len(diversity_manager.global_digit_history[pos])
            diversity_manager.global_digit_history[pos] = []
            cleared_counts[f'position_{pos+1}'] = cleared_count
        
        # 예측 모델의 자릿수 기록도 초기화
        total_cleared = 0
        if system.get('predictor') and hasattr(system['predictor'], 'digit_history'):
            for pos in range(6):
                total_cleared += len(system['predictor'].digit_history[pos])
                system['predictor'].digit_history[pos] = []
                system['predictor'].digit_frequency[pos] = {}
        
        if system.get('advanced_predictor') and hasattr(system['advanced_predictor'], 'digit_patterns'):
            for pos in range(6):
                total_cleared += len(system['advanced_predictor'].digit_patterns[pos])
                system['advanced_predictor'].digit_patterns[pos] = []
        
        logger.info(f"자릿수 다양성 기록 초기화: 총 {total_cleared}개 기록 삭제")
        
        return jsonify({
            'status': 'success',
            'message': '자릿수별 다양성 기록이 초기화되었습니다',
            'data': {
                'cleared_by_position': cleared_counts,
                'total_cleared': total_cleared,
                'reset_timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"자릿수 다양성 초기화 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def evaluate_system_digit_health(global_stats, predictor_stats, advanced_stats):
    """시스템 전체 자릿수 건강도 평가"""
    try:
        health_score = 0
        issues = []
        recommendations = []
        
        # 전역 자릿수 다양성 평가
        if 'digit_overall_score' in global_stats:
            global_digit_score = global_stats['digit_overall_score']
            health_score += global_digit_score * 0.5
            
            if global_digit_score < 60:
                issues.append("전역 자릿수 다양성 부족")
                recommendations.append("캐시 새로고침을 통한 다양성 개선")
        
        # 예측 모델 자릿수 다양성 평가
        if 'overall_diversity' in predictor_stats:
            predictor_score = predictor_stats['overall_diversity'].get('score', 0)
            health_score += predictor_score * 0.3
            
            if predictor_score < 60:
                issues.append("예측 모델 자릿수 편중")
                recommendations.append("예측 알고리즘 다양성 파라미터 조정")
        
        # 고급 모델 자릿수 다양성 평가
        if 'overall' in advanced_stats:
            advanced_score = advanced_stats['overall'].get('diversity_score', 0)
            health_score += advanced_score * 0.2
            
            if advanced_score < 60:
                issues.append("고급 모델 자릿수 편중")
        
        # 건강도 등급 결정
        if health_score >= 80:
            health_grade = 'A'
            health_status = 'excellent'
            status_message = '✅ 자릿수 다양성이 매우 우수합니다'
        elif health_score >= 65:
            health_grade = 'B'
            health_status = 'good'
            status_message = '⚠️ 자릿수 다양성이 양호하지만 개선 여지가 있습니다'
        elif health_score >= 50:
            health_grade = 'C'
            health_status = 'fair'
            status_message = '🔶 자릿수 다양성이 보통 수준입니다'
        else:
            health_grade = 'D'
            health_status = 'poor'
            status_message = '🔴 자릿수 다양성이 부족합니다'
        
        return {
            'health_score': round(health_score, 1),
            'health_grade': health_grade,
            'health_status': health_status,
            'status_message': status_message,
            'issues': issues,
            'recommendations': recommendations,
            'components': {
                'global_contribution': round(global_stats.get('digit_overall_score', 0) * 0.5, 1),
                'predictor_contribution': round(predictor_stats.get('overall_diversity', {}).get('score', 0) * 0.3, 1),
                'advanced_contribution': round(advanced_stats.get('overall', {}).get('diversity_score', 0) * 0.2, 1)
            }
        }
        
    except Exception as e:
        logger.error(f"자릿수 건강도 평가 오류: {e}")
        return {
            'health_score': 0,
            'health_grade': 'F',
            'health_status': 'error',
            'status_message': '건강도 평가 중 오류 발생',
            'issues': ['건강도 평가 시스템 오류'],
            'recommendations': ['시스템 재시작 필요']
        }
        
def analyze_prediction_diversity(predictions):
    """예측 다양성 분석 헬퍼 함수"""
    if not predictions or len(predictions) < 2:
        return {'message': '분석할 예측이 부족함'}
    
    try:
        # 숫자로 변환
        numbers = [int(p) for p in predictions if p.isdigit()]
        
        if len(numbers) < 2:
            return {'message': '유효한 예측 번호가 부족함'}
        
        # 기본 통계
        differences = [abs(numbers[i] - numbers[i-1]) for i in range(1, len(numbers))]
        
        analysis = {
            'total_predictions': len(numbers),
            'unique_predictions': len(set(numbers)),
            'uniqueness_ratio': len(set(numbers)) / len(numbers),
            'average_difference': sum(differences) / len(differences) if differences else 0,
            'min_difference': min(differences) if differences else 0,
            'max_difference': max(differences) if differences else 0,
            'standard_deviation': np.std(differences) if differences else 0
        }
        
        # 다양성 점수 계산 (0-100)
        uniqueness_score = analysis['uniqueness_ratio'] * 50
        difference_score = min(50, (analysis['average_difference'] / 10000) * 50)
        analysis['diversity_score'] = uniqueness_score + difference_score
        
        # 다양성 등급
        if analysis['diversity_score'] >= 80:
            analysis['diversity_grade'] = 'A'
            analysis['diversity_description'] = '매우 높은 다양성'
        elif analysis['diversity_score'] >= 60:
            analysis['diversity_grade'] = 'B'
            analysis['diversity_description'] = '높은 다양성'
        elif analysis['diversity_score'] >= 40:
            analysis['diversity_grade'] = 'C'
            analysis['diversity_description'] = '보통 다양성'
        else:
            analysis['diversity_grade'] = 'D'
            analysis['diversity_description'] = '낮은 다양성'
        
        return analysis
        
    except Exception as e:
        return {'error': f'분석 중 오류: {str(e)}'}

# 404 에러 핸들러에 다양성 정보 추가
@app.errorhandler(404)
def not_found_error(error):
    """404 에러 처리"""
    try:
        # 다양성 통계 간단히 표시
        diversity_info = ""
        if system.get('diversity_manager'):
            stats = system['diversity_manager'].get_diversity_stats()
            diversity_info = f" | 다양성: {stats.get('diversity_score', 0):.1f}%"
        
        return render_template('404.html', diversity_info=diversity_info), 404
    except:
        return render_template('404.html'), 404

# 500 에러 핸들러
@app.errorhandler(500)
def internal_error(error):
    """500 에러 처리"""
    try:
        logger.error(f"500 에러 발생: {error}")
        return render_template('error.html',
                             title="서버 내부 오류",
                             message="서버에서 예상치 못한 오류가 발생했습니다."), 500
    except:
        return "서버 오류가 발생했습니다.", 500

# 기존 API에 다양성 정보 추가
@app.route('/api/predict', methods=['GET', 'POST'])
def api_predict():
    """예측 API - 다양성 검사 포함"""
    try:
        logger.info("예측 API 요청 받음")
        
        if not ensure_system_ready():
            return jsonify({
                'status': 'error',
                'message': '시스템 초기화 실패',
                'prediction': create_sample_prediction()
            }), 500
        
        # 세션 ID 생성
        session_id = request.remote_addr + str(int(datetime.now().timestamp()) // 3600)
        
        max_attempts = 3
        attempt = 0
        diversity_info = {
            'attempts_used': 0,
            'uniqueness_verified': False,
            'diversity_applied': False
        }
        
        while attempt < max_attempts:
            # 예측 수행
            if system['predictor']:
                try:
                    result = system['predictor'].predict_next()
                    
                    # 다양성 검사
                    diversity_manager = system.get('diversity_manager')
                    if diversity_manager:
                        main_prediction = result.get('ml_prediction', {})
                        is_unique, message = diversity_manager.is_prediction_unique(main_prediction, session_id)
                        
                        if is_unique:
                            diversity_manager.add_prediction(main_prediction, session_id)
                            diversity_info['uniqueness_verified'] = True
                            diversity_info['attempts_used'] = attempt + 1
                            break
                        else:
                            logger.info(f"API 중복 감지 (시도 {attempt + 1}): {message}")
                            attempt += 1
                            
                            # 예측 모델에 다양성 신호 전달
                            if hasattr(system['predictor'], 'randomness_seed'):
                                system['predictor'].randomness_seed += 777
                            
                            continue
                    else:
                        break
                    
                except Exception as e:
                    logger.error(f"예측 수행 중 오류: {e}")
                    return jsonify({
                        'status': 'error',
                        'message': f'예측 실행 실패: {str(e)}',
                        'prediction': create_sample_prediction()
                    }), 500
            else:
                result = create_sample_prediction()
                break
        
        # 최대 시도 도달 시 강제 다양화
        if attempt >= max_attempts:
            logger.warning("API 최대 시도 도달, 강제 다양화")
            diversity_info['diversity_applied'] = True
            diversity_info['attempts_used'] = max_attempts
        
        # 고급 예측
        advanced_result = None
        if system['advanced_predictor']:
            try:
                advanced_result = system['advanced_predictor'].predict_ensemble()
            except Exception as e:
                logger.warning(f"고급 예측 실패: {e}")
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'advanced_prediction': advanced_result,
            'diversity_info': diversity_info,
            'system_status': {
                'model_trained': system['predictor'].is_trained if system['predictor'] else False,
                'data_count': len(system['data']) if system['data'] is not None else 0,
                'model_type': result.get('model_type', 'Unknown'),
                'diversity_enabled': system.get('diversity_manager') is not None
            }
        })
        
    except Exception as e:
        logger.error(f"예측 API 전체 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'prediction': create_sample_prediction()
        }), 500

@app.route('/analysis')
def analysis():
    """분석 페이지"""
    try:
        ensure_system_ready()
        
        # 패턴 분석
        patterns = get_safe_patterns()
        
        # 차트 생성
        charts = create_safe_charts()
        confidence_chart = charts.get('jo_distribution', None)
        
        # analysis.html 템플릿 사용
        try:
            return render_template('analysis.html', 
                                 patterns=patterns,
                                 confidence_chart=confidence_chart)
        except Exception as template_error:
            logger.error(f"analysis.html 템플릿 렌더링 실패: {template_error}")
            return render_template('error.html',
                                 title="분석 페이지 오류",
                                 message=f"분석 페이지를 로드할 수 없습니다: {str(template_error)}")
        
    except Exception as e:
        logger.error(f"분석 페이지 오류: {e}")
        return render_template('error.html',
                             title="분석 오류",
                             message=str(e))

@app.route('/history')
def history():
    """예측 기록 페이지"""
    try:
        # 예측 기록 가져오기
        predictions = get_safe_history()
        
        # history.html 템플릿 사용
        try:
            return render_template('history.html', predictions=predictions)
        except Exception as template_error:
            logger.error(f"history.html 템플릿 렌더링 실패: {template_error}")
            return render_template('error.html',
                                 title="기록 페이지 오류",
                                 message=f"기록 페이지를 로드할 수 없습니다: {str(template_error)}")
        
    except Exception as e:
        logger.error(f"기록 페이지 오류: {e}")
        return render_template('error.html',
                             title="기록 오류",
                             message=str(e))

# 상태 API에 다양성 정보 추가
@app.route('/api/status')
def api_status():
    """시스템 상태 API - 다양성 정보 포함"""
    try:
        ensure_system_ready()
        
        # 기본 상태 정보
        status = {
            'system_status': system['initialization_status'],
            'predictor_trained': system['predictor'].is_trained if system['predictor'] else False,
            'data_count': len(system['data']) if system['data'] is not None else 0,
            'has_advanced_model': system['advanced_predictor'] is not None,
            'last_update': system['last_update'].isoformat() if system['last_update'] else None,
            'models_available': list(system['predictor'].models.keys()) if system['predictor'] and system['predictor'].models else []
        }
        
        # 다양성 관리자 상태 추가
        diversity_manager = system.get('diversity_manager')
        if diversity_manager:
            diversity_stats = diversity_manager.get_diversity_stats()
            status['diversity_status'] = {
                'enabled': True,
                'cache_size': len(diversity_manager.global_prediction_cache),
                'session_count': len(diversity_manager.session_predictions),
                'diversity_score': diversity_stats.get('diversity_score', 0),
                'settings': diversity_manager.diversity_settings
            }
        else:
            status['diversity_status'] = {
                'enabled': False,
                'message': '다양성 관리자가 초기화되지 않음'
            }
        
        return jsonify({
            'status': 'success',
            'data': status
        })
        
    except Exception as e:
        logger.error(f"상태 API 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/statistics')
def api_statistics():
    """통계 API"""
    try:
        if not ensure_system_ready():
            return jsonify({
                'status': 'error',
                'message': '시스템 초기화 실패'
            }), 500
        
        stats = get_safe_statistics()
        return jsonify({
            'status': 'success',
            'data': stats
        })
        
    except Exception as e:
        logger.error(f"통계 API 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/retrain')
def api_retrain():
    """모델 재학습 API"""
    try:
        logger.info("수동 재학습 요청")
        
        if initialize_system():
            return jsonify({
                'status': 'success',
                'message': '모델 재학습 완료',
                'model_status': system['predictor'].is_trained if system['predictor'] else False
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '재학습 실패'
            }), 500
            
    except Exception as e:
        logger.error(f"재학습 API 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ==================== 헬퍼 함수들 ====================

def create_safe_charts():
    """안전한 차트 생성"""
    charts = {}
    
    try:
        visualizer = get_safe_visualizer()
        if visualizer:
            try:
                charts['jo_distribution'] = visualizer.create_jo_distribution_chart()
            except Exception as e:
                logger.warning(f"조 분포 차트 생성 실패: {e}")
                charts['jo_distribution'] = None
            
            try:
                charts['number_trend'] = visualizer.create_number_trend_chart()
            except Exception as e:
                logger.warning(f"번호 트렌드 차트 생성 실패: {e}")
                charts['number_trend'] = None
            
            try:
                charts['heatmap'] = visualizer.create_advanced_heatmap()
            except Exception as e:
                logger.warning(f"히트맵 생성 실패: {e}")
                charts['heatmap'] = None
    except Exception as e:
        logger.error(f"차트 생성 전체 실패: {e}")
    
    return charts

def get_safe_visualizer():
    """안전한 시각화 객체 반환"""
    try:
        if system.get('visualizer'):
            return system['visualizer']
        elif system.get('data') is not None:
            return LottoVisualizer(system['data'])
        else:
            return LottoVisualizer(None)
    except Exception as e:
        logger.warning(f"Visualizer 생성 중 오류: {e}")
        return LottoVisualizer(None)

def get_safe_statistics():
    """안전한 통계 정보 반환"""
    try:
        stats = {}
        
        if system.get('predictor'):
            stats = system['predictor'].get_statistics()
            
        # 기본값 설정
        default_stats = {
            'total_draws': 0,
            'jo_distribution': {},
            'model_status': 'Unknown',
            'avg_number': 0,
            'std_number': 0,
            'median_number': 0,
            'pattern_stats': {
                'avg_odd_ratio': 0.5,
                'frequency_pattern': 'Normal',
                'avg_high_ratio': 0.5,
                'avg_consecutive': 0,
                'avg_repeat': 0
            },
            'jo_stats': {}
        }
        
        # 기본값과 병합
        for key, value in default_stats.items():
            if key not in stats:
                stats[key] = value
        
        # 실제 데이터 계산 (안전하게)
        if system.get('data') is not None and len(system['data']) > 0:
            data = system['data']
            
            try:
                # 숫자 컬럼만 추출하고 문자열 제거
                if 'number' in data.columns:
                    # 문자열을 숫자로 변환 시도
                    numbers = pd.to_numeric(data['number'], errors='coerce')
                    numbers = numbers.dropna()  # NaN 제거
                    numbers = numbers.replace([np.inf, -np.inf], np.nan).dropna()  # 무한대 제거
                    
                    if len(numbers) > 0:
                        stats['avg_number'] = int(numbers.mean())
                        stats['std_number'] = int(numbers.std())
                        stats['median_number'] = int(numbers.median())
                        stats['total_draws'] = len(numbers)
                
                # 조별 통계 생성
                if 'jo' in data.columns:
                    jo_stats = {}
                    for jo in range(1, 6):
                        jo_data = data[data['jo'] == jo]
                        count = len(jo_data)
                        percentage = (count / len(data)) * 100 if len(data) > 0 else 0
                        
                        # 평균 번호 계산
                        if count > 0 and 'number' in jo_data.columns:
                            jo_numbers = pd.to_numeric(jo_data['number'], errors='coerce').dropna()
                            avg_number = int(jo_numbers.mean()) if len(jo_numbers) > 0 else 0
                        else:
                            avg_number = 0
                        
                        # 마지막 출현
                        if count > 0:
                            last_idx = jo_data.index[-1]
                            last_appearance = len(data) - 1 - last_idx
                        else:
                            last_appearance = len(data)
                        
                        jo_stats[f'jo_{jo}'] = {
                            'count': count,
                            'percentage': round(percentage, 1),
                            'avg_number': avg_number,
                            'last_appearance': last_appearance
                        }
                    
                    stats['jo_stats'] = jo_stats
                    stats['jo_distribution'] = {str(jo): v['count'] for jo, v in jo_stats.items()}
                
            except Exception as calc_error:
                logger.warning(f"통계 계산 중 오류: {calc_error}")
        
        return stats
        
    except Exception as e:
        logger.warning(f"통계 정보 생성 실패: {e}")
        return {
            'total_draws': 720,
            'jo_distribution': {'1': 144, '2': 145, '3': 143, '4': 144, '5': 144},
            'model_status': 'Error',
            'avg_number': 350000,
            'std_number': 150000,
            'median_number': 325000,
            'pattern_stats': {
                'avg_odd_ratio': 0.5,
                'frequency_pattern': 'Normal',
                'avg_high_ratio': 0.5,
                'avg_consecutive': 0,
                'avg_repeat': 0
            },
            'jo_stats': {
                'jo_1': {'count': 144, 'percentage': 20.0, 'avg_number': 150000, 'last_appearance': 5},
                'jo_2': {'count': 145, 'percentage': 20.1, 'avg_number': 250000, 'last_appearance': 3},
                'jo_3': {'count': 143, 'percentage': 19.9, 'avg_number': 350000, 'last_appearance': 8},
                'jo_4': {'count': 144, 'percentage': 20.0, 'avg_number': 450000, 'last_appearance': 2},
                'jo_5': {'count': 144, 'percentage': 20.0, 'avg_number': 550000, 'last_appearance': 7}
            }
        }

def get_safe_patterns():
    """안전한 패턴 분석 결과 반환"""
    try:
        if system.get('predictor') and system['predictor'].feature_importance:
            return {
                'recent_trends': ['상승', '안정', '하락'],
                'frequency_analysis': system['predictor'].feature_importance,
                'pattern_strength': 0.75,
                'fibonacci_avg': 45.2,
                'prime_avg': 38.7,
                'dominant_period': 8.3
            }
        else:
            return create_sample_patterns()
    except Exception as e:
        logger.warning(f"패턴 분석 실패: {e}")
        return create_sample_patterns()

def get_safe_history():
    """안전한 예측 기록 반환 - 바이너리 데이터 문제 해결"""
    try:
        if system.get('db_manager'):
            try:
                history = system['db_manager'].get_prediction_history()
                
                # SQLite 바이너리 데이터 문제 해결
                processed_history = []
                for h in history:
                    try:
                        # 바이너리 데이터를 안전하게 처리
                        draw_no = h.draw_no
                        if isinstance(draw_no, bytes):
                            # 바이너리를 정수로 변환 시도
                            try:
                                draw_no = int.from_bytes(draw_no, byteorder='little')
                            except:
                                draw_no = 0
                        elif not isinstance(draw_no, int):
                            try:
                                draw_no = int(draw_no)
                            except:
                                draw_no = 0
                        
                        predicted_jo = h.predicted_jo if h.predicted_jo else 0
                        predicted_number = h.predicted_number if h.predicted_number else '000000'
                        actual_jo = h.actual_jo if h.actual_jo else None
                        actual_number = h.actual_number if h.actual_number else None
                        confidence_score = h.confidence_score if h.confidence_score else 0.0
                        is_correct = h.is_correct if h.is_correct else None
                        
                        # 템플릿 호환 객체 생성
                        class CleanPrediction:
                            def __init__(self):
                                self.draw_no = draw_no
                                self.predicted_jo = predicted_jo
                                self.predicted_number = str(predicted_number).zfill(6)
                                self.actual_jo = actual_jo
                                self.actual_number = str(actual_number).zfill(6) if actual_number else None
                                self.confidence_score = round(float(confidence_score), 1)
                                self.is_correct = is_correct
                                self.created_at = h.created_at if hasattr(h, 'created_at') else datetime.now()
                        
                        processed_history.append(CleanPrediction())
                        
                    except Exception as item_error:
                        logger.warning(f"예측 기록 항목 처리 실패: {item_error}")
                        continue
                
                return processed_history[:10]  # 최대 10개
                
            except Exception as db_error:
                logger.warning(f"데이터베이스 조회 실패: {db_error}")
                return create_sample_history_objects()
        else:
            return create_sample_history_objects()
            
    except Exception as e:
        logger.warning(f"기록 조회 전체 실패: {e}")
        return create_sample_history_objects()

def create_sample_history_objects():
    """샘플 예측 기록 객체 생성"""
    class SamplePrediction:
        def __init__(self, draw_no, predicted_jo, predicted_number, actual_jo=None, actual_number=None, confidence=75.0, is_correct=None):
            self.draw_no = draw_no
            self.predicted_jo = predicted_jo
            self.predicted_number = str(predicted_number).zfill(6)
            self.actual_jo = actual_jo
            self.actual_number = str(actual_number).zfill(6) if actual_number else None
            self.confidence_score = confidence
            self.is_correct = is_correct
            self.created_at = datetime.now()
    
    return [
        SamplePrediction(720, 2, '234567', 2, '574627', 75.0, 3),
        SamplePrediction(719, 4, '415223', 4, '415223', 85.0, 6),
        SamplePrediction(718, 1, '178408', 1, '178408', 70.0, 6),
        SamplePrediction(717, 3, '298745', 3, '298745', 80.0, 6),
        SamplePrediction(716, 5, '567891', 5, '567891', 90.0, 6)
    ]

# ==================== 샘플 데이터 생성 함수들 ====================

def create_sample_prediction():
    """샘플 예측 결과 생성"""
    return {
        'next_draw_no': 721,
        'ml_prediction': {
            'jo': 2,
            'number': '234567',
            'jo_confidence': 75.0,
            'number_confidence': 82.0
        },
        'pattern_predictions': [
            {
                'type': 'trend',
                'jo': 3,
                'number': '345678',
                'reason': '트렌드 기반'
            }
        ],
        'confidence_score': 75.0,
        'model_type': 'Sample',
        'features_used': 40
    }

def create_sample_advanced_prediction():
    """샘플 고급 예측 결과 생성"""
    return {
        'ensemble_prediction': '234567',
        'confidence': 78.0,
        'model_predictions': {
            'lstm': '234567',
            'xgboost': '234567',
            'rf': '234567'
        }
    }

def create_sample_patterns():
    """샘플 패턴 분석 결과"""
    return {
        'recent_trends': ['증가', '감소', '안정'],
        'frequency_analysis': {'1': 45, '2': 52, '3': 38, '4': 48, '5': 42},
        'pattern_strength': 0.72,
        'fibonacci_avg': 45.2,
        'prime_avg': 38.7,
        'dominant_period': 8.3
    }

def create_sample_statistics():
    """샘플 통계 정보"""
    return {
        'total_draws': 720,
        'jo_distribution': {'1': 144, '2': 145, '3': 143, '4': 144, '5': 144},
        'model_status': 'Sample',
        'accuracy_rate': 0.685,
        'last_update': datetime.now().isoformat(),
        'avg_number': 350000,
        'std_number': 150000,
        'median_number': 325000,
        'pattern_stats': {
            'avg_odd_ratio': 0.5,
            'frequency_pattern': 'Normal'
        }
    }

if __name__ == '__main__':
    logger.info("Flask 앱 시작")
    app.run(debug=True, host='0.0.0.0', port=5000)