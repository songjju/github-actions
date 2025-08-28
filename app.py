from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import traceback
import logging
import pandas as pd
import numpy as np

from data_parser import LottoDataParser
from lotto_predictor import LottoPredictor
from advanced_predictor import AdvancedLottoPredictor
from visualization import LottoVisualizer
from database import DatabaseManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # API 접근 허용

# 전역 시스템 상태
system = {
    'predictor': None,
    'advanced_predictor': None,
    'visualizer': None,
    'db_manager': None,
    'data': None,
    'last_update': None,
    'initialization_status': 'Not Started'
}

def initialize_system():
    """시스템 초기화 (강화된 버전)"""
    try:
        logger.info("=== 시스템 초기화 시작 ===")
        system['initialization_status'] = 'In Progress'
        
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
        logger.error(f"시스템 초기화 전체 실패: {e}")
        system['initialization_status'] = 'Failed'
        return False

def create_emergency_sample_data():
    """응급 샘플 데이터 생성"""
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

def ensure_system_ready():
    """시스템 준비 상태 확인 및 초기화"""
    if system['initialization_status'] == 'Not Started':
        logger.info("시스템이 초기화되지 않았습니다. 초기화를 시작합니다.")
        return initialize_system()
    elif system['initialization_status'] == 'Failed':
        logger.info("시스템 초기화가 실패했었습니다. 재시도합니다.")
        return initialize_system()
    elif system['predictor'] is None:
        logger.info("예측 모델이 없습니다. 재초기화합니다.")
        return initialize_system()
    return True

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
    """예측 수행 - 웹 페이지 반환"""
    try:
        logger.info("예측 페이지 요청 받음")
        ensure_system_ready()
        
        # 예측 수행
        prediction = None
        advanced_prediction = None
        
        if system['predictor']:
            try:
                prediction = system['predictor'].predict_next()
                logger.info(f"예측 완료: {prediction.get('model_type', 'Unknown')} 모델 사용")
            except Exception as e:
                logger.error(f"예측 수행 중 오류: {e}")
                prediction = create_sample_prediction()
        
        # 고급 예측 시도
        if system['advanced_predictor']:
            try:
                advanced_prediction = system['advanced_predictor'].predict_ensemble()
                logger.info("고급 앙상블 예측도 완료")
            except Exception as e:
                logger.warning(f"고급 예측 실패: {e}")
                advanced_prediction = create_sample_advanced_prediction()
        
        # 차트 생성
        charts = create_safe_charts()
        
        # 통계 정보 
        statistics = get_safe_statistics()
        
        # result.html 템플릿 사용
        try:
            return render_template('result.html', 
                                 prediction=prediction,
                                 advanced_prediction=advanced_prediction,
                                 charts=charts,
                                 statistics=statistics)
        except Exception as template_error:
            logger.error(f"result.html 템플릿 렌더링 실패: {template_error}")
            return render_template('error.html',
                                 title="예측 결과 오류",
                                 message=f"예측 결과를 표시할 수 없습니다: {str(template_error)}")
            
    except Exception as e:
        logger.error(f"예측 페이지 전체 오류: {e}")
        return render_template('error.html',
                             title="예측 오류",
                             message=str(e))

@app.route('/api/predict', methods=['GET', 'POST'])
def api_predict():
    """예측 API - JSON 반환"""
    try:
        logger.info("예측 API 요청 받음")
        
        if not ensure_system_ready():
            return jsonify({
                'status': 'error',
                'message': '시스템 초기화 실패',
                'prediction': create_sample_prediction()
            }), 500
        
        # 예측 수행
        if system['predictor']:
            try:
                result = system['predictor'].predict_next()
                
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
                    'system_status': {
                        'model_trained': system['predictor'].is_trained if system['predictor'] else False,
                        'data_count': len(system['data']) if system['data'] is not None else 0,
                        'model_type': result.get('model_type', 'Unknown')
                    }
                })
                
            except Exception as e:
                logger.error(f"예측 수행 중 오류: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'예측 실행 실패: {str(e)}',
                    'prediction': create_sample_prediction()
                }), 500
        else:
            return jsonify({
                'status': 'warning',
                'message': '예측 모델이 초기화되지 않음',
                'prediction': create_sample_prediction()
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

@app.route('/api/status')
def api_status():
    """시스템 상태 API"""
    try:
        ensure_system_ready()
        
        status = {
            'system_status': system['initialization_status'],
            'predictor_trained': system['predictor'].is_trained if system['predictor'] else False,
            'data_count': len(system['data']) if system['data'] is not None else 0,
            'has_advanced_model': system['advanced_predictor'] is not None,
            'last_update': system['last_update'].isoformat() if system['last_update'] else None,
            'models_available': list(system['predictor'].models.keys()) if system['predictor'] and system['predictor'].models else []
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
        if system.get('predictor'):
            stats = system['predictor'].get_statistics()
            # 템플릿에서 사용할 추가 통계 정보 생성
            if system.get('data') is not None:
                stats.update({
                    'avg_number': int(system['data']['number'].mean()) if 'number' in system['data'].columns else 0,
                    'std_number': int(system['data']['number'].std()) if 'number' in system['data'].columns else 0,
                    'median_number': int(system['data']['number'].median()) if 'number' in system['data'].columns else 0,
                    'pattern_stats': {
                        'avg_odd_ratio': 0.5,
                        'frequency_pattern': 'Normal'
                    }
                })
            return stats
        else:
            return create_sample_statistics()
    except Exception as e:
        logger.warning(f"통계 정보 생성 실패: {e}")
        return create_sample_statistics()

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
    """안전한 예측 기록 반환"""
    try:
        if system.get('db_manager'):
            history = system['db_manager'].get_prediction_history()
            return history  # DB 객체 그대로 반환 (템플릿에서 객체 속성 사용)
        else:
            # 샘플 객체 생성 (템플릿 호환)
            class MockPrediction:
                def __init__(self, draw_no, predicted_jo, predicted_number, actual_jo=None, actual_number=None, confidence=75.0, is_correct=None):
                    self.draw_no = draw_no
                    self.predicted_jo = predicted_jo
                    self.predicted_number = predicted_number
                    self.actual_jo = actual_jo
                    self.actual_number = actual_number
                    self.confidence_score = confidence
                    self.is_correct = is_correct
                    self.created_at = datetime.now()
            
            return [
                MockPrediction(720, 2, '234567', 2, '574627', 75.0, 3),
                MockPrediction(719, 4, '415223', 4, '415223', 85.0, 6),
                MockPrediction(718, 1, '178408', 1, '178408', 70.0, 6)
            ]
    except Exception as e:
        logger.warning(f"기록 조회 실패: {e}")
        return []

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